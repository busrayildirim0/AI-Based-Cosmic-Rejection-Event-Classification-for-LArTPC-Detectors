import os
import glob
import csv
import zlib
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix, brier_score_loss, f1_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler


# ==============================================================================
# 0. DONANIM VE ÇIKTI DİZİNİ AYARLARI
# ==============================================================================
DATA_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(DATA_DIR, "training_outputs_v6_groupshuffle")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 85)
print("🚀 LArTPC V6.0 TAM DEDEKTÖR HİBRİT (WIRE + TABULAR) - GROUP SHUFFLE SPLIT")
print(f"📁 Çalışma Dizini : {DATA_DIR}")
print(f"📁 Çıktı Klasörü  : {OUTPUT_DIR}")
print(f"⚡ Hesaplama Donanımı: {DEVICE}")
if torch.cuda.is_available():
    print(f"   • Donanım Adı    : {torch.cuda.get_device_name(0)}")
print("=" * 85)

# ==============================================================================
# 1. SEYREK GÖRÜNTÜ ÇÖZÜCÜ (SPARSE BINARY -> FULL-VIEW 3x504x864)
# ==============================================================================
def kayit_coz(buf):
    sihir = buf[:4]
    event_id, run, nduzlem = np.frombuffer(buf, '<u4', 3, 4)
    pos = 16
    if sihir == b'LTV5':
        acik, sikisik = np.frombuffer(buf, '<u4', 2, 16)
        buf, pos = zlib.decompress(buf[24:24 + int(sikisik)]), 0
    duzlemler = []
    for _ in range(int(nduzlem)):
        duzlem, ntel, nsatir, npix, bayrak = np.frombuffer(buf, '<u4', 5, pos)
        pos += 20
        n = int(npix)
        tel = np.frombuffer(buf, '<u2', n, pos); pos += 2 * n
        satir = np.frombuffer(buf, '<u2', n, pos); pos += 2 * n
        q = np.frombuffer(buf, '<f4', n, pos); pos += 4 * n
        etiket = np.frombuffer(buf, '<u1', n, pos); pos += n
        if sihir in (b'LTV4', b'LTV5'):
            pos += 2 * n
        if bayrak & 1:
            pos += 4 * n
        duzlemler.append({
            'ntel': int(ntel),
            'nsatir': int(nsatir),
            'tel': tel,
            'satir': satir,
            'q': q
        })
    return duzlemler

def cnn_girdisi(duzlemler, satir_kat=2, genislik=864):
    yukseklik = duzlemler[2]['nsatir'] // satir_kat  # 504
    x = np.zeros((3, yukseklik, genislik), dtype=np.float32)
    for c, p in enumerate(duzlemler):
        tel_kat = -(-p['ntel'] // genislik)
        r = p['satir'].astype(int) // satir_kat
        w = p['tel'].astype(int) // tel_kat
        m = (r < yukseklik) & (w < genislik)
        np.add.at(x[c], (r[m], w[m]), np.clip(p['q'][m], 0, None))
    return np.log10(1.0 + x / 1000.0)

# ==============================================================================
# 2. SIZINTISIZ TABLOSAL ÖZELLİK TÜRETİCİ
# ==============================================================================
def extract_reco_tabular_features(row):
    q_u = float(row.get('reco_Q_U', 0.0))
    q_v = float(row.get('reco_Q_V', 0.0))
    q_y = float(row.get('reco_Q_Y', 0.0))
    npix_u = float(row.get('reco_npix_U', 0.0))
    npix_v = float(row.get('reco_npix_V', 0.0))
    npix_y = float(row.get('reco_npix_Y', 0.0))

    q_tot = q_u + q_v + q_y
    npix_tot = npix_u + npix_v + npix_y

    feats = [
        np.log10(max(1.0, q_u)),
        np.log10(max(1.0, q_v)),
        np.log10(max(1.0, q_y)),
        np.log10(max(1.0, q_tot)),
        np.log10(max(1.0, npix_u)),
        np.log10(max(1.0, npix_v)),
        np.log10(max(1.0, npix_y)),
        np.log10(max(1.0, npix_tot)),
        (q_u + 1.0) / (q_y + 1.0),
        (q_v + 1.0) / (q_y + 1.0),
        (npix_u + 1.0) / (npix_y + 1.0),
        (npix_v + 1.0) / (npix_y + 1.0),
        q_tot / (npix_tot + 1e-4),
        q_y / (npix_y + 1e-4)
    ]
    return feats

# ==============================================================================
# 3. VERİ KÜMESİ VE SIZINTISIZ STRATIFIED GROUP BÖLÜMLEME
# ==============================================================================
class LArTPCHybridDataset(Dataset):
    def __init__(self, metadata_list, tabular_matrix):
        self.metadata = metadata_list
        self.tabular = torch.tensor(tabular_matrix, dtype=torch.float32)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        with open(item['bin_path'], 'rb') as f:
            f.seek(item['offset'])
            buf = f.read(item['nbytes'])

        planes = kayit_coz(buf)
        img_tensor = torch.from_numpy(cnn_girdisi(planes))  # (3, 504, 864)
        tab_tensor = self.tabular[idx]                      # (14,)
        lbl = torch.tensor(item['label'], dtype=torch.float32)
        return img_tensor, tab_tensor, lbl

def scan_and_prepare_dataset(max_samples_per_class=None):
    print("🔍 Tüm veri seti ve CSV dosyaları taranıyor...")
    nu_csvs = sorted(glob.glob("**/*neutrino*_events.csv", recursive=True) + glob.glob("*neutrino*_events.csv"))
    cos_csvs = sorted(glob.glob("**/*cosmic*_events.csv", recursive=True) + glob.glob("*cosmic*_events.csv"))

    if not nu_csvs or not cos_csvs:
        raise FileNotFoundError("Gerekli _events.csv dosyaları bulunamadı!")

    all_meta, all_tab, groups = [], [], []

    # 1. Nötrinolar
    for f_csv in nu_csvs:
        prefix = f_csv.replace("_events.csv", "")
        f_bin = prefix + "_images.bin"
        if not os.path.exists(f_bin): continue

        with open(f_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row.get('truth_npix_Y_nu', 0)) < 10:
                    continue
                
                run_id = int(row.get('run', 0))
                all_meta.append({
                    'bin_path': f_bin,
                    'offset': int(row['img_offset']),
                    'nbytes': int(row['img_nbytes']),
                    'label': 1.0,
                    'run': run_id,
                    'event_id': int(row['event_id'])
                })
                all_tab.append(extract_reco_tabular_features(row))
                # Çakışmayı önlemek için önek ekleniyor
                groups.append(f"nu_{run_id}")
                if max_samples_per_class and len(all_meta) >= max_samples_per_class: break
        if max_samples_per_class and len(all_meta) >= max_samples_per_class: break

    nu_count = len(all_meta)

    # 2. Kozmikler
    for f_csv in cos_csvs:
        prefix = f_csv.replace("_events.csv", "")
        f_bin = prefix + "_images.bin"
        if not os.path.exists(f_bin): continue

        with open(f_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                run_id = int(row.get('run', 0))
                all_meta.append({
                    'bin_path': f_bin,
                    'offset': int(row['img_offset']),
                    'nbytes': int(row['img_nbytes']),
                    'label': 0.0,
                    'run': run_id,
                    'event_id': int(row['event_id'])
                })
                all_tab.append(extract_reco_tabular_features(row))
                # Çakışmayı önlemek için önek ekleniyor
                groups.append(f"cos_{run_id}")
                if max_samples_per_class and (len(all_meta) - nu_count) >= max_samples_per_class: break
        if max_samples_per_class and (len(all_meta) - nu_count) >= max_samples_per_class: break

    cos_count = len(all_meta) - nu_count
    print(f"✅ Toplam Bulunan Olaylar -> Nötrino: {nu_count:,} | Kozmik: {cos_count:,}")

    all_tab = np.array(all_tab, dtype=np.float32)
    labels = np.array([m['label'] for m in all_meta])
    groups = np.array(groups)
    indices = np.arange(len(all_meta))

    # ==============================================================================
    # STRATIFIED GROUP K-FOLD: Run sızıntısız + Dengeli sınıf dağılımı
    # ==============================================================================
    # 1. Aşama: Test kümesini (%20) bağımsız run'lardan ve dengeli sınıfla ayır
    sgkf_test = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_val_idx, te_idx = next(sgkf_test.split(indices, labels, groups=groups))

    # 2. Aşama: Kalan %80'i Train (%64) ve Val (%16) olarak bağımsız run'lara böl
    sgkf_val = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    tr_sub, val_sub = next(sgkf_val.split(
        train_val_idx, 
        labels[train_val_idx], 
        groups=groups[train_val_idx]
    ))

    tr_idx = train_val_idx[tr_sub]
    val_idx = train_val_idx[val_sub]

    # Run sızıntısı kontrolü (Kesişim kesinlikle sıfır olmalı)
    tr_runs = set(groups[tr_idx])
    val_runs = set(groups[val_idx])
    te_runs = set(groups[te_idx])
    assert len(tr_runs.intersection(val_runs)) == 0, "HATA: Train ve Val arasında ortak Run saptandı!"
    assert len(tr_runs.intersection(te_runs)) == 0, "HATA: Train ve Test arasında ortak Run saptandı!"
    assert len(val_runs.intersection(te_runs)) == 0, "HATA: Val ve Test arasında ortak Run saptandı!"

    # Sızıntısız Normalizasyon: Doğrudan alt kümelere dönüştürülür
    scaler = RobustScaler()
    tr_tab = scaler.fit_transform(all_tab[tr_idx])
    val_tab = scaler.transform(all_tab[val_idx])
    te_tab = scaler.transform(all_tab[te_idx])

    tr_tab = np.nan_to_num(tr_tab, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    val_tab = np.nan_to_num(val_tab, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    te_tab = np.nan_to_num(te_tab, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    tr_meta = [all_meta[i] for i in tr_idx]
    val_meta = [all_meta[i] for i in val_idx]
    te_meta = [all_meta[i] for i in te_idx]

    tr_nu_ratio = float(np.mean(labels[tr_idx])) * 100
    val_nu_ratio = float(np.mean(labels[val_idx])) * 100
    te_nu_ratio = float(np.mean(labels[te_idx])) * 100

    print("\n🔒 SIZINTISIZ STRATIFIED GROUP BÖLÜMLEME RAPORU:")
    print(f"   • Train : {len(tr_meta):,} Olay ({len(tr_runs)} Farklı Run Grubu) | Nötrino Oranı: %{tr_nu_ratio:.1f}")
    print(f"   • Val   : {len(val_meta):,} Olay ({len(val_runs)} Farklı Run Grubu) | Nötrino Oranı: %{val_nu_ratio:.1f}")
    print(f"   • Test  : {len(te_meta):,} Olay ({len(te_runs)} Farklı Run Grubu) | Nötrino Oranı: %{te_nu_ratio:.1f}")
    print("   ✅ Sıfır Sızıntı Güvencesi: Run bazlı tam izolasyon ve dengeli sınıf dağılımı sağlandı.\n")

    return tr_meta, val_meta, te_meta, tr_tab, val_tab, te_tab
# ==============================================================================
# 4. HİBRİT MODEL MİMARİSİ
# ==============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels, drop_prob=0.10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(p=drop_prob),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv(x))

class HybridLArTPCNet(nn.Module):
    def __init__(self, num_tab_features=14):
        super().__init__()
        
        # --- 1. GÖRSEL KOL (CNN): 504x864 -> 16x27 ---
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            ResidualBlock(32, drop_prob=0.05),
            nn.MaxPool2d(2, 2)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            ResidualBlock(48, drop_prob=0.08)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            ResidualBlock(96, drop_prob=0.10)
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True),
            ResidualBlock(192, drop_prob=0.12)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))

        # CNN çıkışı 48 yerine 32 boyuta çekildi (Görsel gürültü filtreleme)
        self.img_proj = nn.Sequential(
            nn.Linear(192 * 2, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32)
        )

        # --- 2. TABLOSAL KOL (MLP): 48 yerine 64 boyuta genişletildi ---
        self.tab_proj = nn.Sequential(
            nn.Linear(num_tab_features, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True)
        )

        # --- 3. DİNAMİK GEÇİT (GATING) & BİRLEŞTİRME ---
        # 32 (Görsel) + 64 (Tablosal) = 96
        self.gate = nn.Sequential(
            nn.Linear(32 + 64, 32),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Linear(32 + 64, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, 1)
        )

    def forward(self, img, tab):
        x = self.stage1(img)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        h_img_pool = torch.cat([self.gap(x).flatten(1), self.gmp(x).flatten(1)], dim=1)
        h_img = self.img_proj(h_img_pool)      # (B, 32)
        h_tab = self.tab_proj(tab)            # (B, 64)

        # Geçit: CNN çıktısını tablosal özelliklerin denetimine göre filtreler
        fused_raw = torch.cat([h_img, h_tab], dim=1)
        g = self.gate(fused_raw)              # (B, 32)
        h_img_gated = h_img * g

        # Ağırlığı artırılmış nihai birleşim (32 gated + 64 tablosal)
        fused = torch.cat([h_img_gated, h_tab], dim=1)
        return self.fusion(fused).squeeze(1)

# ==============================================================================
# 5. PROFESYONEL GRAFİK VE DEĞERLENDİRME ÇIKTILARI
# ==============================================================================
def save_performance_plots(history, test_targets, test_probs, opt_thresh, brier_score, out_dir):
    plt.style.use('default')
    epochs = range(1, len(history['train_loss']) + 1)
    
    # --------------------------------------------------------------------------
    # 1. Loss & Accuracy Eğrileri
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
    
    ax1.plot(epochs, history['train_loss'], 'b-o', lw=2, label='Eğitim Kaybı (Train Loss)')
    ax1.plot(epochs, history['val_loss'], 'r--s', lw=2, label='Doğrulama Kaybı (Val Loss)')
    ax1.set_title("Eğitim ve Doğrulama Kayıp (Loss) Eğrisi", fontweight='bold')
    ax1.set_xlabel("Epoch", fontweight='bold')
    ax1.set_ylabel("BCE Loss", fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    ax2.plot(epochs, [a * 100 for a in history['train_acc']], 'b-o', lw=2, label='Train Acc (%)')
    ax2.plot(epochs, [a * 100 for a in history['val_acc']], 'r--s', lw=2, label='Val Acc (%)')
    ax2.set_title("Doğruluk (Accuracy %) Eğrisi", fontweight='bold')
    ax2.set_xlabel("Epoch", fontweight='bold')
    ax2.set_ylabel("Doğruluk (%)", fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Loss_Accuracy_Curve.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # --------------------------------------------------------------------------
    # 2. ROC, Olasılık Dağılımı, Kalibrasyon ve Hata Matrisi (4 Panel)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), facecolor='white')
    
    # Panel 1: ROC Eğrisi
    fpr, tpr, _ = roc_curve(test_targets, test_probs)
    auc_score = roc_auc_score(test_targets, test_probs)
    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC Eğrisi (AUC = {auc_score:.4f})')
    axes[0, 0].plot([0, 1], [0, 1], color='navy', linestyle='--', lw=1.5, label='Rastgele Tahmin')
    axes[0, 0].set_title("1. Test ROC Eğrisi", fontweight='bold')
    axes[0, 0].set_xlabel("Kozmik Yanlış Pozitif Oranı (FPR)", fontweight='bold')
    axes[0, 0].set_ylabel("Nötrino Yakalama Oranı (TPR)", fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc='lower right')

    # Panel 2: Olasılık Dağılımları
    cos_p = test_probs[test_targets == 0]
    nu_p = test_probs[test_targets == 1]
    axes[0, 1].hist(cos_p, bins=40, alpha=0.6, color='crimson', label='Kozmik Arka Plan (0)', density=True)
    axes[0, 1].hist(nu_p, bins=40, alpha=0.6, color='royalblue', label='Nötrino Sinyali (1)', density=True)
    axes[0, 1].axvline(opt_thresh, color='black', linestyle='--', lw=2, label=f'Optimal Eşik (τ*={opt_thresh:.2f})')
    axes[0, 1].set_title("2. Model Çıktı Olasılık Dağılımı", fontweight='bold')
    axes[0, 1].set_xlabel("P(Neutrino)", fontweight='bold')
    axes[0, 1].set_ylabel("Yoğunluk (Density)", fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(loc='upper center')

    # Panel 3: Kalibrasyon Eğrisi
    prob_true, prob_pred = calibration_curve(test_targets, test_probs, n_bins=10)
    axes[1, 0].plot(prob_pred, prob_true, 's-', color='purple', lw=2, label=f'Model (Brier: {brier_score:.4f})')
    axes[1, 0].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Mükemmel Kalibrasyon')
    axes[1, 0].set_title("3. Güvenilirlik / Kalibrasyon Eğrisi", fontweight='bold')
    axes[1, 0].set_xlabel("Tahmin Edilen Ortalama Olasılık", fontweight='bold')
    axes[1, 0].set_ylabel("Gerçek Pozitif Oranı", fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(loc='upper left')

    # Panel 4: Normalize Hata Matrisi
    test_preds_binary = (test_probs >= opt_thresh).astype(int)
    cm = confusion_matrix(test_targets, test_preds_binary, normalize='true')
    im = axes[1, 1].imshow(cm, cmap='Blues', vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            val_txt = f"%{cm[i, j]*100:.2f}"
            color = "white" if cm[i, j] > 0.5 else "black"
            axes[1, 1].text(j, i, val_txt, ha='center', va='center', color=color, fontweight='bold', fontsize=14)
            
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_xticklabels(['Kozmik (0)', 'Nötrino (1)'], fontsize=11)
    axes[1, 1].set_yticklabels(['Kozmik (0)', 'Nötrino (1)'], fontsize=11)
    axes[1, 1].set_xlabel("Tahmin Edilen Sınıf", fontweight='bold')
    axes[1, 1].set_ylabel("Gerçek Sınıf", fontweight='bold')
    axes[1, 1].set_title(f"4. Normalize Hata Matrisi (τ*={opt_thresh:.2f})", fontweight='bold')
    
    # Eksen bozulmasını engelleyen orantılı colorbar
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(out_dir, "ROC_Calibration_ConfusionMatrix.png"), dpi=200, bbox_inches='tight')
    plt.close()

# ==============================================================================
# 6. EĞİTİM BORU HATTI VE DENGELİ ERKEN DURDURMA (DÜZELTİLMİŞ)
# ==============================================================================
def main():
    tr_meta, val_meta, te_meta, tr_tab, val_tab, te_tab = scan_and_prepare_dataset(max_samples_per_class=None)

    batch_size = 24
    num_workers = 6

    train_ds = LArTPCHybridDataset(tr_meta, tr_tab)
    val_ds = LArTPCHybridDataset(val_meta, val_tab)
    test_ds = LArTPCHybridDataset(te_meta, te_tab)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = HybridLArTPCNet(num_tab_features=14).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()

    max_epochs = 8
    patience = 2               
    patience_counter = 0
    MAX_ACC_GAP = 0.030     # %3.0 tolerans sınırı
    MIN_AUC_TARGET = 0.80      # Taban AUC (0.85 yerine 0.80 ile başlatmak daha güvenli)

    base_lr = 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=2e-2)

    warmup_epochs = 1
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        denom = max(1, max_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * (epoch - warmup_epochs) / denom))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_auc': []}
    best_balanced_auc = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, "best_hybrid_v6_model.pt")

    print("\n🚀 Hibrit (Tel Görüntüsü + Tablosal) Dengeli Eğitim Başlıyor...")
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for imgs, tabs, lbls in train_loader:
            imgs, tabs, lbls = imgs.to(DEVICE), tabs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
                logits = model(imgs, tabs)
                loss = criterion(logits, lbls)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            tr_loss += loss.item() * len(lbls)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            tr_correct += (preds == lbls).sum().item()
            tr_total += len(lbls)

        scheduler.step()
        ep_tr_loss = tr_loss / tr_total
        ep_tr_acc = tr_correct / tr_total

        # --- Doğrulama Döngüsü ---
        model.eval()
        val_loss, val_total = 0.0, 0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for imgs, tabs, lbls in val_loader:
                imgs, tabs, lbls = imgs.to(DEVICE), tabs.to(DEVICE), lbls.to(DEVICE)
                with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
                    logits = model(imgs, tabs)
                    loss = criterion(logits, lbls)

                val_loss += loss.item() * len(lbls)
                val_total += len(lbls)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                val_targets.extend(lbls.cpu().numpy().flatten())

        ep_val_loss = val_loss / val_total
        val_preds_arr = np.array(val_preds)
        val_targets_arr = np.array(val_targets)
        val_auc = roc_auc_score(val_targets_arr, val_preds_arr)

        # 1. Eşit şartlarda (0.50 eşiği) Val Acc (Makas için):
        val_acc_standard = np.mean((val_preds_arr >= 0.5).astype(int) == val_targets_arr)

        # 2. Raporlama için Dinamik Eşik:
        p_v, r_v, th_v = precision_recall_curve(val_targets_arr, val_preds_arr)
        f1_v = 2 * (p_v * r_v) / (p_v + r_v + 1e-8)
        cur_opt_th = float(th_v[min(np.argmax(f1_v), len(th_v) - 1)])
        val_acc_opt = np.mean((val_preds_arr >= cur_opt_th).astype(int) == val_targets_arr)

        dur = time.time() - t0

        history['train_loss'].append(ep_tr_loss)
        history['val_loss'].append(ep_val_loss)
        history['train_acc'].append(ep_tr_acc)
        history['val_acc'].append(val_acc_standard)
        history['val_auc'].append(val_auc)

        # Makas hesabı aynı standartta (0.50 vs 0.50):
        acc_gap = abs(ep_tr_acc - val_acc_standard)
        print(f"Epoch {epoch:02d}/{max_epochs:02d} | Tr Loss: {ep_tr_loss:.4f} | Val Loss: {ep_val_loss:.4f} | Tr Acc: %{ep_tr_acc*100:.2f} | Val Acc: %{val_acc_standard*100:.2f} (Opt: %{val_acc_opt*100:.1f}, τ*={cur_opt_th:.2f}) | Makas: %{acc_gap*100:.2f} | Val AUC: {val_auc:.4f} | Süre: {dur:.1f}s")

        # Karar Kuralı: Zirvede Kayıt ve Acil Fren
        if val_auc >= MIN_AUC_TARGET and acc_gap <= MAX_ACC_GAP:
            if val_auc > best_balanced_auc:
                best_balanced_auc = val_auc
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"   ⭐ [ZİRVEDE KAYIT]: En İyi Dengeli Model Saklandı (Val AUC: {best_balanced_auc:.4f}, Makas: %{acc_gap*100:.2f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n🛑 [DENGELİ BİTİRME]: Model plato çizdi, AUC artmıyor. Eğitim {epoch}. epochta tamamlandı.")
                    break
        else:
            if val_auc >= MIN_AUC_TARGET and acc_gap > MAX_ACC_GAP:
                patience_counter += 1
                print(f"   ⚠️ Uyarı: Genelleme makası açılıyor (Makas: %{acc_gap*100:.2f} > %{MAX_ACC_GAP*100:.1f}). Sabır: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"\n🛑 [ACİL FREN]: Aşırı öğrenme tespit edildi. Eğitim {epoch}. epochta durduruldu.")
                    break

    # ==============================================================================
    # 7. KÖR TEST DEĞERLENDİRMESİ VE RAPORLAMA
    # ==============================================================================
    print("\n" + "=" * 80)
    print("🧪 EN İYİ MODEL YÜKLENİYOR VE TAM KÖR TEST SETİ DEĞERLENDİRİLİYOR...")
    print("=" * 80)
    model.load_state_dict(torch.load(best_model_path, weights_only=False))
    model.eval()

    val_probs, val_targets = [], []
    with torch.no_grad():
        for imgs, tabs, lbls in val_loader:
            imgs, tabs = imgs.to(DEVICE), tabs.to(DEVICE)
            val_probs.extend(torch.sigmoid(model(imgs, tabs)).cpu().numpy().flatten())
            val_targets.extend(lbls.numpy().flatten())

    precision_v, recall_v, thresholds_v = precision_recall_curve(val_targets, val_probs)
    f1_scores_v = 2 * (precision_v * recall_v) / (precision_v + recall_v + 1e-8)
    opt_idx = np.argmax(f1_scores_v)
    opt_thresh = float(thresholds_v[min(opt_idx, len(thresholds_v) - 1)])
    print(f"📌 Validasyon Kümesinde Kilitlenen Optimal Eşik: tau* = {opt_thresh:.4f}")

    test_probs, test_targets = [], []
    with torch.no_grad():
        for imgs, tabs, lbls in test_loader:
            imgs, tabs = imgs.to(DEVICE), tabs.to(DEVICE)
            test_probs.extend(torch.sigmoid(model(imgs, tabs)).cpu().numpy().flatten())
            test_targets.extend(lbls.numpy().flatten())

    test_probs = np.array(test_probs)
    test_targets = np.array(test_targets)
    test_preds_binary = (test_probs >= opt_thresh).astype(int)

    test_auc = roc_auc_score(test_targets, test_probs)
    brier_score = brier_score_loss(test_targets, test_probs)
    cm = confusion_matrix(test_targets, test_preds_binary)
    cm_norm = confusion_matrix(test_targets, test_preds_binary, normalize='true')
    cls_rep = classification_report(test_targets, test_preds_binary, target_names=['Kozmik (0)', 'Nötrino (1)'], digits=4)
    macro_f1 = f1_score(test_targets, test_preds_binary, average='macro')
    weighted_f1 = f1_score(test_targets, test_preds_binary, average='weighted')

    print("\n" + "=" * 70)
    print("📊 HİBRİT MODEL TEST SONUÇLARI (GROUP SHUFFLE SPLIT):")
    print("=" * 70)
    print(cls_rep)
    print(f"• Macro F1-Score   : {macro_f1:.4f}")
    print(f"• Weighted F1-Score: {weighted_f1:.4f}")
    print(f"• Test ROC-AUC     : {test_auc:.4f}")
    print(f"• Brier Kalib.     : {brier_score:.4f}")
    print(f"• Kilitli Eşik     : {opt_thresh:.4f}")
    print("=" * 70)

    save_performance_plots(history, test_targets, test_probs, opt_thresh, brier_score, OUTPUT_DIR)

    rep_path = os.path.join(OUTPUT_DIR, "hybrid_v6_evaluation_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("LArTPC V6.0 TAM DEDEKTÖR HİBRİT (GROUP SHUFFLE SPLIT) PERFORMANS RAPORU\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Nihai Test ROC-AUC Skoru     : {test_auc:.4f}\n")
        f.write(f"Macro F1-Score               : {macro_f1:.4f}\n")
        f.write(f"Weighted F1-Score            : {weighted_f1:.4f}\n")
        f.write(f"Brier Kalibrasyon Skoru      : {brier_score:.4f}\n")
        f.write(f"Validasyondan Kilitli Eşik   : {opt_thresh:.4f}\n")
        f.write(f"Nötrino Sinyal Verimi (TPR)  : %{cm_norm[1, 1]*100:.2f}\n")
        f.write(f"Kozmik Reddetme (1-FPR)      : %{cm_norm[0, 0]*100:.2f}\n\n")
        f.write("Confusion Matrix (Ham Sayılar):\n")
        f.write(f"{cm}\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write(cls_rep + "\n")
        f.write("=" * 80 + "\n")

    print(f"\n📈 Eğitim Grafikleri  : {OUTPUT_DIR}/Loss_Accuracy_Curve.png")
    print(f"📊 Test Analiz Paneli : {OUTPUT_DIR}/ROC_Calibration_ConfusionMatrix.png")
    print(f"📄 Metin Raporu       : {rep_path}")
    print("=" * 85)

if __name__ == '__main__':
    main()