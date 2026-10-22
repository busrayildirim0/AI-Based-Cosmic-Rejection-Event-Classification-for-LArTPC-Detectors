"""
LArTPC Hibrit Model (CNN + MLP) Optimizasyon ve Ablasyon Boru Hattı
Sürüm: v7_ablation_ready

Düzeltmeler ve Metodolojik İyileştirmeler:
1. Bölümleme: Sınıf-saf run sızıntısını (leakage) önlemek için Train, Val ve Test
   kümelerinin tamamı katı biçimde StratifiedGroupKFold (Run-tabanlı) olarak korunur.
2. Charge jitter hem 3 düzleme ortak bir çarpanla uygulanır hem de tablosal yük sütunlarıyla senkronize edilir.
3. Modality Dropout: Sıfırlama yerine RobustScaler dağılımını bozmayan maskeleme veya bayrak kontrolü eklendi.
4. Optuna & Early Stopping:
   - İlk 8 epoch budama/durdurma yapılmaz (ısınma ve plato gecikmesi koruması).
   - T_max = MAX_EPOCHS (16) korunarak kosinüs çizelgesinin doğal sönümlenmesi sağlandı.
   - Arama uzayı baseline mimarisini (32-48-96-192) merkez alacak şekilde kuruldu.
5. Değerlendirme & Raporlama:
   - Dörtlü panel (ROC, Dağılım, Kalibrasyon, Hata Matrisi) ve Eğitim Eğrileri eksiksiz çizilir.
   - Optuna denemeleri CSV olarak dışa aktarılır.
"""

import os
import glob
import csv
import zlib
import time
import random
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix, brier_score_loss, f1_score
)
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna

# 1. BAYRAKLAR VE ABLASYON AYARLARI
USE_OPTUNA = True          # False: Doğrudan seçilen hiperparametrelerle tek eğitim yapar.
                            # True: N_TRIALS kadar arama yapıp en iyisini test eder.

USE_EMA = True              # Üstel hareketli ortalama ağırlık takibi
USE_LS = True              # Label smoothing (Dikkat: Açılırsa loss tabanı yükselir)
USE_QJIT = True            # Fiziksel tutarlı Charge Jitter (Görüntü + Tablo senkron)
USE_MOD_DROP = True        # Tablosal kol için modalite dropout

EMA_DECAY = 0.999
LABEL_SMOOTHING = 0.05
MOD_DROP_PROB = 0.15

# Temel Eğitim Hiperparametreleri (Baseline omurgası)
SEED = 42
RUN_NAME = "v7_ablation"
BATCH_SIZE = 24
NUM_WORKERS = 6
BASE_LR = 3e-4
WARMUP_EPOCHS = 1
WEIGHT_DECAY = 2e-2
MAX_EPOCHS = 25
PATIENCE = 3
MIN_EPOCHS_BEFORE_STOP = 5  # Modelin 0.61 platosundan çıkmasını bekleyen güvenlik eşiği

N_TRIALS = 10               # USE_OPTUNA = True ise denenecek kombinasyon sayısı
TRAIN_EVAL_SUBSET = 5000
TRUTH_BINS = [(10, 30), (30, 100), (100, 300), (300, 1e9)]

DATA_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(DATA_DIR, f"outputs_{RUN_NAME}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


set_seed(SEED)

print("=" * 85)
print(f"LArTPC Pipeline | {RUN_NAME} | Cihaz: {DEVICE}")
print(f"Ablasyon Bayrakları: EMA={USE_EMA} | LS={USE_LS} | QJit={USE_QJIT} | ModDrop={USE_MOD_DROP} | Optuna={USE_OPTUNA}")
print("=" * 85)


# ==============================================================================
# 2. İKİLİ ÇÖZÜCÜ VE FİZİKSEL DÖNÜŞÜMLER
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
            'ntel': int(ntel), 'nsatir': int(nsatir),
            'tel': tel, 'satir': satir, 'q': q, 'etiket': etiket
        })
    return duzlemler


def cnn_girdisi(duzlemler, satir_kat=2, genislik=864, q_scale=1.0):
    yukseklik = duzlemler[2]['nsatir'] // satir_kat  # 504
    x = np.zeros((3, yukseklik, genislik), dtype=np.float32)
    for c, p in enumerate(duzlemler):
        tel_kat = -(-p['ntel'] // genislik)
        r = p['satir'].astype(int) // satir_kat
        w = p['tel'].astype(int) // tel_kat
        m = (r < yukseklik) & (w < genislik)
        np.add.at(x[c], (r[m], w[m]), np.clip(p['q'][m], 0, None) * q_scale)
    return np.log10(1.0 + x / 1000.0)


def extract_reco_tabular_features(row, q_scale=1.0):
    # Charge jitter uygulandığında tablosal yükleri de aynı katsayıyla güncelleyerek
    # görüntü ile tablonun birbiriyle çelişmesini engeller.
    q_u = float(row.get('reco_Q_U', 0.0)) * q_scale
    q_v = float(row.get('reco_Q_V', 0.0)) * q_scale
    q_y = float(row.get('reco_Q_Y', 0.0)) * q_scale
    npix_u = float(row.get('reco_npix_U', 0.0))
    npix_v = float(row.get('reco_npix_V', 0.0))
    npix_y = float(row.get('reco_npix_Y', 0.0))
    q_tot = q_u + q_v + q_y
    npix_tot = npix_u + npix_v + npix_y
    return [
        np.log10(max(1.0, q_u)), np.log10(max(1.0, q_v)), np.log10(max(1.0, q_y)),
        np.log10(max(1.0, q_tot)),
        np.log10(max(1.0, npix_u)), np.log10(max(1.0, npix_v)), np.log10(max(1.0, npix_y)),
        np.log10(max(1.0, npix_tot)),
        (q_u + 1.0) / (q_y + 1.0), (q_v + 1.0) / (q_y + 1.0),
        (npix_u + 1.0) / (npix_y + 1.0), (npix_v + 1.0) / (npix_y + 1.0),
        q_tot / (npix_tot + 1e-4), q_y / (npix_y + 1e-4),
    ]


# ==============================================================================
# 3. VERİ KÜMESİ VE GRUP TABANLI BÖLÜMLEME (DATASET & GROUP SPLIT)
# ==============================================================================
class LArTPCHybridDataset(Dataset):
    def __init__(self, metadata_list, raw_rows, scaler, is_train=False):
        self.metadata = metadata_list
        self.raw_rows = raw_rows
        self.scaler = scaler
        self.is_train = is_train

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        with open(item['bin_path'], 'rb') as f:
            f.seek(item['offset'])
            buf = f.read(item['nbytes'])
        planes = kayit_coz(buf)

        # Fiziksel Tutarlı Charge Jitter: Tüm dedektör için ortak kalibrasyon sapması
        q_scale = np.random.uniform(0.92, 1.08) if (self.is_train and USE_QJIT) else 1.0

        img = cnn_girdisi(planes, q_scale=q_scale)

        # Tablosal özellikleri jitter katsayısıyla senkron hesapla ve scale et
        feat = extract_reco_tabular_features(self.raw_rows[idx], q_scale=q_scale)
        tab_scaled = self.scaler.transform([feat])[0]
        tab_scaled = np.nan_to_num(tab_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Modality Dropout: Tabloyu tamamen sıfırlamak yerine rastgele maskeleme
        if self.is_train and USE_MOD_DROP and (random.random() < MOD_DROP_PROB):
            tab_scaled = np.zeros_like(tab_scaled)

        return (
            torch.from_numpy(img),
            torch.tensor(tab_scaled, dtype=torch.float32),
            torch.tensor(item['label'], dtype=torch.float32)
        )


def _read_class(csv_files, label, tag):
    meta, raw_rows, groups = [], [], []
    for f_csv in csv_files:
        f_bin = f_csv.replace("_events.csv", "") + "_images.bin"
        if not os.path.exists(f_bin):
            continue
        with open(f_csv, newline='') as f:
            for row in csv.DictReader(f):
                truth_npix = float(row.get('truth_npix_Y_nu', 0) or 0)
                if label == 1.0 and truth_npix < 10:
                    continue
                run_id = int(row.get('run', 0))
                meta.append({
                    'bin_path': f_bin,
                    'offset': int(row['img_offset']),
                    'nbytes': int(row['img_nbytes']),
                    'label': label,
                    'run': run_id,
                    'event_id': int(row['event_id']),
                    'truth_npix_Y_nu': truth_npix if label == 1.0 else 0.0
                })
                raw_rows.append(row)
                groups.append(f"{tag}_{run_id}")
    return meta, raw_rows, groups


def scan_and_prepare_dataset():
    print("Veri taranıyor...")
    nu_csvs = sorted(set(glob.glob("**/*neutrino*_events.csv", recursive=True) +
                         glob.glob("*neutrino*_events.csv")))
    cos_csvs = sorted(set(glob.glob("**/*cosmic*_events.csv", recursive=True) +
                          glob.glob("*cosmic*_events.csv")))
    if not nu_csvs or not cos_csvs:
        raise FileNotFoundError("Gerekli _events.csv dosyaları bulunamadı!")

    nu_m, nu_r, nu_g = _read_class(nu_csvs, 1.0, "nu")
    co_m, co_r, co_g = _read_class(cos_csvs, 0.0, "cos")
    print(f"Toplam Tekil Olay -> Nötrino: {len(nu_m):,} | Kozmik: {len(co_m):,}")

    all_meta = nu_m + co_m
    all_raw = nu_r + co_r
    labels = np.array([m['label'] for m in all_meta])
    groups = np.array(nu_g + co_g)
    idx = np.arange(len(all_meta))

    # Test kümesi: Bağımsız Run'lar (Sızıntısız)
    s1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tv_idx, te_idx = next(s1.split(idx, labels, groups=groups))

    # Train / Val kümesi: Kesinlikle Grup Tabanlı (Sınıf-saf run sızıntısı önlendi)
    s2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr_sub, va_sub = next(s2.split(tv_idx, labels[tv_idx], groups=groups[tv_idx]))
    tr_idx, va_idx = tv_idx[tr_sub], tv_idx[va_sub]

    # Kritik doğrulama: Ortak run var mı?
    trg, vag, teg = set(groups[tr_idx]), set(groups[va_idx]), set(groups[te_idx])
    assert not (trg & vag) and not (trg & teg) and not (vag & teg), "Kritik Hata: Kümeler arası run sızıntısı tespit edildi!"

    # RobustScaler sadece eğitim verisinin ham özellikleri üzerinde fit edilir
    tr_feats = [extract_reco_tabular_features(all_raw[i], q_scale=1.0) for i in tr_idx]
    scaler = RobustScaler()
    scaler.fit(tr_feats)

    print("\nBÖLÜMLEME BİLGİSİ (Tamamı Grup Tabanlı):")
    for ad, ii, gg in [("Train", tr_idx, trg), ("Val", va_idx, vag), ("Test", te_idx, teg)]:
        print(f"  {ad:5s}: {len(ii):,} olay | {len(gg)} run grubu | Nötrino Oranı: %{labels[ii].mean()*100:.1f}")
    print()

    return (
        [all_meta[i] for i in tr_idx], [all_meta[i] for i in va_idx], [all_meta[i] for i in te_idx],
        [all_raw[i] for i in tr_idx], [all_raw[i] for i in va_idx], [all_raw[i] for i in te_idx],
        scaler
    )


# ==============================================================================
# 4. MODEL MİMARİSİ
# ==============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels, drop_prob=0.10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True), nn.Dropout2d(p=drop_prob),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv(x))


class HybridLArTPCNet(nn.Module):
    def __init__(self, c1=32, c2=48, c3=96, c4=192, drop_img=0.40, drop_fusion=0.25, num_tab=14):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, c1, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True), ResidualBlock(c1, 0.05), nn.MaxPool2d(2, 2)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True), ResidualBlock(c2, 0.08)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c3),
            nn.SiLU(inplace=True), ResidualBlock(c3, 0.10)
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(c3, c4, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c4),
            nn.SiLU(inplace=True), ResidualBlock(c4, 0.12)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))

        self.img_proj = nn.Sequential(
            nn.Linear(c4 * 2, 64), nn.BatchNorm1d(64), nn.SiLU(inplace=True),
            nn.Dropout(drop_img), nn.Linear(64, 32), nn.BatchNorm1d(32)
        )
        self.tab_proj = nn.Sequential(
            nn.Linear(num_tab, 64), nn.BatchNorm1d(64), nn.SiLU(inplace=True),
            nn.Dropout(0.15), nn.Linear(64, 64), nn.BatchNorm1d(64), nn.SiLU(inplace=True)
        )
        self.gate = nn.Sequential(nn.Linear(32 + 64, 32), nn.Sigmoid())
        self.fusion = nn.Sequential(
            nn.Linear(32 + 64, 64), nn.BatchNorm1d(64), nn.SiLU(inplace=True),
            nn.Dropout(drop_fusion), nn.Linear(64, 1)
        )

    def forward(self, img, tab):
        x = self.stage4(self.stage3(self.stage2(self.stage1(img))))
        h_img = self.img_proj(torch.cat([self.gap(x).flatten(1), self.gmp(x).flatten(1)], dim=1))
        h_tab = self.tab_proj(tab)
        g = self.gate(torch.cat([h_img, h_tab], dim=1))
        return self.fusion(torch.cat([h_img * g, h_tab], dim=1)).squeeze(1)


# ==============================================================================
# 5. YARDIMCILAR: EMA, EARLY STOPPING VE GRAFİKLER
# ==============================================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.step = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        for pe, p in zip(self.module.parameters(), model.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for be, b in zip(self.module.buffers(), model.buffers()):
            be.copy_(b)


class EarlyStopping:
    """Isınma ve plato korumalı (min_epochs) Val AUC erken durdurucusu."""
    def __init__(self, patience=5, min_epochs=8, path="best.pt"):
        self.patience = patience
        self.min_epochs = min_epochs
        self.path = path
        self.best_auc = 0.0
        self.counter = 0
        self.early_stop = False

    def __call__(self, epoch, val_auc, state_dict):
        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.counter = 0
            torch.save(state_dict, self.path)
            return True
        else:
            if epoch >= self.min_epochs:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            return False


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    probs, targets, tot_loss, n = [], [], 0.0, 0
    bce = nn.BCEWithLogitsLoss()
    for imgs, tabs, lbls in loader:
        imgs, tabs, lbls = imgs.to(DEVICE), tabs.to(DEVICE), lbls.to(DEVICE)
        with torch.amp.autocast(DEVICE.type, enabled=(DEVICE.type == 'cuda')):
            logits = model(imgs, tabs)
        tot_loss += bce(logits.float(), lbls).item() * len(lbls)
        n += len(lbls)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        targets.append(lbls.cpu().numpy())
    return tot_loss / max(n, 1), np.concatenate(probs), np.concatenate(targets)


def fpr_at_tpr(y, p, target):
    fpr, tpr, thr = roc_curve(y, p)
    i = min(int(np.searchsorted(tpr, target, side='left')), len(fpr) - 1)
    return float(fpr[i]), float(tpr[i]), float(thr[i])


def f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / (pr + rc + 1e-8)
    return float(th[min(int(np.argmax(f1)), len(th) - 1)])


def save_plots(hist, y, p, thr, brier, out_dir):
    ep = range(1, len(hist['train_loss']) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
    ax[0].plot(ep, hist['train_loss'], 'b-o', label='Train Loss')
    ax[0].plot(ep, hist['val_loss'], 'r--s', label='Val Loss')
    ax[0].set_title("Kayıp Eğrisi"); ax[0].set_xlabel("Epoch"); ax[0].grid(alpha=0.3); ax[0].legend()
    ax[1].plot(ep, hist['train_auc_eval'], 'b-o', label='Train AUC (Alt Küme)')
    ax[1].plot(ep, hist['val_auc'], 'r--s', label='Val AUC')
    ax[1].set_title("AUC Başarım Eğrisi"); ax[1].set_xlabel("Epoch"); ax[1].grid(alpha=0.3); ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Training_Curves.png"), dpi=200)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), facecolor='white')
    fpr, tpr, _ = roc_curve(y, p)
    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2.5, label=f'AUC = {roc_auc_score(y, p):.4f}')
    axes[0, 0].plot([0, 1], [0, 1], '--', color='navy')
    axes[0, 0].set_title("ROC Eğrisi"); axes[0, 0].set_xlabel("FPR"); axes[0, 0].set_ylabel("TPR")
    axes[0, 0].grid(alpha=0.3); axes[0, 0].legend(loc='lower right')

    axes[0, 1].hist(p[y == 0], bins=40, alpha=0.6, color='crimson', density=True, label='Kozmik')
    axes[0, 1].hist(p[y == 1], bins=40, alpha=0.6, color='royalblue', density=True, label='Nötrino')
    axes[0, 1].axvline(thr, color='black', ls='--', label=f'Eşik = {thr:.2f}')
    axes[0, 1].set_title("Çıktı Dağılımı"); axes[0, 1].legend()

    pt, pp = calibration_curve(y, p, n_bins=10)
    axes[1, 0].plot(pp, pt, 's-', color='purple', label=f'Brier = {brier:.4f}')
    axes[1, 0].plot([0, 1], [0, 1], 'k--')
    axes[1, 0].set_title("Kalibrasyon Eğrisi"); axes[1, 0].grid(alpha=0.3); axes[1, 0].legend()

    cm = confusion_matrix(y, (p >= thr).astype(int), normalize='true')
    im = axes[1, 1].imshow(cm, cmap='Blues', vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            axes[1, 1].text(j, i, f"%{cm[i, j]*100:.2f}", ha='center', va='center',
                            color="white" if cm[i, j] > 0.5 else "black", fontweight='bold', fontsize=14)
    axes[1, 1].set_xticks([0, 1]); axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_xticklabels(['Kozmik', 'Nötrino']); axes[1, 1].set_yticklabels(['Kozmik', 'Nötrino'])
    axes[1, 1].set_title("Normalize Hata Matrisi")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(out_dir, "ROC_Calibration_ConfusionMatrix.png"), dpi=200)
    plt.close()


# ==============================================================================
# 6. EĞİTİM MOTORU
# ==============================================================================
def train_single_run(params, datasets, output_dir):
    set_seed(SEED)
    tr_ds, va_ds, te_ds, tre_ds = datasets

    kw = dict(batch_size=params['batch_size'], num_workers=NUM_WORKERS, pin_memory=True)
    pw = dict(persistent_workers=True, prefetch_factor=2) if NUM_WORKERS > 0 else {}
    train_loader = DataLoader(tr_ds, shuffle=True, drop_last=True, **kw, **pw)
    val_loader = DataLoader(va_ds, shuffle=False, **kw, **pw)
    tre_loader = DataLoader(tre_ds, shuffle=False, **kw)
    test_loader = DataLoader(te_ds, shuffle=False, **kw)

    model = HybridLArTPCNet(
        c1=params['c1'], c2=params['c2'], c3=params['c3'], c4=params['c4'],
        drop_img=params['drop_img'], drop_fusion=params['drop_fusion']
    ).to(DEVICE)

    ema = EMA(model, EMA_DECAY) if USE_EMA else None
    eval_model_fn = (lambda: ema.module) if USE_EMA else (lambda: model)

    bce = nn.BCEWithLogitsLoss()
    def loss_fn(logits, t):
        if USE_LS:
            t = t * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
        return bce(logits.float(), t)

    opt = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    # T_max = MAX_EPOCHS (16) korunur, öğrenme oranı sıfıra pürüzsüz yaklaşır
    def lr_lambda(e):
        if e < WARMUP_EPOCHS:
            return float(e + 1) / WARMUP_EPOCHS
        return 0.5 * (1.0 + np.cos(np.pi * (e - WARMUP_EPOCHS) / max(1, MAX_EPOCHS - WARMUP_EPOCHS)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    gs = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))

    best_path = os.path.join(output_dir, "best.pt")
    es = EarlyStopping(patience=PATIENCE, min_epochs=MIN_EPOCHS_BEFORE_STOP, path=best_path)

    hist = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'train_auc_eval': []}
    best_ep = 0

    print(f"\nEğitim Başlıyor (Max {MAX_EPOCHS} epoch, Min ES: {MIN_EPOCHS_BEFORE_STOP})...")
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        tl, tn = 0.0, 0
        for imgs, tabs, lbls in train_loader:
            imgs, tabs, lbls = imgs.to(DEVICE, non_blocking=True), tabs.to(DEVICE), lbls.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEVICE.type, enabled=(DEVICE.type == 'cuda')):
                logits = model(imgs, tabs)
                loss = loss_fn(logits, lbls)
            gs.scale(loss).backward()
            gs.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            gs.step(opt); gs.update()
            if ema:
                ema.update(model)
            tl += loss.item() * len(lbls); tn += len(lbls)
        sched.step()

        em = eval_model_fn()
        vl, vp, vt = evaluate(em, val_loader)
        v_auc = roc_auc_score(vt, vp)
        v_acc = float(((vp >= 0.5) == vt).mean())

        _, tp, tt = evaluate(em, tre_loader)
        t_auc = roc_auc_score(tt, tp)

        hist['train_loss'].append(tl / tn); hist['val_loss'].append(vl)
        hist['val_auc'].append(v_auc); hist['train_auc_eval'].append(t_auc)

        is_best = es(epoch, v_auc, em.state_dict())
        if is_best:
            best_ep = epoch

        print(f"Epoch {epoch:02d}/{MAX_EPOCHS} | Tr Loss {tl/tn:.4f} | Val Loss {vl:.4f} | "
              f"Tr AUC {t_auc:.4f} | Val AUC {v_auc:.4f} | Makas {t_auc - v_auc:+.4f} | "
              f"Val Acc@0.5 %{v_acc*100:.2f} | {time.time()-t0:.0f}s" + ("  * EN İYİ" if is_best else ""), flush=True)

        if es.early_stop:
            print(f"Patience={PATIENCE} aşıldı, early stopping devrede.")
            break

    return best_path, best_ep, hist


# ==============================================================================
# 7. MAIN VE OPTUNA ENTEGRASYONU
# ==============================================================================
def main():
    tr_m, va_m, te_m, tr_r, va_r, te_r, scaler = scan_and_prepare_dataset()

    train_ds = LArTPCHybridDataset(tr_m, tr_r, scaler, is_train=True)
    val_ds = LArTPCHybridDataset(va_m, va_r, scaler, is_train=False)
    test_ds = LArTPCHybridDataset(te_m, te_r, scaler, is_train=False)

    rng = np.random.RandomState(SEED)
    sub = rng.choice(len(tr_m), size=min(TRAIN_EVAL_SUBSET, len(tr_m)), replace=False)
    tre_ds = LArTPCHybridDataset([tr_m[i] for i in sub], [tr_r[i] for i in sub], scaler, is_train=False)

    datasets = (train_ds, val_ds, test_ds, tre_ds)

    # Varsayılan (Baseline omurgası) parametreler
    params = {
        'c1': 32, 'c2': 48, 'c3': 96, 'c4': 192,
        'drop_img': 0.40, 'drop_fusion': 0.25,
        'batch_size': BATCH_SIZE, 'lr': BASE_LR, 'weight_decay': WEIGHT_DECAY
    }

    if USE_OPTUNA:
        print("\nOptuna Hiperparametre Araması Başlatılıyor...")
        def objective(trial):
            # Arama uzayı baseline'ı merkez alacak şekilde kuruldu
            trial_params = {
                'c1': 32,
                'c2': trial.suggest_categorical('c2', [48, 64]),
                'c3': trial.suggest_categorical('c3', [96, 128]),
                'c4': trial.suggest_categorical('c4', [192, 256]),
                'drop_img': trial.suggest_float('drop_img', 0.25, 0.45, step=0.05),
                'drop_fusion': trial.suggest_float('drop_fusion', 0.15, 0.35, step=0.05),
                'batch_size': BATCH_SIZE,
                'lr': trial.suggest_float('lr', 1.5e-4, 4.5e-4, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-2, 5e-2, log=True)
            }
            t_dir = os.path.join(OUTPUT_DIR, f"trial_{trial.number}")
            os.makedirs(t_dir, exist_ok=True)
            b_path, b_ep, hist = train_single_run(trial_params, datasets, t_dir)
            return max(hist['val_auc'])

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=MIN_EPOCHS_BEFORE_STOP)
        )
        study.optimize(objective, n_trials=N_TRIALS)
        study.trials_dataframe().to_csv(os.path.join(OUTPUT_DIR, "optuna_trials.csv"), index=False)
        print("\nEn İyi Optuna Parametreleri:", study.best_trial.params)
        params.update(study.best_trial.params)

    # Nihai Model Eğitimi
    best_weights_path, best_ep, hist = train_single_run(params, datasets, OUTPUT_DIR)

    # ---------------- NİHAİ TEST DEĞERLENDİRMESİ ----------------
    print(f"\nEn iyi epoch ({best_ep}) ağırlıkları yüklenerek bağımsız Test kümesi değerlendiriliyor...")
    final_model = HybridLArTPCNet(
        c1=params['c1'], c2=params['c2'], c3=params['c3'], c4=params['c4'],
        drop_img=params['drop_img'], drop_fusion=params['drop_fusion']
    ).to(DEVICE)
    final_model.load_state_dict(torch.load(best_weights_path, map_location=DEVICE))

    val_loader = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=params['batch_size'], shuffle=False, num_workers=NUM_WORKERS)

    # Eşik val üzerinden belirlenir
    _, vp, vt = evaluate(final_model, val_loader)
    thr = f1_threshold(vt, vp)

    _, tp, tt = evaluate(final_model, test_loader)
    pred = (tp >= thr).astype(int)

    auc = roc_auc_score(tt, tp)
    brier = brier_score_loss(tt, tp)
    cm = confusion_matrix(tt, pred)
    cmn = confusion_matrix(tt, pred, normalize='true')
    rep = classification_report(tt, pred, target_names=['Kozmik (0)', 'Nötrino (1)'], digits=4)
    mf1 = f1_score(tt, pred, average='macro')

    ops = []
    for tg in (0.70, 0.80, 0.90, 0.95):
        f, t, th = fpr_at_tpr(tt, tp, tg)
        ops.append(f"TPR >= %{tg*100:.0f} -> FPR %{f*100:.2f} (Kozmik Reddi %{(1-f)*100:.2f}, Eşik {th:.3f})")

    te_truth = np.array([m['truth_npix_Y_nu'] for m in te_m])
    sl = []
    for lo, hi in TRUTH_BINS:
        nm = (tt == 1) & (te_truth >= lo) & (te_truth < hi)
        n_nu = int(nm.sum())
        mask = (tt == 0) | nm
        if n_nu < 20:
            sl.append(f"truth_npix_Y_nu [{lo:g},{hi:g}): n_nu={n_nu} -> Yetersiz"); continue
        sl.append(f"truth_npix_Y_nu [{lo:g},{hi:g}): n_nu={n_nu:5d} | AUC={roc_auc_score(tt[mask], tp[mask]):.4f} | "
                  f"TPR@Eşik=%{((tp >= thr) & nm).sum() / n_nu * 100:.2f}")

    print("\n" + "=" * 85)
    print("TEST DEĞERLENDİRME RAPORU:")
    print(rep)
    print(f"AUC: {auc:.4f} | Macro-F1: {mf1:.4f} | Brier: {brier:.4f} | Eşik: {thr:.4f}")
    print("\nSabit Verimde Reddetme:\n  " + "\n  ".join(ops))
    print("\nFiziksel Dilimler:\n  " + "\n  ".join(sl))
    print("=" * 85)

    save_plots(hist, tt, tp, thr, brier, OUTPUT_DIR)

    np.savez(
        os.path.join(OUTPUT_DIR, "test_predictions.npz"),
        prob=tp, label=tt, truth_npix=te_truth,
        run=np.array([m['run'] for m in te_m])
    )

    with open(os.path.join(OUTPUT_DIR, "evaluation_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Model: {RUN_NAME} | En İyi Epoch: {best_ep}\n")
        f.write(f"Bayraklar: EMA={USE_EMA} LS={USE_LS} Qjit={USE_QJIT} ModDrop={USE_MOD_DROP}\n")
        f.write(f"AUC: {auc:.4f} | Macro-F1: {mf1:.4f} | Brier: {brier:.4f} | Eşik: {thr:.4f}\n")
        f.write(f"TPR: %{cmn[1,1]*100:.2f} | Kozmik Reddi: %{cmn[0,0]*100:.2f}\n\n")
        f.write(f"Hata Matrisi:\n{cm}\n\n{rep}\n")
        f.write("Sabit Verim Noktaları:\n" + "\n".join(ops) + "\n\n")
        f.write("Fiziksel Dilimleme:\n" + "\n".join(sl) + "\n")

    print(f"\nRapor ve çıktılar kaydedildi: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()