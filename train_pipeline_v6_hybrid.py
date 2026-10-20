import os
import glob
import csv
import copy
import zlib
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, brier_score_loss,
    classification_report, f1_score
)
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# 0. AYARLAR
# ==============================================================================
SEED = 42
DATA_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(DATA_DIR, "training_outputs_v7")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SIGNAL_MIN_TRUTH_NPIX_Y = 10   # Sinyal TANIMI (yalnızca nötrino etiketi için, truth)
PRESEL_MIN_RECO_NPIX_Y = 10    # Simetrik reco ön seçimi (iki sınıfa da)

MAX_EPOCHS = 20
BATCH_SIZE = 24
NUM_WORKERS = 6
BASE_LR = 1e-3
WARMUP_EPOCHS = 2
WEIGHT_DECAY = 2e-2
LABEL_SMOOTHING = 0.05
MODALITY_DROPOUT = 0.10
EMA_DECAY = 0.999
TRAIN_EVAL_SUBSET = 5000       # eval-modunda train metriği için alt küme
TRUTH_BINS = [(10, 30), (30, 100), (100, 300), (300, 1e9)]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

print("=" * 85)
print("LArTPC V7 HİBRİT (WIRE + TABULAR)")
print(f"Çalışma Dizini: {DATA_DIR}")
print(f"Çıktı Klasörü : {OUTPUT_DIR}")
print(f"Donanım       : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU           : {torch.cuda.get_device_name(0)}")
print("=" * 85)


# ==============================================================================
# 1. SEYREK GÖRÜNTÜ ÇÖZÜCÜ (v6 ile aynı gerçek decoder)
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


def cnn_girdisi(duzlemler, satir_kat=2, genislik=864, q_scale=None):
    """Seyrek düzlemleri (3, 504, 864) görüntüye çevirir.
    q_scale: düzlem başına yük ölçek çarpanı (augmentation için, log'dan ÖNCE uygulanır)."""
    yukseklik = duzlemler[2]['nsatir'] // satir_kat
    x = np.zeros((3, yukseklik, genislik), dtype=np.float32)
    for c, p in enumerate(duzlemler):
        tel_kat = -(-p['ntel'] // genislik)
        r = p['satir'].astype(int) // satir_kat
        w = p['tel'].astype(int) // tel_kat
        m = (r < yukseklik) & (w < genislik)
        s = 1.0 if q_scale is None else float(q_scale[c])
        np.add.at(x[c], (r[m], w[m]), np.clip(p['q'][m], 0, None) * s)
    return np.log10(1.0 + x / 1000.0)


# ==============================================================================
# 2. TABLOSAL ÖZELLİKLER (14 adet, log'lu, reco tabanlı)
# ==============================================================================
def _L(x):
    return float(np.log10(1.0 + max(x, 0.0)))


def extract_reco_tabular_features(row):
    q_u = float(row.get('reco_Q_U', 0.0) or 0.0)
    q_v = float(row.get('reco_Q_V', 0.0) or 0.0)
    q_y = float(row.get('reco_Q_Y', 0.0) or 0.0)
    n_u = float(row.get('reco_npix_U', 0.0) or 0.0)
    n_v = float(row.get('reco_npix_V', 0.0) or 0.0)
    n_y = float(row.get('reco_npix_Y', 0.0) or 0.0)

    q_tot = q_u + q_v + q_y
    n_tot = n_u + n_v + n_y
    charge_density = q_tot / (n_tot + 1e-4)
    induction_asym = (q_u - q_v) / (q_u + q_v + 1e-6)
    induction_asym = float(np.clip(induction_asym, -1.0, 1.0))

    feats = [
        _L(q_u), _L(q_v), _L(q_y), _L(q_tot),                      # 1-4
        _L(n_u), _L(n_v), _L(n_y), _L(n_tot),                      # 5-8
        _L(charge_density),                                        # 9
        float(np.log10((max(q_u, 0) + 1.0) / (max(q_y, 0) + 1.0))),  # 10
        float(np.log10((max(q_v, 0) + 1.0) / (max(q_y, 0) + 1.0))),  # 11
        induction_asym,                                            # 12
        float(np.log10((max(n_u, 0) + 1.0) / (max(n_y, 0) + 1.0))),  # 13
        float(np.log10((max(n_v, 0) + 1.0) / (max(n_y, 0) + 1.0))),  # 14
    ]
    return np.array(feats, dtype=np.float32)


# ==============================================================================
# 3. VERİ TOPLAMA (simetrik reco kesimi) VE BÖLME
# ==============================================================================
def collect_events(csv_files, label, prefix_tag, max_n=None):
    meta, tab, groups = [], [], []
    stats = {'read': 0, 'fail_signal_def': 0, 'fail_presel': 0, 'kept': 0}
    for f_csv in csv_files:
        f_bin = f_csv.replace("_events.csv", "") + "_images.bin"
        if not os.path.exists(f_bin):
            continue
        with open(f_csv, newline='') as f:
            for row in csv.DictReader(f):
                stats['read'] += 1
                truth_npix = float(row.get('truth_npix_Y_nu', 0) or 0)

                # (1) Sinyal tanımı: sadece nötrino sınıfı için
                if label == 1.0 and truth_npix < SIGNAL_MIN_TRUTH_NPIX_Y:
                    stats['fail_signal_def'] += 1
                    continue
                # (2) Simetrik reco ön seçimi: iki sınıfa da aynı
                if float(row.get('reco_npix_Y', 0) or 0) < PRESEL_MIN_RECO_NPIX_Y:
                    stats['fail_presel'] += 1
                    continue

                run_id = int(row.get('run', 0))
                meta.append({
                    'bin_path': f_bin,
                    'offset': int(row['img_offset']),
                    'nbytes': int(row['img_nbytes']),
                    'label': label,
                    'run': run_id,
                    'event_id': int(row['event_id']),
                    'truth_npix_Y_nu': truth_npix if label == 1.0 else 0.0,  # sadece analiz
                })
                tab.append(extract_reco_tabular_features(row))
                groups.append(f"{prefix_tag}_{run_id}")
                stats['kept'] += 1
                if max_n and stats['kept'] >= max_n:
                    return meta, tab, groups, stats
    return meta, tab, groups, stats


def prepare_data(max_samples_per_class=None):
    print("Veri seti taranıyor...")
    nu_csvs = sorted(set(glob.glob("**/*neutrino*_events.csv", recursive=True) + glob.glob("*neutrino*_events.csv")))
    cos_csvs = sorted(set(glob.glob("**/*cosmic*_events.csv", recursive=True) + glob.glob("*cosmic*_events.csv")))
    if not nu_csvs or not cos_csvs:
        raise FileNotFoundError("Gerekli _events.csv dosyaları bulunamadı!")

    nu_meta, nu_tab, nu_grp, nu_st = collect_events(nu_csvs, 1.0, "nu", max_samples_per_class)
    cos_meta, cos_tab, cos_grp, cos_st = collect_events(cos_csvs, 0.0, "cos", max_samples_per_class)
    print(f"Nötrino: {nu_st}")
    print(f"Kozmik : {cos_st}")

    all_meta = nu_meta + cos_meta
    all_tab = np.array(nu_tab + cos_tab, dtype=np.float32)
    labels = np.array([m['label'] for m in all_meta])
    groups = np.array(nu_grp + cos_grp)
    indices = np.arange(len(all_meta))

    sgkf_test = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_val_idx, te_idx = next(sgkf_test.split(indices, labels, groups=groups))
    sgkf_val = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr_sub, val_sub = next(sgkf_val.split(train_val_idx, labels[train_val_idx], groups=groups[train_val_idx]))
    tr_idx, val_idx = train_val_idx[tr_sub], train_val_idx[val_sub]

    tr_runs, val_runs, te_runs = set(groups[tr_idx]), set(groups[val_idx]), set(groups[te_idx])
    assert not (tr_runs & val_runs), "Train-Val ortak run!"
    assert not (tr_runs & te_runs), "Train-Test ortak run!"
    assert not (val_runs & te_runs), "Val-Test ortak run!"

    scaler = RobustScaler()
    tr_tab = scaler.fit_transform(all_tab[tr_idx])
    val_tab = scaler.transform(all_tab[val_idx])
    te_tab = scaler.transform(all_tab[te_idx])

    def clean(a):
        return np.clip(np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0), -10, 10).astype(np.float32)

    tr_tab, val_tab, te_tab = clean(tr_tab), clean(val_tab), clean(te_tab)
    tr_meta = [all_meta[i] for i in tr_idx]
    val_meta = [all_meta[i] for i in val_idx]
    te_meta = [all_meta[i] for i in te_idx]

    print("\nBÖLÜMLEME RAPORU:")
    for ad, mt, rn, idx in [("Train", tr_meta, tr_runs, tr_idx), ("Val", val_meta, val_runs, val_idx),
                            ("Test", te_meta, te_runs, te_idx)]:
        print(f"  {ad:5s}: {len(mt):,} olay | {len(rn)} run grubu | Nötrino oranı %{labels[idx].mean()*100:.1f}")
    print("  Not: gruplar sınıf-saf (nu_/cos_ önekli); tek bölme = tek ölçüm.\n")
    return tr_meta, val_meta, te_meta, tr_tab, val_tab, te_tab


# ==============================================================================
# 4. DATASET
# ==============================================================================
class LArTPCDataset(Dataset):
    def __init__(self, metadata, tabular, is_train=False):
        self.metadata = metadata
        self.tabular = torch.tensor(tabular, dtype=torch.float32)
        self.is_train = is_train

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        with open(item['bin_path'], 'rb') as f:
            f.seek(item['offset'])
            buf = f.read(item['nbytes'])

        planes = kayit_coz(buf)
        q_scale = np.random.uniform(0.9, 1.1, size=3) if self.is_train else None
        img = torch.from_numpy(cnn_girdisi(planes, q_scale=q_scale))  # (3, 504, 864)

        tab = self.tabular[idx].clone()
        if self.is_train and random.random() < MODALITY_DROPOUT:
            tab.zero_()

        lbl = torch.tensor(item['label'], dtype=torch.float32)
        return img, tab, lbl


# ==============================================================================
# 5. MODEL (GroupNorm, stride-2 stem, CoordConv, attention+max pooling)
# ==============================================================================
class ResidualBlockGN(nn.Module):
    def __init__(self, channels, drop_prob=0.10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(p=drop_prob),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv(x))


class AttentionPool2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        a = F.softmax(self.attn(x).float().view(B, 1, -1), dim=-1)
        out = torch.bmm(x.float().view(B, C, -1), a.transpose(1, 2)).view(B, C)
        return out


class HybridLArTPCNetV7(nn.Module):
    def __init__(self, num_tab_features=14):
        super().__init__()
        # Giriş: 3 düzlem + 2 koordinat kanalı = 5
        self.stage1 = nn.Sequential(                       # 504x864 -> 252x432
            nn.Conv2d(5, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.SiLU(inplace=True),
            ResidualBlockGN(32, 0.05),
        )
        self.stage2 = nn.Sequential(                       # -> 126x216
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64), nn.SiLU(inplace=True),
            ResidualBlockGN(64, 0.05),
        )
        self.stage3 = nn.Sequential(                       # -> 63x108
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128), nn.SiLU(inplace=True),
            ResidualBlockGN(128, 0.10),
        )
        self.stage4 = nn.Sequential(                       # -> 32x54
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 256), nn.SiLU(inplace=True),
            ResidualBlockGN(256, 0.15),
        )
        self.att_pool = AttentionPool2d(256)

        self.img_proj = nn.Sequential(
            nn.Linear(256 * 2, 64), nn.LayerNorm(64), nn.SiLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(64, 32), nn.LayerNorm(32),
        )
        self.tab_proj = nn.Sequential(
            nn.Linear(num_tab_features, 64), nn.LayerNorm(64), nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.SiLU(inplace=True),
        )
        self.gate = nn.Sequential(nn.Linear(32 + 64, 32), nn.Sigmoid())
        self.fusion = nn.Sequential(
            nn.Linear(32 + 64, 64), nn.LayerNorm(64), nn.SiLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, 1),
        )

    def forward(self, img, tab):
        B, _, H, W = img.shape
        yc = torch.linspace(-1, 1, H, device=img.device, dtype=img.dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        xc = torch.linspace(-1, 1, W, device=img.device, dtype=img.dtype).view(1, 1, 1, W).expand(B, 1, H, W)
        x = torch.cat([img, yc, xc], dim=1)

        x = self.stage4(self.stage3(self.stage2(self.stage1(x))))
        h_pool = torch.cat([self.att_pool(x), F.adaptive_max_pool2d(x.float(), 1).flatten(1)], dim=1)
        h_img = self.img_proj(h_pool)
        h_tab = self.tab_proj(tab)

        g = self.gate(torch.cat([h_img, h_tab], dim=1))
        fused = torch.cat([h_img * g, h_tab], dim=1)
        return self.fusion(fused).squeeze(1)


# ==============================================================================
# 6. EMA VE DEĞERLENDİRME YARDIMCILARI
# ==============================================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.step = 0

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        for pe, p in zip(self.module.parameters(), model.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for be, b in zip(self.module.buffers(), model.buffers()):
            be.copy_(b)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    probs, targets, tot_loss, n = [], [], 0.0, 0
    bce = nn.BCEWithLogitsLoss()
    for imgs, tabs, lbls in loader:
        imgs, tabs, lbls = imgs.to(DEVICE), tabs.to(DEVICE), lbls.to(DEVICE)
        with torch.amp.autocast(DEVICE.type, enabled=(DEVICE.type == 'cuda')):
            logits = model(imgs, tabs)
        loss = bce(logits.float(), lbls)
        tot_loss += loss.item() * len(lbls)
        n += len(lbls)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        targets.append(lbls.cpu().numpy())
    return tot_loss / max(n, 1), np.concatenate(probs), np.concatenate(targets)


def fpr_at_tpr(y, p, target_tpr):
    fpr, tpr, thr = roc_curve(y, p)
    i = int(np.searchsorted(tpr, target_tpr, side='left'))
    i = min(i, len(fpr) - 1)
    return float(fpr[i]), float(tpr[i]), float(thr[i])


def best_acc_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    n_pos, n_neg = y.sum(), (1 - y).sum()
    acc = (tpr * n_pos + (1 - fpr) * n_neg) / (n_pos + n_neg)
    i = int(np.argmax(acc))
    return float(thr[i]), float(acc[i])


# ==============================================================================
# 7. GRAFİKLER
# ==============================================================================
def save_plots(hist, y, p, thr, brier, out_dir):
    ep = range(1, len(hist['train_loss']) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(20, 5), facecolor='white')
    ax[0].plot(ep, hist['train_loss'], 'b-o', label='Train (smoothed, train-mode)')
    ax[0].plot(ep, hist['val_loss'], 'r--s', label='Val (EMA)')
    ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].grid(alpha=.3); ax[0].legend()
    ax[1].plot(ep, hist['train_auc_eval'], 'b-o', label='Train AUC (eval-mod, alt küme)')
    ax[1].plot(ep, hist['val_auc'], 'r--s', label='Val AUC (EMA)')
    ax[1].set_title("AUC (adil karşılaştırma)"); ax[1].set_xlabel("Epoch"); ax[1].grid(alpha=.3); ax[1].legend()
    ax[2].plot(ep, hist['lr'], 'g-o')
    ax[2].set_title("Öğrenme Hızı"); ax[2].set_xlabel("Epoch"); ax[2].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Training_Curves.png"), dpi=200, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), facecolor='white')
    fpr, tpr, _ = roc_curve(y, p)
    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2.5, label=f'AUC = {roc_auc_score(y, p):.4f}')
    axes[0, 0].plot([0, 1], [0, 1], '--', color='navy')
    axes[0, 0].set_title("ROC"); axes[0, 0].set_xlabel("FPR (kozmik)"); axes[0, 0].set_ylabel("TPR (nötrino)")
    axes[0, 0].grid(alpha=.3); axes[0, 0].legend(loc='lower right')

    axes[0, 1].hist(p[y == 0], bins=40, alpha=.6, color='crimson', density=True, label='Kozmik')
    axes[0, 1].hist(p[y == 1], bins=40, alpha=.6, color='royalblue', density=True, label='Nötrino')
    axes[0, 1].axvline(thr, color='k', ls='--', label=f'Eşik={thr:.2f}')
    axes[0, 1].set_title("Çıktı Dağılımı"); axes[0, 1].set_xlabel("P(nötrino)"); axes[0, 1].legend()

    pt, pp = calibration_curve(y, p, n_bins=10)
    axes[1, 0].plot(pp, pt, 's-', color='purple', label=f'Brier={brier:.4f}')
    axes[1, 0].plot([0, 1], [0, 1], 'k--')
    axes[1, 0].set_title("Kalibrasyon"); axes[1, 0].grid(alpha=.3); axes[1, 0].legend()

    cm = confusion_matrix(y, (p >= thr).astype(int), normalize='true')
    im = axes[1, 1].imshow(cm, cmap='Blues', vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            axes[1, 1].text(j, i, f"%{cm[i, j]*100:.2f}", ha='center', va='center',
                            color="white" if cm[i, j] > .5 else "black", fontweight='bold', fontsize=14)
    axes[1, 1].set_xticks([0, 1]); axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_xticklabels(['Kozmik', 'Nötrino']); axes[1, 1].set_yticklabels(['Kozmik', 'Nötrino'])
    axes[1, 1].set_title("Normalize Hata Matrisi")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(out_dir, "ROC_Calibration_ConfusionMatrix.png"), dpi=200, bbox_inches='tight')
    plt.close()


# ==============================================================================
# 8. ANA EĞİTİM
# ==============================================================================
def main():
    tr_meta, val_meta, te_meta, tr_tab, val_tab, te_tab = prepare_data()

    train_ds = LArTPCDataset(tr_meta, tr_tab, is_train=True)
    val_ds = LArTPCDataset(val_meta, val_tab, is_train=False)
    test_ds = LArTPCDataset(te_meta, te_tab, is_train=False)

    # Eval-modunda ölçülecek sabit train alt kümesi
    rng = np.random.RandomState(SEED)
    sub_idx = rng.choice(len(tr_meta), size=min(TRAIN_EVAL_SUBSET, len(tr_meta)), replace=False)
    train_eval_ds = LArTPCDataset([tr_meta[i] for i in sub_idx], tr_tab[sub_idx], is_train=False)

    # Sağlık kontrolü: görüntüler sıfır olmamalı
    print("Görüntü sağlık kontrolü:")
    for i in range(5):
        im, tb, lb = val_ds[i]
        print(f"  #{i} label={int(lb)} toplam={float(im.sum()):.1f} dolu_oran={float((im > 0).float().mean()):.4f} "
              f"tab_std={float(tb.std()):.3f}")
        assert float(im.sum()) > 0, "Sıfır görüntü! Decoder kontrol edilmeli."

    kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    pw = dict(persistent_workers=True, prefetch_factor=2) if NUM_WORKERS > 0 else {}
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **kw, **pw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw, **pw)
    train_eval_loader = DataLoader(train_eval_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    model = HybridLArTPCNetV7(num_tab_features=14).to(DEVICE)
    ema = EMA(model, EMA_DECAY)
    bce = nn.BCEWithLogitsLoss()

    def smoothed_loss(logits, targets):
        t = targets * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
        return bce(logits.float(), t)

    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / WARMUP_EPOCHS
        denom = max(1, MAX_EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1.0 + np.cos(np.pi * (epoch - WARMUP_EPOCHS) / denom))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    grad_scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))

    hist = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'train_auc_eval': [], 'lr': []}
    best_auc, best_epoch = 0.0, 0
    best_path = os.path.join(OUTPUT_DIR, "best_v7_ema.pt")
    last_path = os.path.join(OUTPUT_DIR, "last_v7_ema.pt")

    print("\nEğitim başlıyor...")
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        tr_loss, tr_n = 0.0, 0
        cur_lr = optimizer.param_groups[0]['lr']

        for imgs, tabs, lbls in train_loader:
            imgs, tabs, lbls = imgs.to(DEVICE, non_blocking=True), tabs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEVICE.type, enabled=(DEVICE.type == 'cuda')):
                logits = model(imgs, tabs)
                loss = smoothed_loss(logits, lbls)

            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)                     # clipping'ten önce şart
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            ema.update(model)

            tr_loss += loss.item() * len(lbls)
            tr_n += len(lbls)

        scheduler.step()
        ep_tr_loss = tr_loss / tr_n

        # Val ve train alt kümesi, ikisi de eval-modunda ve EMA ağırlıklarıyla
        val_loss, vp, vt = evaluate(ema.module, val_loader)
        val_auc = roc_auc_score(vt, vp)
        _, tp, tt = evaluate(ema.module, train_eval_loader)
        tr_auc_eval = roc_auc_score(tt, tp)

        hist['train_loss'].append(ep_tr_loss); hist['val_loss'].append(val_loss)
        hist['val_auc'].append(val_auc); hist['train_auc_eval'].append(tr_auc_eval); hist['lr'].append(cur_lr)

        print(f"Epoch {epoch:02d}/{MAX_EPOCHS} | LR {cur_lr:.2e} | Tr Loss {ep_tr_loss:.4f} | Val Loss {val_loss:.4f} | "
              f"Tr AUC(eval) {tr_auc_eval:.4f} | Val AUC {val_auc:.4f} | AUC makası {tr_auc_eval - val_auc:+.4f} | "
              f"{time.time() - t0:.0f}s")

        torch.save(ema.module.state_dict(), last_path)
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            torch.save(ema.module.state_dict(), best_path)
            print(f"   * En iyi model kaydedildi (Val AUC {best_auc:.4f})")

    # ==========================================================================
    # TEST
    # ==========================================================================
    print("\n" + "=" * 80)
    print(f"En iyi model (epoch {best_epoch}) yükleniyor, test değerlendirmesi...")
    print("=" * 80)
    final = copy.deepcopy(ema.module)
    final.load_state_dict(torch.load(best_path, map_location=DEVICE))

    _, vp, vt = evaluate(final, val_loader)
    thr, val_best_acc = best_acc_threshold(vt, vp)          # eşik val'den kilitlenir
    print(f"Val'den kilitlenen eşik (max accuracy): {thr:.4f} (Val acc {val_best_acc*100:.2f}%)")

    _, tp, tt = evaluate(final, test_loader)
    pred = (tp >= thr).astype(int)
    auc = roc_auc_score(tt, tp)
    brier = brier_score_loss(tt, tp)
    cm = confusion_matrix(tt, pred)
    cm_n = confusion_matrix(tt, pred, normalize='true')
    acc = float((pred == tt).mean())
    rep = classification_report(tt, pred, target_names=['Kozmik (0)', 'Nötrino (1)'], digits=4)
    macro_f1 = f1_score(tt, pred, average='macro')

    op_lines = []
    for target in (0.70, 0.80, 0.90, 0.95):
        fpr_t, tpr_t, th_t = fpr_at_tpr(tt, tp, target)
        op_lines.append(f"TPR>=%{target*100:.0f} -> FPR %{fpr_t*100:.2f} (kozmik reddi %{(1-fpr_t)*100:.2f}, gerçek TPR %{tpr_t*100:.2f}, eşik {th_t:.3f})")

    # Truth_npix dilimleri: ilgili nötrino dilimi vs TÜM kozmikler
    te_truth = np.array([m['truth_npix_Y_nu'] for m in te_meta])
    slice_lines = []
    for lo, hi in TRUTH_BINS:
        mask = (tt == 0) | ((tt == 1) & (te_truth >= lo) & (te_truth < hi))
        n_nu = int(((tt == 1) & (te_truth >= lo) & (te_truth < hi)).sum())
        if n_nu < 20 or len(np.unique(tt[mask])) < 2:
            slice_lines.append(f"truth_npix_Y_nu [{lo:g},{hi:g}): n_nu={n_nu} -> yetersiz örnek")
            continue
        s_auc = roc_auc_score(tt[mask], tp[mask])
        tpr_s = float(((tp >= thr) & (tt == 1) & (te_truth >= lo) & (te_truth < hi)).sum() / n_nu)
        slice_lines.append(f"truth_npix_Y_nu [{lo:g},{hi:g}): n_nu={n_nu:6d} | AUC={s_auc:.4f} | TPR@eşik=%{tpr_s*100:.2f}")

    print("\n" + rep)
    print(f"Accuracy={acc*100:.2f}%  Macro-F1={macro_f1:.4f}  AUC={auc:.4f}  Brier={brier:.4f}")
    print("\nSabit sinyal veriminde arka plan reddi:")
    print("\n".join("  " + l for l in op_lines))
    print("\nTruth_npix_Y_nu dilimleri (dilim vs tüm kozmikler):")
    print("\n".join("  " + l for l in slice_lines))

    save_plots(hist, tt, tp, thr, brier, OUTPUT_DIR)
    np.savez(os.path.join(OUTPUT_DIR, "test_predictions.npz"), prob=tp, label=tt, truth_npix=te_truth,
             run=np.array([m['run'] for m in te_meta]), event_id=np.array([m['event_id'] for m in te_meta]))

    rep_path = os.path.join(OUTPUT_DIR, "v7_evaluation_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("LArTPC V7 HİBRİT PERFORMANS RAPORU\n" + "=" * 80 + "\n")
        f.write(f"En iyi epoch                 : {best_epoch}\n")
        f.write(f"Test ROC-AUC                 : {auc:.4f}\n")
        f.write(f"Test Accuracy                : {acc*100:.2f}%\n")
        f.write(f"Macro F1                     : {macro_f1:.4f}\n")
        f.write(f"Brier                        : {brier:.4f}\n")
        f.write(f"Val'den kilitli eşik         : {thr:.4f}\n")
        f.write(f"TPR (eşikte)                 : %{cm_n[1, 1]*100:.2f}\n")
        f.write(f"Kozmik reddi (eşikte)        : %{cm_n[0, 0]*100:.2f}\n\n")
        f.write(f"Confusion Matrix:\n{cm}\n\n{rep}\n")
        f.write("Sabit sinyal veriminde arka plan reddi:\n" + "\n".join(op_lines) + "\n\n")
        f.write("Truth_npix_Y_nu dilimleri:\n" + "\n".join(slice_lines) + "\n")
    print(f"\nRapor: {rep_path}\nGrafikler: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()