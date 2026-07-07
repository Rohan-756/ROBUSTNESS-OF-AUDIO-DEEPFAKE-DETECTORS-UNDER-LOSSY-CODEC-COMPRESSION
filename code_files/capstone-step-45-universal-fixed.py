# ============================================================
#  STEP 4+5 — Codec Augmentation Ablation (Universal Fixed)
#
#  SESSION 1 assignments:
#    M1 → MODEL='AASIST', CODEC_SET='mp3_only'   (Run 2)
#    M2 → MODEL='AASIST', CODEC_SET='opus_only'  (Run 3)
#    M3 → MODEL='AASIST', CODEC_SET='all'        (Run 4)
#    M4 → MODEL='AASIST', CODEC_SET='mp3_opus'   (Run 5)
#
#  SESSION 2 assignments:
#    M1 → MODEL='AASIST',  CODEC_SET='aac_only'  (Run 6)
#    M2 → MODEL='RawNet2', CODEC_SET='mp3_only'  (Run 7)
#    M3 → MODEL='RawNet2', CODEC_SET='all'       (Run 8)
#    M4 → spare
#
#  FIXES APPLIED IN THIS VERSION:
#    [1] Force-fresh AASIST clone every session (prevents stale .pyc)
#    [2] Correct RawNet2Spoof.py written to disk BEFORE any import
#    [3] Module cache flushed at start of Phase 2 AND Phase 3
#    [4] Evaluation always uses last_checkpoint.pth (Epoch 30)
# ============================================================

# ▼▼▼  CHANGE THESE TWO LINES PER MEMBER PER SESSION  ▼▼▼
MODEL     = 'RawNet2'   # 'AASIST' or 'RawNet2'
CODEC_SET = 'all'       # 'mp3_only' | 'opus_only' | 'all' | 'mp3_opus' | 'aac_only'
# ▲▲▲  CHANGE THESE TWO LINES PER MEMBER PER SESSION  ▲▲▲

NUM_EPOCHS = 30
AUG_SEED   = 42

RUN_NAMES = {
    ('AASIST',  'mp3_only'):  'run2_aasist_mp3_only',
    ('AASIST',  'opus_only'): 'run3_aasist_opus_only',
    ('AASIST',  'all'):       'run4_aasist_all_codecs',
    ('AASIST',  'mp3_opus'):  'run5_aasist_mp3_opus',
    ('AASIST',  'aac_only'):  'run6_aasist_aac_only',
    ('RawNet2', 'mp3_only'):  'run7_rawnet2_mp3_only',
    ('RawNet2', 'all'):       'run8_rawnet2_all_codecs',
}
key = (MODEL, CODEC_SET)
assert key in RUN_NAMES, f'Unknown combination MODEL={MODEL}, CODEC_SET={CODEC_SET}'
RUN_NAME = RUN_NAMES[key]

BATCH_SIZES = {'AASIST': 16, 'RawNet2': 24}
BATCH_SIZE  = BATCH_SIZES[MODEL]

print(f'Run     : {RUN_NAME}')
print(f'Model   : {MODEL}')
print(f'Codec   : {CODEC_SET}')
print(f'Batch   : {BATCH_SIZE}')

# ── Imports & Setup ───────────────────────────────────────────────
import subprocess, sys, os, json, shutil, random, time, warnings
import numpy as np
from pathlib import Path

subprocess.run(['apt-get', 'install', '-y', 'ffmpeg'], check=True, capture_output=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'einops', 'scikit-learn', 'soundfile', 'librosa'], check=True)

import torch, soundfile as sf, librosa
from torch.utils.data import Dataset, DataLoader
print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')

WORK_DIR    = Path('/kaggle/working/project4')
AASIST_DIR  = WORK_DIR / 'aasist'
AUG_DIR     = WORK_DIR / f'augmented_train_{RUN_NAME}'
MODEL_DIR   = WORK_DIR / f'model_{RUN_NAME}'
RESULTS_DIR = WORK_DIR / 'results'
CKPT_PATH   = MODEL_DIR / 'last_checkpoint.pth'
for d in [WORK_DIR, AUG_DIR, MODEL_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def _find(candidates):
    for p in candidates:
        if Path(p).exists(): return Path(p)
    return None

DATA_BASE = _find([
    '/kaggle/input/asvpoof-2019-dataset/LA/LA',
    '/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA',
])
assert DATA_BASE, '❌ ASVspoof 2019 not found — attach via Add Data'
print(f'✓ ASVspoof: {DATA_BASE}')

WAVEFAKE_INPUT = _find([
    '/kaggle/input/wavefake-vocoders-subset',
    '/kaggle/input/datasets/rohannrahulshah/wavefake-vocoders-subset',
])
assert WAVEFAKE_INPUT, '❌ wavefake-vocoders-subset not found'
print(f'✓ WaveFake: {WAVEFAKE_INPUT}')

PROTO_TRAIN = DATA_BASE / 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
PROTO_DEV   = DATA_BASE / 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
PROTO_EVAL  = DATA_BASE / 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
TRAIN_FLAC  = DATA_BASE / 'ASVspoof2019_LA_train/flac'
DEV_FLAC    = DATA_BASE / 'ASVspoof2019_LA_dev/flac'
EVAL_FLAC   = DATA_BASE / 'ASVspoof2019_LA_eval/flac'
for p in [PROTO_TRAIN, PROTO_DEV, TRAIN_FLAC, DEV_FLAC]:
    assert p.exists(), f'❌ Missing: {p}'
print('✓ All paths verified')

# ── Recovery Cell ─────────────────────────────────────────────────
print(f"\nSearching for crashed run data for '{RUN_NAME}' in /kaggle/input/...")
found_ckpt = None
for p in Path('/kaggle/input').rglob('last_checkpoint.pth'):
    if RUN_NAME in str(p):
        found_ckpt = p; break

if not found_ckpt:
    print('❌ Could not find last_checkpoint.pth — will train from scratch')
else:
    src_dir = found_ckpt.parent
    print(f'✓ Found model data at: {src_dir}')
    shutil.copy(found_ckpt, CKPT_PATH)
    if (src_dir / 'best.pth').exists():
        shutil.copy(src_dir / 'best.pth', MODEL_DIR / 'best.pth')
    src_aug = src_dir.parent / f'augmented_train_{RUN_NAME}'
    if src_aug.exists():
        print('Copying augmented audio (~25,000 files, ~2 min)...')
        shutil.copytree(src_aug, AUG_DIR, dirs_exist_ok=True)
        print('✓ Audio recovered! Phase 1 will be skipped.')
    else:
        print('⚠️  No augmented audio found — Phase 1 will re-run.')
    print('\n✓ Recovery complete!')

# ── FIX [1 + 2]: Clone AASIST fresh + Write correct RawNet2Spoof.py ──
if AASIST_DIR.exists():
    shutil.rmtree(AASIST_DIR)
subprocess.run(['git', 'clone', 'https://github.com/clovaai/aasist.git',
                str(AASIST_DIR)], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                '-r', str(AASIST_DIR / 'requirements.txt')], check=True)
for fpath in AASIST_DIR.rglob('*.py'):
    txt = fpath.read_text(encoding='utf-8')
    for old, new in [('np.float)', 'float)'), ('np.float,', 'float,'),
                     ('np.float ', 'float '), ('num_workers=4', 'num_workers=0'),
                     ('num_workers=8', 'num_workers=0')]:
        txt = txt.replace(old, new)
    fpath.write_text(txt, encoding='utf-8')

# Write correct RawNet2Spoof.py BEFORE sys.path.insert or any import
_RAWNET2_CODE = '''\
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np

class FixedSincBank(nn.Module):
    """Fixed (non-learnable) mel-spaced sinc filterbank.
    Weights are computed deterministically and stored as a plain tensor,
    so they produce ZERO entries in the model state_dict."""
    def __init__(self, out_channels, kernel_size, sample_rate=16000):
        super().__init__()
        if kernel_size % 2 == 0: kernel_size += 1
        to_mel = lambda hz: 2595 * np.log10(1 + hz / 700)
        to_hz  = lambda mel: 700 * (10 ** (mel / 2595) - 1)
        hz  = to_hz(np.linspace(to_mel(30), to_mel(sample_rate / 2 - 1), out_channels + 1))
        t   = np.arange(-(kernel_size // 2), kernel_size // 2 + 1) / sample_rate
        win = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(kernel_size) / (kernel_size - 1))
        flt = np.zeros((out_channels, 1, kernel_size), np.float32)
        for i in range(out_channels):
            h = 2*hz[i+1]*np.sinc(2*hz[i+1]*t) - 2*hz[i]*np.sinc(2*hz[i]*t)
            h *= win
            h /= (np.abs(h).max() + 1e-8)
            flt[i, 0] = h
        self._flt = torch.FloatTensor(flt)   # plain attr — NOT in state_dict
    def forward(self, x):
        return torch.abs(F.conv1d(x, self._flt.to(x.device)))


class Res_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = nn.BatchNorm1d(nb_filts[0])
        self.conv1 = nn.Conv1d(nb_filts[0], nb_filts[1], 3, padding=1)   # bias=True (default)
        self.bn2   = nn.BatchNorm1d(nb_filts[1])
        self.conv2 = nn.Conv1d(nb_filts[1], nb_filts[1], 3, padding=1)   # bias=True
        self.relu  = nn.LeakyReLU(0.3)
        self.mp    = nn.MaxPool1d(3)
        if nb_filts[0] != nb_filts[1]:
            self.conv_downsample = nn.Conv1d(nb_filts[0], nb_filts[1], 1) # bias=True
    def forward(self, x):
        identity = x
        out = x if self.first else self.relu(self.bn1(x))
        out = self.relu(self.bn2(self.conv1(out)))
        out = self.conv2(out)
        if hasattr(self, "conv_downsample"):
            identity = self.conv_downsample(x)
        return self.mp(out + identity)


class Model(nn.Module):
    def __init__(self, d_args):
        super().__init__()
        filts = d_args["filts"]   # [20, [20,20], [20,128], [128,128]]
        # FixedSincBank contributes ZERO keys to state_dict — matches checkpoint exactly
        self._sinc    = FixedSincBank(filts[0], d_args["first_conv"])
        self.first_bn = nn.BatchNorm1d(filts[0])
        self.selu     = nn.SELU()
        # 6 separate block attributes — matches checkpoint keys block0…block5
        self.block0 = nn.Sequential(Res_block(filts[1], first=True))
        self.block1 = nn.Sequential(Res_block(filts[1], first=False))
        self.block2 = nn.Sequential(Res_block(filts[2], first=False))  # [20→128]
        self.block3 = nn.Sequential(Res_block(filts[3], first=False))  # [128→128]
        self.block4 = nn.Sequential(Res_block(filts[3], first=False))
        self.block5 = nn.Sequential(Res_block(filts[3], first=False))
        # Channel-attention per block — Linear(N, N) wrapped in Sequential
        self.fc_attention0 = nn.Sequential(nn.Linear(filts[1][-1], filts[1][-1]))
        self.fc_attention1 = nn.Sequential(nn.Linear(filts[1][-1], filts[1][-1]))
        self.fc_attention2 = nn.Sequential(nn.Linear(filts[2][-1], filts[2][-1]))
        self.fc_attention3 = nn.Sequential(nn.Linear(filts[3][-1], filts[3][-1]))
        self.fc_attention4 = nn.Sequential(nn.Linear(filts[3][-1], filts[3][-1]))
        self.fc_attention5 = nn.Sequential(nn.Linear(filts[3][-1], filts[3][-1]))
        self.bn_before_gru = nn.BatchNorm1d(filts[3][-1])
        self.gru     = nn.GRU(filts[3][-1], d_args["gru_node"],
                              d_args["nb_gru_layer"], batch_first=True)
        self.fc1_gru = nn.Linear(d_args["gru_node"],   d_args["nb_fc_node"])
        self.fc2_gru = nn.Linear(d_args["nb_fc_node"],  d_args["nb_classes"])
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.selu(self.first_bn(self._sinc(x.unsqueeze(1))))
        for blk, attn in zip(
            [self.block0, self.block1, self.block2,
             self.block3, self.block4, self.block5],
            [self.fc_attention0, self.fc_attention1, self.fc_attention2,
             self.fc_attention3, self.fc_attention4, self.fc_attention5]):
            x = blk(x)
            w = torch.sigmoid(attn(x.mean(-1))).unsqueeze(-1)
            x = x * w
        x = self.selu(self.bn_before_gru(x))
        self.gru.flatten_parameters()
        x, _ = self.gru(x.permute(0, 2, 1))
        x = self.fc2_gru(self.fc1_gru(x[:, -1, :]))
        return x, self.logsoftmax(x)
'''
(AASIST_DIR / 'models' / 'RawNet2Spoof.py').write_text(_RAWNET2_CODE)
print('✓ RawNet2Spoof.py overwritten with correct original architecture')

sys.path.insert(0, str(AASIST_DIR))
from importlib import import_module
print('✓ AASIST repo ready')

# ── Codec Configs ─────────────────────────────────────────────────
CODEC_CONFIGS = {
    'mp3_32':  {'bitrate': '32k',  'ext': 'mp3', 'acodec': 'libmp3lame'},
    'mp3_64':  {'bitrate': '64k',  'ext': 'mp3', 'acodec': 'libmp3lame'},
    'mp3_128': {'bitrate': '128k', 'ext': 'mp3', 'acodec': 'libmp3lame'},
    'aac_64':  {'bitrate': '64k',  'ext': 'aac', 'acodec': 'aac'},
    'aac_96':  {'bitrate': '96k',  'ext': 'aac', 'acodec': 'aac'},
    'opus_32': {'bitrate': '32k',  'ext': 'ogg', 'acodec': 'libopus'},
    'opus_64': {'bitrate': '64k',  'ext': 'ogg', 'acodec': 'libopus'},
}
CODEC_SETS = {
    'mp3_only':  ['mp3_64', 'mp3_128'],
    'opus_only': ['opus_64'],
    'aac_only':  ['aac_64', 'aac_96'],
    'all':       ['mp3_64', 'mp3_128', 'aac_96', 'opus_64'],
    'mp3_opus':  ['mp3_64', 'mp3_128', 'opus_64'],
}
ACTIVE_CODECS = CODEC_SETS[CODEC_SET]
print(f'✓ Training codecs: {ACTIVE_CODECS}')

EVAL_CONDITIONS = {
    'clean':   None,
    'mp3_32':  'mp3_32',
    'mp3_64':  'mp3_64',
    'mp3_128': 'mp3_128',
    'aac_64':  'aac_64',
    'aac_96':  'aac_96',
    'opus_32': 'opus_32',
    'opus_64': 'opus_64',
}

def apply_codec(audio_np, sr, codec_name):
    import tempfile
    cfg = CODEC_CONFIGS[codec_name]
    with tempfile.TemporaryDirectory() as tmp:
        in_p  = os.path.join(tmp, 'in.wav')
        out_p = os.path.join(tmp, f'out.{cfg["ext"]}')
        dec_p = os.path.join(tmp, 'dec.wav')
        sf.write(in_p, audio_np, sr)
        r = subprocess.run(
            ['ffmpeg', '-y', '-loglevel', 'error', '-i', in_p,
             '-acodec', cfg['acodec'], '-b:a', cfg['bitrate'], out_p],
            capture_output=True)
        if r.returncode != 0:
            warnings.warn(f'Codec {codec_name} failed: {r.stderr.decode()[:80]}')
            return audio_np
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', out_p,
                        '-ar', str(sr), '-ac', '1', dec_p],
                       check=True, capture_output=True)
        out, _ = sf.read(dec_p, dtype='float32')
    n = len(audio_np)
    if len(out) > n: out = out[:n]
    elif len(out) < n: out = np.pad(out, (0, n - len(out)))
    return out

# ── PHASE 1: Pre-Augment Training Data ───────────────────────────
SR      = 16000
MAX_LEN = 64600

def parse_protocol(proto_path):
    samples = []
    with open(proto_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            samples.append((parts[1], 1 if parts[4] == 'spoof' else 0, parts[3]))
    return samples

train_samples = parse_protocol(PROTO_TRAIN)
print(f'Training files: {len(train_samples)}')

rng = random.Random(AUG_SEED)
codec_map_path = AUG_DIR / 'codec_assignments.json'
if codec_map_path.exists():
    with open(codec_map_path) as f: codec_map = json.load(f)
    print(f'✓ Loaded codec assignments ({len(codec_map)} files)')
else:
    codec_map = {fid: (rng.choice(ACTIVE_CODECS) if ACTIVE_CODECS else None)
                 for fid, _, _ in train_samples}
    with open(codec_map_path, 'w') as f: json.dump(codec_map, f)
    print(f'✓ Generated codec assignments for {len(codec_map)} files')

done = set(f.stem for f in AUG_DIR.glob('*.wav'))
todo = [(fid, lbl, sys_id) for fid, lbl, sys_id in train_samples if fid not in done]
print(f'Pre-augmentation: {len(done)}/{len(train_samples)} done, {len(todo)} remaining')

t0 = time.time()
for i, (file_id, label, system) in enumerate(todo):
    src = TRAIN_FLAC / f'{file_id}.flac'
    if not src.exists(): src = TRAIN_FLAC / f'{file_id}.wav'
    dst = AUG_DIR / f'{file_id}.wav'
    try:
        audio, _ = librosa.load(str(src), sr=SR, mono=True)
        codec_name = codec_map.get(file_id)
        if codec_name:
            audio = apply_codec(audio, SR, codec_name)
        if len(audio) > MAX_LEN: audio = audio[:MAX_LEN]
        elif len(audio) < MAX_LEN: audio = np.pad(audio, (0, MAX_LEN - len(audio)))
        sf.write(str(dst), audio, SR)
    except Exception as e:
        warnings.warn(f'Failed {file_id}: {e}')
    if (i+1) % 1000 == 0:
        elapsed = time.time() - t0
        eta = (len(todo)-i-1) / ((i+1)/elapsed)
        print(f'  {len(done)+i+1}/{len(train_samples)} | {elapsed/60:.1f}min | ETA {eta/60:.1f}min')

print(f'✓ Pre-augmentation done: {len(list(AUG_DIR.glob("*.wav")))}/{len(train_samples)}')

# ── PHASE 2: Training ─────────────────────────────────────────────
# FIX [3a]: Flush stale module cache before importing model
for _k in [_k for _k in sys.modules
           if 'RawNet2' in _k or (_k.startswith('models') and 'torch' not in _k)]:
    del sys.modules[_k]
print('✓ Module cache cleared for Phase 2')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(AASIST_DIR / 'config/AASIST.conf') as f:
    aasist_full_cfg = json.load(f)
MODEL_CONFIGS = {
    'AASIST': aasist_full_cfg['model_config'],
    'RawNet2': {
        'architecture': 'RawNet2Spoof', 'nb_samp': 64600,
        'first_conv': 1024, 'filts': [20, [20, 20], [20, 128], [128, 128]],
        'in_channels': 1, 'blocks': [2, 4], 'nb_fc_node': 1024,
        'gru_node': 1024, 'nb_gru_layer': 3, 'nb_classes': 2,
    },
}
MODEL_ARCHS = {'AASIST': 'AASIST', 'RawNet2': 'RawNet2Spoof'}

arch   = MODEL_ARCHS[MODEL]
m_cfg  = MODEL_CONFIGS[MODEL]
module = import_module(f'models.{arch}')
model  = getattr(module, 'Model')(m_cfg).to(device)
print(f'✓ {MODEL} loaded ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params)')

class AugDataset(Dataset):
    def __init__(self, samples, aug_dir):
        self.samples = [(fid, lbl) for fid, lbl, _ in samples
                        if (aug_dir / f'{fid}.wav').exists()]
        self.aug_dir = aug_dir
        print(f'  AugDataset: {len(self.samples)} files')
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        fid, lbl = self.samples[idx]
        wav, _ = sf.read(str(self.aug_dir / f'{fid}.wav'), dtype='float32')
        if len(wav) > MAX_LEN: wav = wav[:MAX_LEN]
        elif len(wav) < MAX_LEN: wav = np.pad(wav, (0, MAX_LEN - len(wav)))
        return torch.FloatTensor(wav), lbl

class CleanDataset(Dataset):
    def __init__(self, samples, audio_dir, sr=16000):
        self.samples = samples; self.audio_dir = Path(audio_dir); self.sr = sr
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        fid, lbl, sys_id = self.samples[idx]
        p = self.audio_dir / f'{fid}.flac'
        if not p.exists(): p = self.audio_dir / f'{fid}.wav'
        wav, _ = librosa.load(str(p), sr=self.sr, mono=True)
        if len(wav) > MAX_LEN: wav = wav[:MAX_LEN]
        elif len(wav) < MAX_LEN: wav = np.pad(wav, (0, MAX_LEN - len(wav)))
        return torch.FloatTensor(wav), lbl, sys_id

def compute_eer(bona, spoof):
    b, s = np.sort(np.array(bona, np.float32)), np.sort(np.array(spoof, np.float32))
    thr  = np.unique(np.concatenate([b, s]))
    frr  = np.searchsorted(b, thr, side='left')  / len(b)
    far  = 1.0 - np.searchsorted(s, thr, side='left') / len(s)
    idx  = np.argmin(np.abs(far - frr))
    return float((far[idx] + frr[idx]) / 2 * 100)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=5e-6)
criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]).to(device))

start_epoch, best_eer, best_epoch = 0, 999.0, 0
if CKPT_PATH.exists():
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    start_epoch = ckpt['epoch']
    best_eer    = ckpt.get('best_eer', 999.0)
    best_epoch  = ckpt.get('best_epoch', 0)
    print(f'► Resumed from epoch {start_epoch}, best EER: {best_eer:.2f}%')
else:
    print('Starting from scratch')

dev_samples   = parse_protocol(PROTO_DEV)
train_dataset = AugDataset(train_samples, AUG_DIR)
dev_dataset   = CleanDataset(dev_samples, DEV_FLAC)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, drop_last=True)
dev_loader    = DataLoader(dev_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f'\n{"="*55}')
print(f'  TRAINING — {RUN_NAME}')
print(f'  Epochs: {start_epoch+1}→{NUM_EPOCHS}  |  Codec: {CODEC_SET}')
print(f'  Train: {len(train_dataset)}  |  Dev: {len(dev_dataset)}')
print(f'{"="*55}\n')

log_path = MODEL_DIR / 'training.log'
with open(log_path, 'a') as log:
    log.write(f'Run: {RUN_NAME} | resumed from epoch {start_epoch}\n')
    for epoch in range(start_epoch + 1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch_audio, batch_labels in train_loader:
            batch_audio  = batch_audio.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            _, output = model(batch_audio)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= len(train_loader)

        model.eval()
        bona_dev, spoof_dev = [], []
        with torch.no_grad():
            for batch_audio, batch_labels, _ in dev_loader:
                _, out = model(batch_audio.to(device))
                scores = out[:, 0].cpu().numpy()
                for sc, lbl in zip(scores, batch_labels.numpy()):
                    (bona_dev if lbl == 0 else spoof_dev).append(sc)
        dev_eer = compute_eer(bona_dev, spoof_dev)

        is_best = dev_eer < best_eer
        if is_best:
            best_eer, best_epoch = dev_eer, epoch
            torch.save(model.state_dict(), MODEL_DIR / 'best.pth')

        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_eer': best_eer, 'best_epoch': best_epoch}, CKPT_PATH)

        line = (f'Epoch {epoch:3d}/{NUM_EPOCHS} | loss {train_loss:.4f} | '
                f'dev_EER {dev_eer:.2f}%' + (' *** BEST ***' if is_best else ''))
        print(line); log.write(line + '\n'); log.flush()

print(f'\n✓ Training done. Best dev EER: {best_eer:.2f}% at epoch {best_epoch}')

# ── PHASE 3: ASVspoof 2019 Eval (uses last_checkpoint.pth = Epoch 30) ──
# FIX [3b]: Flush stale module cache before importing model for eval
for _k in [_k for _k in sys.modules
           if 'RawNet2' in _k or (_k.startswith('models') and 'torch' not in _k)]:
    del sys.modules[_k]
print('✓ Module cache cleared for Phase 3')

# FIX [4]: Always load last_checkpoint.pth (Epoch 30), not best.pth
best_model = getattr(import_module(f'models.{arch}'), 'Model')(m_cfg).to(device)
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
best_model.load_state_dict(ckpt['model'])
best_model.eval()

trained_epoch = ckpt.get('epoch', 30)
print(f'✓ Epoch {trained_epoch} model loaded for evaluation')

results = {'run': RUN_NAME, 'model': MODEL, 'codec_set': CODEC_SET,
           'best_dev_eer': best_eer, 'best_epoch': trained_epoch,
           'asvspooof2019_eval': {}, 'wavefake': {}}

eval_samples = parse_protocol(PROTO_EVAL)
eval_dataset = CleanDataset(eval_samples, EVAL_FLAC)
eval_loader  = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=0)

bona_eval, spoof_eval, per_sys = [], [], {}
with torch.no_grad():
    for batch_audio, batch_labels, batch_systems in eval_loader:
        _, out = best_model(batch_audio.to(device))
        for sc, lbl, sys_id in zip(out[:,0].cpu().numpy(), batch_labels.numpy(), batch_systems):
            (bona_eval if lbl == 0 else spoof_eval).append(float(sc))
            if lbl == 1:
                per_sys.setdefault(sys_id, []).append(float(sc))

overall_eer = compute_eer(bona_eval, spoof_eval)
per_sys_eer = {s: compute_eer(bona_eval, v) for s, v in per_sys.items()}
results['asvspooof2019_eval']['clean'] = {'overall_eer': overall_eer, 'per_system': per_sys_eer}

print(f'\nASVspoof 2019 eval (clean): Overall EER = {overall_eer:.2f}%')
for s, e in sorted(per_sys_eer.items()): print(f'  {s}: {e:.2f}%')

out_path = Path(f'/kaggle/working/{RUN_NAME}_results.json')
with open(out_path, 'w') as f: json.dump(results, f, indent=2)

# ── PHASE 4: WaveFake Eval ────────────────────────────────────────
from concurrent.futures import ThreadPoolExecutor

vocoder_dirs = {}
for vdir in sorted(WAVEFAKE_INPUT.rglob('*')):
    if vdir.is_dir() and 'ljspeech_' in vdir.name:
        wavs = sorted(vdir.glob('*.wav'))
        if wavs: vocoder_dirs[vdir.name] = wavs
assert vocoder_dirs, '❌ No ljspeech_ vocoder dirs found'
print(f'WaveFake vocoders: {list(vocoder_dirs.keys())}')

LJ_DIR = Path('/kaggle/working/ljspeech/wavs')
if not LJ_DIR.exists() or len(list(LJ_DIR.glob('*.wav'))) == 0:
    lj_tar = Path('/kaggle/working/LJSpeech-1.1.tar.bz2')
    if not lj_tar.exists():
        subprocess.run(['wget', '-q', '--show-progress',
                        'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2',
                        '-O', str(lj_tar)], check=True)
    subprocess.run(['tar', '-xjf', str(lj_tar), '-C', '/kaggle/working/'], check=True)
    p = Path('/kaggle/working/LJSpeech-1.1')
    if p.exists(): p.rename('/kaggle/working/ljspeech')
    lj_tar.unlink()

bona_wavs = sorted(LJ_DIR.glob('*.wav'))
np.random.seed(42)
bona_wavs = np.random.choice(bona_wavs, size=2000, replace=False).tolist()
print(f'✓ LJSpeech bonafide: {len(bona_wavs)} files')

def load_and_encode(p, codec_name):
    try:
        wav, orig_sr = sf.read(str(p), dtype='float32')
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if orig_sr != SR: wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=SR)
        if codec_name: wav = apply_codec(wav, SR, codec_name)
        if len(wav) > MAX_LEN: wav = wav[:MAX_LEN]
        elif len(wav) < MAX_LEN: wav = np.pad(wav, (0, MAX_LEN - len(wav)))
        return torch.FloatTensor(wav)
    except:
        return None

def score_files(model, files, codec_name=None, batch_size=32):
    scores = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i in range(0, len(files), batch_size):
            batch_paths = files[i:i+batch_size]
            batch_tensors = list(executor.map(lambda p: load_and_encode(p, codec_name), batch_paths))
            batch_tensors = [t for t in batch_tensors if t is not None]
            if not batch_tensors: continue
            with torch.no_grad():
                _, out = model(torch.stack(batch_tensors).to(device))
                scores.extend(out[:, 0].cpu().numpy().tolist())
    return scores

print('\n── WaveFake evaluation ──')
for cond_name, codec_name in EVAL_CONDITIONS.items():
    print(f'\n  Condition: {cond_name}')
    results['wavefake'][cond_name] = {}

    t0 = time.time()
    print(f'    Scoring bonafide...', end='', flush=True)
    bona_s = score_files(best_model, bona_wavs, codec_name)
    print(f' done in {time.time()-t0:.1f}s')

    for voc_name, fake_files in sorted(vocoder_dirs.items()):
        print(f'    [{voc_name}]...', end='', flush=True)
        t_voc = time.time()
        spoof_s = score_files(best_model, fake_files, codec_name)
        eer = compute_eer(bona_s, spoof_s)
        results['wavefake'][cond_name][voc_name] = eer
        print(f' EER={eer:.2f}% ({time.time()-t_voc:.1f}s)')

        out_path = Path(f'/kaggle/working/{RUN_NAME}_results.json')
        with open(out_path, 'w') as f: json.dump(results, f, indent=2)

    mean = np.mean(list(results['wavefake'][cond_name].values()))
    print(f'    Mean EER [{cond_name}]: {mean:.2f}%')

print('\n✓ WaveFake evaluation complete')

# ── Final Save & Summary ──────────────────────────────────────────
out_path = Path(f'/kaggle/working/{RUN_NAME}_results.json')
with open(out_path, 'w') as f: json.dump(results, f, indent=2)
shutil.copy(out_path, RESULTS_DIR / out_path.name)

print(f'\n{"="*68}')
print(f'  RESULTS — {RUN_NAME}')
print(f'  Model: {MODEL}  |  Codec: {CODEC_SET}  |  Best dev EER: {best_eer:.2f}% (ep {best_epoch})')
print(f'  ASVspoof 2019 eval (clean): {results["asvspooof2019_eval"]["clean"]["overall_eer"]:.2f}%')
print(f'{"="*68}')

conds = list(EVAL_CONDITIONS.keys())
vocs  = sorted(vocoder_dirs.keys())
hdr   = f'  {"Vocoder":<32}' + ''.join(f'{c:>10}' for c in conds)
print('\n  WaveFake EER (%) per Vocoder x Codec Condition:')
print(hdr)
print('  ' + '-'*(len(hdr)-2))
for voc in vocs:
    row = f'  {voc:<32}'
    for cond in conds:
        val = results['wavefake'].get(cond, {}).get(voc)
        row += f'{val:>9.2f}%' if val is not None else f'{"N/A":>10}'
    print(row)
avgs = [np.mean([results['wavefake'].get(c, {}).get(v, 0) for v in vocs]) for c in conds]
print(f'  {"Mean":<32}' + ''.join(f'{a:>9.2f}%' for a in avgs))

print(f'\n✓ Saved → /kaggle/working/{RUN_NAME}_results.json')
print('► Share this JSON with team for results table')
