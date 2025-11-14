#!/usr/bin/env python
# coding: utf-8

# **Title**:

# **Data Pre-Processing**: *Shyamkumar Moradiya*

# In[ ]:


# imports
from pathlib import Path
import os, random, hashlib, gc
import numpy as np
import pandas as pd
from PIL import Image

# Reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# dataset path
DATA_ROOT = Path(r"C:\Users\shyam\Desktop\SSM\CSC594\Project\datasets")
TRAIN_DIR = DATA_ROOT / "Training"
TEST_DIR  = DATA_ROOT / "Testing"

# Artifacts
ARTIFACTS_DIR = DATA_ROOT / "_artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

print("Train:", TRAIN_DIR.resolve())
print("Test :", TEST_DIR.resolve())
print("Artifacts ->", ARTIFACTS_DIR.resolve())


# In[2]:


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Helpers
def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def md5_of_file(p: Path, chunk=8192):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def scan_split(split_dir: Path, split_name: str) -> pd.DataFrame:
    rows = []
    for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        label = cls_dir.name
        for p in cls_dir.rglob("*"):
            if p.is_file() and is_image_file(p):
                # Read image metadata (safe)
                try:
                    with Image.open(p) as im:
                        w, h = im.size; mode = im.mode
                except Exception:
                    w, h, mode = None, None, "CORRUPT"
                rows.append({
                    "split": split_name,
                    "path": str(p),
                    "label": label,
                    "width": w, "height": h, "mode": mode,
                    "md5": md5_of_file(p) if w is not None else None
                })
    return pd.DataFrame(rows)


# In[ ]:


# Scan dataset → DataFrames & save raw metadata
train_df = scan_split(TRAIN_DIR, "train")
test_df  = scan_split(TEST_DIR,  "test")
meta = pd.concat([train_df, test_df], ignore_index=True)

# Save raw metadata
meta.to_csv(ARTIFACTS_DIR / "metadata_raw.csv", index=False)

print("Train images:", len(train_df), " | Test images:", len(test_df))
display(train_df.head())


# In[4]:


# Quick audit & duplicates
print("By split:\n", meta["split"].value_counts(), "\n")
print("By label (train):\n", train_df["label"].value_counts(), "\n")
print("Image modes (train):\n", train_df["mode"].value_counts(dropna=False), "\n")

dups = train_df.groupby("md5").size().sort_values(ascending=False)
print("Potential duplicates (md5 -> count, top 10):")
print(dups[dups > 1].head(10))


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt

# Safer plotting for large data
matplotlib.rcParams['agg.path.chunksize'] = 100000
plt.close('all'); gc.collect()

# Filter out corrupts & invalid sizes for visualization
viz_train = train_df[(train_df['mode'] != 'CORRUPT') & (train_df['width']>0) & (train_df['height']>0)].copy()
viz_test  = test_df[(test_df['mode'] != 'CORRUPT') & (test_df['width']>0) & (test_df['height']>0)].copy()

def bar_chart_counts(df, title, outpath: Path):
    plt.close('all'); gc.collect()
    counts = df["label"].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(6,4))
    counts.plot(kind="bar")
    plt.title(title); plt.xlabel("Class"); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.show(); plt.close()

bar_chart_counts(viz_train, "Training class counts", ARTIFACTS_DIR/"class_counts_train.png")
bar_chart_counts(viz_test,  "Testing class counts",  ARTIFACTS_DIR/"class_counts_test.png")


# In[6]:


#Image size scatter
sizes = viz_train[['width','height']].dropna().astype('float32')
MAX_POINTS = 50000
if len(sizes) > MAX_POINTS:
    sizes = sizes.sample(n=MAX_POINTS, random_state=SEED)

try:
    plt.figure(figsize=(6,4))
    plt.scatter(sizes['width'].to_numpy(), sizes['height'].to_numpy(), s=6, alpha=0.5)
    plt.title("Training image sizes")
    plt.xlabel("Width"); plt.ylabel("Height")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR/"train_sizes_scatter.png", dpi=150)
    plt.show(); plt.close()
except RuntimeError:
    plt.figure(figsize=(6,4))
    plt.hexbin(sizes['width'].to_numpy(), sizes['height'].to_numpy(), gridsize=40)
    plt.title("Training image sizes (hexbin)")
    plt.xlabel("Width"); plt.ylabel("Height")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR/"train_sizes_hexbin.png", dpi=150)
    plt.show(); plt.close()


# In[7]:


from sklearn.model_selection import train_test_split

# Keep only clean rows for splitting
clean_train = train_df[(train_df["mode"] != "CORRUPT")].copy()

# Stratified by label
idx_all = np.arange(len(clean_train))
tr_idx, va_idx = train_test_split(
    idx_all, test_size=0.2, random_state=SEED, stratify=clean_train["label"]
)

clean_train.loc[tr_idx, "subset"] = "train"
clean_train.loc[va_idx, "subset"] = "val"

# Merge subset back
meta = meta.merge(clean_train[["path","subset"]], on="path", how="left")

# Save bootstraps
meta.to_csv(ARTIFACTS_DIR / "metadata_with_split.csv", index=False)

print(meta["subset"].value_counts(dropna=False))
display(meta.head())


# In[8]:


# Compute mean/std on training subset
def load_image_as_rgb(path, target_size=(224,224)):
    im = Image.open(path).convert("RGB")
    im = im.resize(target_size, Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0

subset_paths = meta[(meta["subset"]=="train") & (meta["mode"]!="CORRUPT")]["path"].tolist()
sample_paths = subset_paths if len(subset_paths) <= 2000 else random.sample(subset_paths, 2000)

acc = []
for p in sample_paths:
    try:
        rgb = load_image_as_rgb(p)
        acc.append(rgb.reshape(-1,3))
    except Exception:
        pass

if len(acc) == 0:
    raise RuntimeError("No valid images found to compute mean/std. Check training subset.")

acc = np.concatenate(acc, axis=0)
mean = acc.mean(axis=0)   # R,G,B
std  = acc.std(axis=0) + 1e-6

print("Train mean:", mean)
print("Train std :", std)

np.save(ARTIFACTS_DIR/"train_mean.npy", mean)
np.save(ARTIFACTS_DIR/"train_std.npy", std)


# In[ ]:


# PyTorch Dataset & DataLoaders
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, utils as vutils
from collections import Counter

TARGET_SIZE   = 224
BATCH_SIZE    = 32
NUM_WORKERS   = 0    
use_pin       = torch.cuda.is_available()

class ImageClsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, subset: str, target=224, mean=None, std=None, augment=False):
        if "subset" in df.columns:
            self.df = df[df["subset"]==subset].reset_index(drop=True)
        else:
            self.df = df[df["split"]==subset].reset_index(drop=True)
        # keep only clean rows
        self.df = self.df[self.df["mode"]!="CORRUPT"].reset_index(drop=True)

        self.labels = sorted(self.df["label"].unique().tolist())
        self.cls2idx = {c:i for i,c in enumerate(self.labels)}
        self.idx2cls = {i:c for c,i in self.cls2idx.items()}

        self.mean = mean if mean is not None else np.array([0.485,0.456,0.406])
        self.std  = std  if std  is not None else np.array([0.229,0.224,0.225])

        tfs = []
        if augment:
            tfs += [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(0.1,0.1)
            ]
        tfs += [
            transforms.Resize((target,target)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean.tolist(), self.std.tolist())
        ]
        self.tf = transforms.Compose(tfs)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        im = Image.open(row["path"]).convert("RGB")
        x = self.tf(im)
        y = self.cls2idx[row["label"]]
        return x, y

# Load normalization stats
mean_npy, std_npy = ARTIFACTS_DIR/"train_mean.npy", ARTIFACTS_DIR/"train_std.npy"
mean = np.load(mean_npy) if mean_npy.exists() else None
std  = np.load(std_npy)  if std_npy.exists()  else None

train_ds = ImageClsDataset(meta, "train", target=TARGET_SIZE, mean=mean, std=std, augment=True)
val_ds   = ImageClsDataset(meta, "val",   target=TARGET_SIZE, mean=mean, std=std, augment=False)

# Build test view
test_df_ = meta[meta["split"]=="test"].copy(); test_df_["subset"]="test"
test_ds  = ImageClsDataset(test_df_, "test", target=TARGET_SIZE, mean=mean, std=std, augment=False)

# Class weights & sampler
labels = [y for _,y in (train_ds[i] for i in range(len(train_ds)))]
cnt = Counter(labels); total = sum(cnt.values())
class_weight = {c: total/(len(cnt)*n) for c,n in cnt.items()}
weights = torch.DoubleTensor([class_weight[y] for y in labels])
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# DataLoaders
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=use_pin, persistent_workers=False)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=use_pin, persistent_workers=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=use_pin, persistent_workers=False)

print("Sizes ->", len(train_ds), len(val_ds), len(test_ds))
print("Classes:", train_ds.cls2idx)


# In[10]:


import matplotlib.pyplot as plt

# Fast montage without DataLoader
subset_paths_for_preview = meta[(meta["subset"]=="train") & (meta["mode"]!="CORRUPT")]["path"].tolist()
sample_paths = random.sample(subset_paths_for_preview, min(16, len(subset_paths_for_preview)))

tf_vis = transforms.Compose([transforms.Resize((TARGET_SIZE,TARGET_SIZE)), transforms.ToTensor()])

imgs = []
for p in sample_paths:
    try:
        im = Image.open(p).convert("RGB")
        imgs.append(tf_vis(im))
    except Exception:
        pass

if len(imgs) == 0:
    raise RuntimeError("No images available for preview montage.")

grid = vutils.make_grid(torch.stack(imgs), nrow=8, normalize=True)
plt.figure(figsize=(10,4))
plt.imshow(grid.permute(1,2,0).cpu().numpy())
plt.axis("off"); plt.title("train_augment_preview (fast)")
plt.tight_layout()
plt.savefig(ARTIFACTS_DIR/"montage_train_fast.png", dpi=150)
plt.show(); plt.close()

# Histograms
subset_paths_for_hist = subset_paths_for_preview[:200]  # limit pool for speed
plt.figure(figsize=(6,4))
for p in random.sample(subset_paths_for_hist, min(6, len(subset_paths_for_hist))):
    arr = np.asarray(Image.open(p).convert("L"), dtype=np.float32)/255.0
    plt.hist(arr.flatten(), bins=50, histtype="step", alpha=0.6)
plt.title("Grayscale intensity histograms (sampled)")
plt.xlabel("Intensity [0..1]"); plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(ARTIFACTS_DIR/"intensity_histograms.png", dpi=150)
plt.show(); plt.close()


# In[11]:


# cache resized 224×224 images
CACHE_DIR = DATA_ROOT / "_cache_224"
CACHE_DIR.mkdir(exist_ok=True)

def cache_split(df, subset, size=(224,224)):
    outdir = CACHE_DIR / subset
    for cls in sorted(df["label"].dropna().unique()):
        (outdir/cls).mkdir(parents=True, exist_ok=True)

    rows = df[(df["subset"]==subset) if "subset" in df.columns else (df["split"]==subset)]
    for _, r in rows.iterrows():
        try:
            im = Image.open(r["path"]).convert("RGB").resize(size, Image.BILINEAR)
            out = outdir / r["label"] / Path(r["path"]).name
            im.save(out)
        except Exception:
            pass


# In[ ]:




