import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import SimpleFlowNet

# dataset that reads csvs
class CSVFlowDataset(Dataset):
    def __init__(self, root_split_dir):
        self.sample_dirs = sorted(
            d for d in glob.glob(os.path.join(root_split_dir, "sample_*")) if os.path.isdir(d)
        )
        if not self.sample_dirs:
            raise RuntimeError(f"No samples found in {root_split_dir}")

    def __len__(self):
        return len(self.sample_dirs)

    def _read_csv(self, path, dtype=np.float32):
        return np.loadtxt(path, delimiter=",", dtype=dtype)

    def __getitem__(self, idx):
        d = self.sample_dirs[idx]
        mask  = self._read_csv(os.path.join(d, "mask.csv"))
        inlet = self._read_csv(os.path.join(d, "inlet.csv"))
        u     = self._read_csv(os.path.join(d, "u.csv"))
        v     = self._read_csv(os.path.join(d, "v.csv"))

        # stack to channel-first (2,H,W)
        x = np.stack([mask, inlet], axis=0).astype(np.float32)
        y = np.stack([u, v], axis=0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data_csv")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--outdir", type=str, default="runs")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    train_ds = CSVFlowDataset(os.path.join(args.data_root, "train"))
    val_ds   = CSVFlowDataset(os.path.join(args.data_root, "val"))
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=args.bs, shuffle=False)

    model = SimpleFlowNet(in_ch=2, mid=16, out_ch=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.MSELoss()

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        # --- training
        model.train()
        tr = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.float().to(device), yb.float().to(device)
            pred = model(xb)
            loss = crit(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr += loss.item()
        tr /= len(train_dl)

        # --- validation
        model.eval()
        va = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.float().to(device), yb.float().to(device)
                va += crit(model(xb), yb).item()
        va /= len(val_dl)
        print(f"[{ep}] train={tr:.4f}  val={va:.4f}")

        if va < best:
            best = va
            torch.save(model.state_dict(), os.path.join(args.outdir, "best_simple_flownet.pth"))

if __name__ == "__main__":
    main()
