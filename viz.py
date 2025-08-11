# viz.py
import os, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import SimpleFlowNet
from train import CSVFlowDataset  # reuse the CSV loader

def save_quiver(u, v, title, path):
    H, W = u.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    plt.figure()
    plt.quiver(X, Y, u, -v, scale=40)
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def save_stream(u, v, title, path):
    H, W = u.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    speed = np.sqrt(u*u + v*v) + 1e-6
    uu, vv = u/speed, v/speed
    plt.figure()
    plt.streamplot(X, Y, uu, -vv, density=1.2, linewidth=1.0, arrowsize=1.0)
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data_csv")
    ap.add_argument("--outdir", type=str, default="runs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_ds = CSVFlowDataset(os.path.join(args.data_root, "test"))
    xb, yb = test_ds[0]  # first test sample
    gt = yb.numpy()

    model = SimpleFlowNet().to(device)
    model.load_state_dict(torch.load(os.path.join(args.outdir, "best_simple_flownet.pth"), map_location=device))
    model.eval()

    with torch.no_grad():
        pred = model(xb.unsqueeze(0).float().to(device)).squeeze(0).cpu().numpy()

    save_quiver(gt[0], gt[1], "Ground Truth Flow (Quiver)", os.path.join(args.outdir, "gt_quiver.png"))
    save_quiver(pred[0], pred[1], "Predicted Flow (Quiver)", os.path.join(args.outdir, "pred_quiver.png"))
    save_stream(gt[0], gt[1], "Ground Truth Flow (Streamplot)", os.path.join(args.outdir, "gt_stream.png"))
    save_stream(pred[0], pred[1], "Predicted Flow (Streamplot)", os.path.join(args.outdir, "pred_stream.png"))

    print("Saved images to", args.outdir)
