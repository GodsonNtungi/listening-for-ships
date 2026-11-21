"""Small conv net on the raw 15 x 25 grid, trained on the same broad recipe mixes as v8.

  python3 src/net.py local    # train on the lbcheck fold, score on the frozen local scorer
  python3 src/net.py submit   # train on everything, write submissions/net_v1.csv
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import mix
import train
import v8

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
N_TRAIN, EPOCHS, BATCH = int(os.environ.get("NET_N", 400_000)), int(os.environ.get("NET_EPOCHS", 30)), 512


def prep(X):
    """dB with NaN minutes -> (n, 2, 15, 25): level relative to the window's mean, and an observed mask."""
    obs = np.isfinite(X).all(2, keepdims=True)
    rel = X - np.nanmean(np.where(obs, X, np.nan), (1, 2), keepdims=True)
    return np.stack([np.where(obs, rel, 0.0) / 10, np.broadcast_to(obs, X.shape).astype(np.float32)], 1).astype(np.float32)


class Net(nn.Module):
    def __init__(self, w=64):
        super().__init__()
        block = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU())
        self.body = nn.Sequential(block(2, w), block(w, w), block(w, 2 * w), block(2 * w, 2 * w), block(2 * w, 2 * w))
        self.head = nn.Sequential(nn.Linear(4 * w, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, x):
        h = self.body(x)
        mask = x[:, 1:2]
        mean = (h * mask).sum((2, 3)) / mask.sum((2, 3)).clamp(min=1)
        mx = h.masked_fill(mask == 0, -1e4).amax((2, 3))
        return torch.sigmoid(self.head(torch.cat([mean, mx], 1))).squeeze(1)


def fit(X, y, seed=0):
    torch.manual_seed(seed)
    net = Net().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    steps = EPOCHS * (len(X) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    Xt, yt = torch.from_numpy(prep(X)), torch.from_numpy(y.astype(np.float32))
    r = np.random.default_rng(seed)
    for ep in range(EPOCHS):
        net.train()
        perm = r.permutation(len(X))
        tot = 0.0
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            b = perm[i:i + BATCH]
            xb = Xt[b]
            if r.random() < 0.5:  # a passage heard backwards is still a passage
                xb = xb.flip(2)
            loss = ((net(xb.to(DEV)) - yt[b].to(DEV)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        print(f"epoch {ep} train rmse {np.sqrt(tot / len(perm)):.4f}", flush=True)
    return net


@torch.no_grad()
def predict(net, X):
    net.eval()
    Xt = torch.from_numpy(prep(X))
    return np.concatenate([((net(Xt[i:i + 4096].to(DEV)) + net(Xt[i:i + 4096].flip(2).to(DEV))) / 2).cpu().numpy()
                           for i in range(0, len(Xt), 4096)])


if __name__ == "__main__":
    mode = sys.argv[1]
    r = np.random.default_rng(train.SEED + 9)
    if mode == "local":
        import lbcheck
        import score_local
        X, y = v8.make_train(lbcheck.bg, lbcheck.st, lbcheck.tr_bg, lbcheck.tr_c, N_TRAIN, r)
        print("mixes ready", X.shape, flush=True)
        net = fit(X, y)
        z = np.load(score_local.VAL)
        p = predict(net, z["X"])
        np.save(mix.ROOT / "models/net_local_pred.npy", p)
        loc = score_local.local_scores({"net": p})["net"]
        print(f"net local {loc:.4f} -> predicted public {score_local.predicted_public(loc):.4f}")
        torch.save(net.state_dict(), mix.ROOT / "models/net_fold.pt")
    else:
        import pandas as pd
        bg, st, _ = mix.load_backgrounds()
        clips, _, _ = mix.load_passages()
        X, y = v8.make_train(bg, st, np.arange(len(bg)), clips, N_TRAIN, r)
        net = fit(X, y)
        test = pd.read_csv(mix.ROOT / "data/Test.csv", dtype={"ID": str})
        p = np.clip(predict(net, test.drop(columns="ID").to_numpy(np.float32).reshape(-1, 15, 25)), 0, 1)
        pd.DataFrame({"ID": test.ID, "share_added_vessel": p, "score_added_vessel": p}).to_csv(mix.ROOT / "submissions/net_v1.csv", index=False)
        torch.save(net.state_dict(), mix.ROOT / "models/net_v1.pt")
        print(f"wrote net_v1.csv mean {p.mean():.3f}")
