"""Score saved v8 fold models on the exam named by LOCALVAL."""
import pickle, sys
from pathlib import Path
import lightgbm as lgb, numpy as np
sys.path.insert(0, "src")
import lbcheck, score_local, v8
m = pickle.loads(lbcheck.CACHE.read_bytes())
v8.RICH = True
z = np.load(score_local.VAL)
F = v8.features(z["X"], m["D16"], m["D32"])
for f in sys.argv[1:]:
    p = np.clip(lgb.Booster(model_file=f"models/{f}").predict(F), 0, 1)
    print(f"{score_local.VAL.name} {f} {score_local.local_scores({'x': p})['x']:.4f}", flush=True)
