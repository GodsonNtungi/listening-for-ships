"""Write viz/listen.html: example windows split into sea, natural ship and added ship, playable in the browser.

  python3 src/listen.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import floor
import mix
import nmf

CASES = [  # (title, note, added share, natural ship level in dB vs sea or None, background pick)
    ("Just the sea", "No ship at all. The answer is 0.", 0.0, None, "quiet"),
    ("A loud ship passes", "The added ship carries 60% of the energy.", 0.6, None, "quiet"),
    ("A faint ship", "Only 5% of the energy is ship. Hard to hear, hard to score.", 0.05, None, "quiet"),
    ("The trap: a ship that doesn't count", "A ship was already in the recording. It is part of the sea, so the answer is 0.", 0.0, 0.0, "quiet"),
    ("Two ships, one counts", "A natural ship plus a faint added one. Only the added one counts: 10%.", 0.1, -3.0, "quiet"),
    ("Noisy sea, medium ship", "Loud high-pitched sea (snapping shrimp, wind) with a 30% ship.", 0.3, None, "busy"),
]


def main():
    r = np.random.default_rng(3)
    bg, _, _ = mix.load_backgrounds()
    clips, _, _ = mix.load_passages()
    sh = np.load(mix.ROOT / "models/nmf_v1_shapes.npz")
    full = np.where(np.isfinite(bg).all((1, 2)))[0]
    hi = 10 ** (bg[full][:, :, 18:].astype(np.float64) / 10).mean((1, 2)) / 10 ** (bg[full].astype(np.float64) / 10).mean((1, 2))
    picks = {"quiet": full[np.argsort(hi)[: len(hi) // 3]], "busy": full[np.argsort(hi)[-len(hi) // 10:]]}
    cfg = {**mix.CFG, "stretch": (1, 1), "tilt_db": 0, "bump_db": 0}
    out = []
    for title, note, share, nat_db, pick in CASES:
        S = 10 ** (bg[r.choice(picks[pick])].astype(np.float64) / 10)
        N = np.zeros_like(S)
        if nat_db is not None:
            N = mix.passage_minutes(clips[r.integers(len(clips))], r, cfg)
            N *= 10 ** (nat_db / 10) * S.sum() / N.sum()
        A = np.zeros_like(S)
        if share > 0:
            A = mix.passage_minutes(clips[r.integers(len(clips))], r, cfg)
            A *= share / (1 - share) * (S + N).sum() / A.sum()
        M = S + N + A
        X = 10 * np.log10(M)[None].astype(np.float32)
        guess = min(float(np.clip(nmf.split(X, sh["D_ship"], sh["D_sea"])[0][0], 0, 1)), float(floor.share(X, 3.0, 2)[0]))
        db = lambda P: np.round(10 * np.log10(P + 1e-12), 2).tolist()
        out.append(dict(title=title, note=note, truth=round(float(A.sum() / M.sum()), 3), guess=round(guess, 3),
                        sea=db(S), natural=db(N), added=db(A)))
    freqs = np.round(10.0 ** (1.5 + 0.1 * np.arange(25)), 1).tolist()
    tpl = (mix.ROOT / "viz/listen_template.html").read_text()
    (mix.ROOT / "viz/listen.html").write_text(tpl.replace("/*DATA*/null", json.dumps(dict(freqs=freqs, cases=out))))
    print("wrote viz/listen.html", [(c["title"], c["truth"], c["guess"]) for c in out])


if __name__ == "__main__":
    main()
