"""Stream VTUAD v1.3.2 strict vessel clips from Zenodo and keep only 2-second band powers.

Each clip -> data/vtuad/bands/<n>.npy, float32 (segments, 25): mean-square power per
2 s segment in the 25 test bands, uncalibrated digital units. Raw audio never hits disk.
"""
import hashlib
import io
import struct
import sys
import time
import urllib.request
import wave
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.signal import periodogram

URL = "https://zenodo.org/records/23037895/files/{}?download=1"
ARCHIVES = {  # name -> list of (file, size) byte-split parts, concatenated in order
    "vessel": [(f"vtuad_strict_wav_vessel.zip.part0{i}", 1_237_445_900 - (i == 3)) for i in (1, 2, 3)],
    "background": [("vtuad_strict_wav_background.zip", 926_037_631)],
}
PARTS = ARCHIVES["vessel"]
TOTAL = sum(s for _, s in PARTS)
DATA = Path(__file__).resolve().parents[1] / "data" / "vtuad"
OUT = DATA / "bands"
CENTRES = 10.0 ** (1.5 + 0.1 * np.arange(25))


def fetch(start, end):
    """Bytes [start, end) of the virtual concatenated archive, across part boundaries."""
    chunks = []
    while start < end:
        base = 0
        for name, size in PARTS:
            if start < base + size:
                break
            base += size
        off, stop = start - base, min(end, base + size)
        req = urllib.request.Request(URL.format(name), headers={"Range": f"bytes={off}-{off + stop - start - 1}"})
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) == stop - start:
                    break
            except Exception:
                pass
            time.sleep(2 ** attempt)
        else:
            raise IOError(f"range {start}-{stop} failed")
        chunks.append(data)
        start = stop
    return b"".join(chunks)


class RemoteFile(io.RawIOBase):
    """Seekable read-only view used only to parse the zip central directory."""

    def __init__(self):
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: TOTAL + off}[whence]
        return self.pos

    def readinto(self, b):
        n = min(len(b), TOTAL - self.pos)
        if n <= 0:
            return 0
        b[:n] = fetch(self.pos, self.pos + n)
        self.pos += n
        return n


def band_power(samples, fs):
    """1-D waveform -> (2 s segments, 25) band mean-square, as in the starter notebook."""
    seg = int(2 * fs)
    n = samples.size // seg
    out = np.full((n, 25), np.nan, dtype=np.float32)
    for i in range(0, n, 64):
        block = samples[i * seg:min(n, i + 64) * seg].astype(np.float64).reshape(-1, seg)
        f, psd = periodogram(block, fs=fs, window="hann", detrend="constant", return_onesided=True, scaling="density", axis=1)
        df = f[1] - f[0]
        for b, c in enumerate(CENTRES):
            lo, hi = c / 10 ** 0.05, c * 10 ** 0.05
            sel = (f >= lo) & (f < hi)
            if hi <= fs / 2 and sel.any():
                out[i:i + len(block), b] = psd[:, sel].sum(axis=1) * df
    return out


def process(info, sha):
    dest = OUT / (Path(info.filename).stem + ".npy")
    if dest.exists():
        return info.filename, "skip"
    raw = fetch(info.header_offset, min(TOTAL, info.header_offset + 30 + 65536 + info.compress_size))
    name_len, extra_len = struct.unpack("<HH", raw[26:30])
    start = 30 + name_len + extra_len
    comp = raw[start:start + info.compress_size]
    data = zlib.decompress(comp, -15) if info.compress_type == zipfile.ZIP_DEFLATED else comp
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError(f"sha mismatch {info.filename}")
    with wave.open(io.BytesIO(data)) as w:
        fs, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{info.filename}: {width}-byte samples")
    x = np.frombuffer(frames, dtype="<i2").reshape(-1, ch)[:, 0]
    np.save(dest, band_power(x, fs))
    return info.filename, f"{x.size / fs:.0f}s"


def main(archive="vessel", workers=8):
    global PARTS, TOTAL, OUT
    PARTS = ARCHIVES[archive]
    TOTAL = sum(s for _, s in PARTS)
    OUT = DATA / ("bands" if archive == "vessel" else f"bands_{archive}")
    OUT.mkdir(parents=True, exist_ok=True)
    shas = {}
    for line in (DATA / "strict_interval_wav_SHA256SUMS.txt").read_text().splitlines():
        h, p = line.split(maxsplit=1)
        shas[p.strip()] = h
    zf = zipfile.ZipFile(io.BufferedReader(RemoteFile(), buffer_size=1 << 20))
    infos = [i for i in zf.infolist() if i.filename.endswith(".wav")]
    print(f"{len(infos)} clips", flush=True)
    done = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(process, i, shas[i.filename]) for i in infos]
        for fut in as_completed(futs):
            name, msg = fut.result()
            done += 1
            print(f"[{done}/{len(infos)}] {name} {msg}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "vessel", int(sys.argv[2]) if len(sys.argv) > 2 else 8)
