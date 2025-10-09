# src/data.py
import gzip, os, urllib.request, urllib.error, time
import numpy as np

# Mirrors HTTPS (ordenados por fiabilidad)
MIRRORS = [
    # Mirror en GitHub (raw) – rápido y estable
    "https://raw.githubusercontent.com/fgnt/mnist/master/{fname}",  # :contentReference[oaicite:0]{index=0}
    # Mirror de CVDF (Google Cloud Storage)
    "https://storage.googleapis.com/cvdf-datasets/mnist/{fname}",   # :contentReference[oaicite:1]{index=1}
]

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")

def _download(path, fname):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    last_err = None
    for base in MIRRORS:
        url = base.format(fname=fname)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r, open(path, "wb") as f:
                f.write(r.read())
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last_err = e
            time.sleep(0.5)
    raise RuntimeError(f"No se pudo descargar {fname} desde los mirrors. Último error: {last_err}")

def _read_images(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic, n, rows, cols = np.frombuffer(data[:16], dtype=">i4")
    assert magic == 2051, "Cabecera IDX no válida para imágenes"
    arr = np.frombuffer(data[16:], dtype=np.uint8).reshape(n, 1, rows, cols)
    return arr.astype(np.float32) / 255.0

def _read_labels(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic, n = np.frombuffer(data[:8], dtype=">i4")
    assert magic == 2049, "Cabecera IDX no válida para labels"
    return np.frombuffer(data[8:], dtype=np.uint8)

def load_mnist(data_dir="data", valid_split=5000, seed=123):
    paths = {k: os.path.join(data_dir, FILES[k]) for k in FILES}
    for k, fname in FILES.items():
        _download(paths[k], fname)

    x_train = _read_images(paths["train_images"])
    y_train = _read_labels(paths["train_labels"])
    x_test  = _read_images(paths["test_images"])
    y_test  = _read_labels(paths["test_labels"])

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x_train))
    val_idx, tr_idx = idx[:valid_split], idx[valid_split:]
    return (x_train[tr_idx], y_train[tr_idx]), (x_train[val_idx], y_train[val_idx]), (x_test, y_test)

def batches(x, y, batch_size=64, shuffle=True, seed=123):
    N = len(x)
    idx = np.arange(N)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for i in range(0, N, batch_size):
        j = idx[i:i+batch_size]
        yield x[j], y[j]
