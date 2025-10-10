# make_confusions_extras.py
import os
import numpy as np
import matplotlib.pyplot as plt

from src.data import load_mnist
from src.model_fp32 import LeNetLite, softmax
from src.calibrator import (
    calibrate_weights_greedy,
    init_activation_qparams,
    build_qsim_blocks,
    predict_qsim,
)

def confusion(y_true, y_pred, K=10):
    M = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        M[int(t), int(p)] += 1
    return M

def row_normalize(M):
    M = M.astype(np.float64)
    return M / (M.sum(axis=1, keepdims=True) + 1e-12)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_heatmap(M, title, path, vmin=None, vmax=None, cmap="viridis"):
    ensure_dir(os.path.dirname(path))
    plt.figure(figsize=(5.4, 4.6))
    im = plt.imshow(M, interpolation="nearest", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    plt.title(title)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(M.shape[0])
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_bars_per_class(acc_fp32, acc_int8, path):
    ensure_dir(os.path.dirname(path))
    K = len(acc_fp32)
    x = np.arange(K)
    w = 0.38
    plt.figure(figsize=(6.8, 3.4))
    plt.bar(x - w/2, acc_fp32*100.0, width=w, label="FP32")
    plt.bar(x + w/2, acc_int8*100.0, width=w, label="INT8")
    plt.xticks(x, [str(i) for i in range(K)])
    plt.ylabel("Accuracy por clase (%)")
    plt.xlabel("Clase real")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def main():
    # 1) Datos y modelo FP32
    (x_tr, y_tr), (x_val, y_val), (x_te, y_te) = load_mnist()
    m = LeNetLite()
    sd = np.load("results/baseline_fp32.npz")
    m.load_state_dict({k: sd[k] for k in sd.files})

    # 2) Predicciones FP32
    y_fp32 = m.predict(x_te)
    acc_fp32_global = (y_fp32 == y_te).mean() * 100.0
    M_fp32 = confusion(y_te, y_fp32)
    R_fp32 = row_normalize(M_fp32)  # proporciones por clase

    # 3) INT8 (PTQ-init rápido: greedy pesos + p=99)
    Xcal = x_val[:512]
    wq = calibrate_weights_greedy(m, Xcal)
    aq = init_activation_qparams(m, Xcal, p=99.0)
    qs = build_qsim_blocks(m, wq, aq)
    y_int8 = predict_qsim(m, qs, x_te)
    acc_int8_global = (y_int8 == y_te).mean() * 100.0
    M_int8 = confusion(y_te, y_int8)
    R_int8 = row_normalize(M_int8)

    # 4) Heatmaps básicos (norm por clase)
    plot_heatmap(R_fp32, f"FP32 - acc={acc_fp32_global:.2f}%", "results/figuras/confusion_fp32.png")
    plot_heatmap(R_int8, f"INT8-QSim (init) - acc={acc_int8_global:.2f}%", "results/figuras/confusion_int8.png")

    # 5) Heatmap de diferencias (INT8 - FP32) con colormap divergente
    D = R_int8 - R_fp32
    m = np.max(np.abs(D))
    plot_heatmap(D, "Δ Confusión (INT8 − FP32, normalizado por fila)",
                 "results/figuras/confusion_delta.png",
                 vmin=-m, vmax=m, cmap="bwr")

    # 6) Accuracy por clase y barras comparativas
    acc_cls_fp32 = np.diag(R_fp32)
    acc_cls_int8 = np.diag(R_int8)
    plot_bars_per_class(acc_cls_fp32, acc_cls_int8, "results/figuras/acc_por_clase.png")

    # 7) Top confusiones que más cambian (tabla CSV)
    # Quitamos la diagonal, quedándonos con cambios en errores
    K = D.shape[0]
    mask = np.ones_like(D, dtype=bool)
    np.fill_diagonal(mask, False)
    changes = []
    for i in range(K):
        for j in range(K):
            if mask[i, j]:
                changes.append((abs(D[i, j]), D[i, j], i, j))
    changes.sort(reverse=True)  # por magnitud
    top = changes[:12]

    ensure_dir("results/tablas")
    with open("results/tablas/top_confusiones.csv", "w", encoding="utf-8") as f:
        f.write("rank,real,pred,delta_norm\n")
        for r, (mag, delta, i, j) in enumerate(top, 1):
            f.write(f"{r},{i},{j},{delta:.5f}\n")

    print("Listo:")
    print(" - results/figuras/confusion_fp32.png")
    print(" - results/figuras/confusion_int8.png")
    print(" - results/figuras/confusion_delta.png")
    print(" - results/figuras/acc_por_clase.png")
    print(" - results/tablas/top_confusiones.csv")
    print(f"FP32 acc={acc_fp32_global:.2f}% | INT8 acc={acc_int8_global:.2f}%")

if __name__ == "__main__":
    main()
