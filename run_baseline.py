# run_baseline.py
import numpy as np, os
from src.data import load_mnist
from src.model_fp32 import LeNetLite
from src.trainer import train, evaluate

(x_tr,y_tr),(x_val,y_val),(x_te,y_te) = load_mnist()
m = LeNetLite()
m = train(m, x_tr, y_tr, x_val, y_val, epochs=1, lr=0.05, batch=64)
lat_ms, acc = evaluate(m, x_te, y_te)
print(f"test_acc={acc:.2f}%   lat(ms per predict batch=64)≈{lat_ms:.2f}")
os.makedirs("results", exist_ok=True)
np.savez("results/baseline_fp32.npz", **m.state_dict())
print("Pesos guardados en results/baseline_fp32.npz")
