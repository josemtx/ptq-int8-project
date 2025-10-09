# run_qsim.py
import os, csv, numpy as np
from src.data import load_mnist
from src.model_fp32 import LeNetLite
from src.calibrator import calibrate_weights_greedy, init_activation_qparams, tune_percentiles_SA, build_qsim_blocks, predict_qsim
from src.profiler import latency_ms, model_size_bytes_fp32, model_size_bytes_int8, fmt_size

def main():
    (x_tr,y_tr),(x_val,y_val),(x_te,y_te)=load_mnist()
    m=LeNetLite()
    sd=np.load("results/baseline_fp32.npz"); m.load_state_dict({k:sd[k] for k in sd.files})

    acc_fp32_val=(m.predict(x_val)==y_val).mean()*100.0
    acc_fp32_te =(m.predict(x_te )==y_te ).mean()*100.0
    size_fp32 = model_size_bytes_fp32(m.state_dict())
    print(f"[FP32] val={acc_fp32_val:.2f}%  test={acc_fp32_te:.2f}%  size={fmt_size(size_fp32)}")

    Xcal=x_val[:512]
    print("[PTQ] Pesos (greedy)..."); wq=calibrate_weights_greedy(m,Xcal)

    print("[PTQ] Acts init p=99.0..."); aq0=init_activation_qparams(m,Xcal,p=99.0)
    qs0=build_qsim_blocks(m,wq,aq0)
    acc0=(predict_qsim(m,qs0,x_val)==y_val).mean()*100.0
    print(f"[PTQ-init] val={acc0:.2f}%  Δ={acc0-acc_fp32_val:.2f} pts")

    print("[PTQ] SA percentiles...")
    aq=tune_percentiles_SA(m,wq,Xcal, x_val[:2000], y_val[:2000], acc_fp32=acc_fp32_val, iters=60, p0=99.0, step=0.7)
    qs=build_qsim_blocks(m,wq,aq)

    acc_ptq_val=(predict_qsim(m,qs,x_val)==y_val).mean()*100.0
    acc_ptq_te =(predict_qsim(m,qs,x_te )==y_te ).mean()*100.0
    size_int8 = model_size_bytes_int8(m.state_dict())

    lat_fp32=latency_ms(lambda: m.predict(x_te[:64]))
    lat_int8=latency_ms(lambda: predict_qsim(m,qs,x_te[:64]))

    print(f"[PTQ] val={acc_ptq_val:.2f}%  test={acc_ptq_te:.2f}%  Δtest={acc_ptq_te-acc_fp32_te:.2f} pts")
    print(f"[SIZE] fp32={fmt_size(size_fp32)}  int8≈{fmt_size(size_int8)}  (~{size_fp32/size_int8:.1f}x)")
    print(f"[LAT ] fp32 mean={lat_fp32['mean']:.1f}ms  int8 mean={lat_int8['mean']:.1f}ms")

    os.makedirs("results/tablas", exist_ok=True)
    with open("results/tablas/summary.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["mode","val_acc","test_acc","lat_mean_ms","lat_p50_ms","lat_p90_ms","size_bytes"])
        w.writerow(["fp32", f"{acc_fp32_val:.2f}", f"{acc_fp32_te:.2f}",
                    f"{lat_fp32['mean']:.2f}", f"{lat_fp32['p50']:.2f}", f"{lat_fp32['p90']:.2f}", size_fp32])
        w.writerow(["int8-qsim", f"{acc_ptq_val:.2f}", f"{acc_ptq_te:.2f}",
                    f"{lat_int8['mean']:.2f}", f"{lat_int8['p50']:.2f}", f"{lat_int8['p90']:.2f}", size_int8])
    print("OK → results/tablas/summary.csv")

if __name__=="__main__": main()
