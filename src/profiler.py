# src/profiler.py
import time, numpy as np

def latency_ms(fn, warmup=3, reps=30):
    for _ in range(warmup): fn()
    samples=[]
    for _ in range(reps):
        t0=time.perf_counter(); fn(); samples.append((time.perf_counter()-t0)*1000)
    arr=np.array(samples, dtype=np.float64)
    return {"mean": float(arr.mean()),
            "p50": float(np.percentile(arr,50)),
            "p90": float(np.percentile(arr,90))}

def model_size_bytes_fp32(state_dict):
    return sum(v.size * 4 for v in state_dict.values())

def model_size_bytes_int8(state_dict):
    # aproximación: todos los pesos a 1 byte (bias despreciable en este baseline)
    return sum(v.size for v in state_dict.values())

def fmt_size(n):
    for u in ["B","KB","MB","GB"]:
        if n<1024: return f"{n:.2f} {u}"
        n/=1024
    return f"{n:.2f} TB"
