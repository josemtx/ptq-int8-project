# src/quantizer.py
import numpy as np

# ----- helpers -----
def percentile_range(x, p=99.0):
    lo = np.percentile(x, 100 - p)
    hi = np.percentile(x, p)
    if lo == hi:  # evitar degenerados
        lo, hi = float(x.min()), float(x.max())
    return lo, hi

def minmax_range(x):
    return float(np.min(x)), float(np.max(x))

def choose_range(x, method="percentile", p=99.0):
    if method == "percentile": return percentile_range(x, p)
    elif method == "minmax":  return minmax_range(x)
    else: raise ValueError(f"method {method}")

# ----- escala y punto cero -----
def affine_params(a, b, num_bits=8, symmetric=False, signed=False):
    qmin, qmax = (-(2**(num_bits-1)), 2**(num_bits-1)-1) if signed else (0, 2**num_bits - 1)
    # asegurar que 0 cae en [a,b]
    a = min(a, 0.0); b = max(b, 0.0)
    if symmetric:
        m = max(abs(a), abs(b)); a, b = -m, m
    # escala y zp
    scale = (b - a) / (qmax - qmin) if b > a else 1.0
    if signed:
        zp = 0
    else:
        zp = round(qmin - a / scale)
        zp = int(np.clip(zp, qmin, qmax))
    return float(scale if scale != 0 else 1.0), int(zp), int(qmin), int(qmax)

def quantize(x, scale, zp, qmin, qmax, dtype=np.int8):
    q = np.round(x / scale + zp)
    q = np.clip(q, qmin, qmax).astype(np.int32)
    return q.astype(dtype)

def dequantize(q, scale, zp):
    return scale * (q.astype(np.float32) - zp)

# ----- pesos: per-tensor o per-canal -----
def fit_weight_qparams(W, scheme="per_tensor", symmetric=True, num_bits=8):
    # W: (OC, IC, KH, KW)
    signed = True  # pesos suelen ir en signed int8
    if scheme == "per_tensor":
        a, b = float(W.min()), float(W.max())
        s, z, qmin, qmax = affine_params(a, b, num_bits=num_bits, symmetric=symmetric, signed=signed)
        return {"scheme":"per_tensor","scale":s,"zp":z,"qmin":qmin,"qmax":qmax,"signed":signed}
    elif scheme == "per_channel":
        OC = W.shape[0]
        scales = np.zeros((OC,), np.float32); zps = np.zeros((OC,), np.int32)
        qmin = -(2**(num_bits-1)); qmax = 2**(num_bits-1)-1
        for oc in range(OC):
            a, b = float(W[oc].min()), float(W[oc].max())
            s, z, _, _ = affine_params(a, b, num_bits=num_bits, symmetric=symmetric, signed=signed)
            scales[oc], zps[oc] = s, z  # z=0 si signed
        return {"scheme":"per_channel","scale":scales,"zp":zps,"qmin":qmin,"qmax":qmax,"signed":signed}
    else:
        raise ValueError("scheme debe ser per_tensor o per_channel")

def quantize_weights(W, qparams):
    if qparams["scheme"] == "per_tensor":
        q = quantize(W, qparams["scale"], qparams["zp"], qparams["qmin"], qparams["qmax"], dtype=np.int8)
        return q
    else:
        OC = W.shape[0]; q = np.zeros_like(W, dtype=np.int8)
        for oc in range(OC):
            q[oc] = quantize(W[oc], qparams["scale"][oc], qparams["zp"][oc], qparams["qmin"], qparams["qmax"], dtype=np.int8)
        return q

# ----- activaciones (per-tensor) -----
def fit_act_qparams(x_samples, method="percentile", p=99.0, num_bits=8, symmetric=False, signed=False):
    a, b = choose_range(x_samples, method=method, p=p)
    s, z, qmin, qmax = affine_params(a, b, num_bits=num_bits, symmetric=symmetric, signed=signed)
    dtype = np.int8 if signed else np.uint8
    return {"scale":s,"zp":z,"qmin":qmin,"qmax":qmax,"dtype":dtype, "method":method, "p":p, "symmetric":symmetric, "signed":signed}
