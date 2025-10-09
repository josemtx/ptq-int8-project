# src/calibrator.py
import numpy as np
from .quantizer import fit_weight_qparams, fit_act_qparams
from .qsim_engine import QSimConvReLU, QSimLinear
from .model_fp32 import softmax

def forward_intermediates(m, x):
    z1 = m.c1.forward(x); a1 = np.maximum(z1, 0); p1 = m.p1.forward(a1)
    z2 = m.c2.forward(p1); a2 = np.maximum(z2, 0); p2 = m.p2.forward(a2)
    flat = p2.reshape(p2.shape[0], -1)
    z3 = m.fc1.forward(flat); a3 = np.maximum(z3, 0)
    logits = m.fc2.forward(a3)
    return dict(inp=x, a1=a1, p1=p1, a2=a2, p2=p2, flat=flat, a3=a3, logits=logits)

def build_qsim_blocks(m, wq, aq):
    qs = {}
    qs["c1"]  = QSimConvReLU(m.c1.W, m.c1.b, wq["c1"], aq["inp"], aq["a1"], fuse_relu=True)
    qs["c2"]  = QSimConvReLU(m.c2.W, m.c2.b, wq["c2"], aq["p1"],  aq["a2"], fuse_relu=True)  # <- p1
    qs["fc1"] = QSimLinear  (m.fc1.W, m.fc1.b, wq["fc1"], aq["flat"], aq["a3"])
    qs["fc2"] = QSimLinear  (m.fc2.W, m.fc2.b, wq["fc2"], aq["a3"],  aq["out"])
    return qs

def predict_qsim(m, qs, x):
    x=qs["c1"].forward(x); x=m.p1.forward(x)
    x=qs["c2"].forward(x); x=m.p2.forward(x)
    x=x.reshape(x.shape[0],-1)
    x=qs["fc1"].forward(x)
    logits=qs["fc2"].forward(x)
    probs=softmax(logits)
    return np.argmax(probs, axis=1)

# --- Greedy pesos (por capa) ---
def calibrate_weights_greedy(m, Xcal, max_samples=256):
    X = Xcal[:max_samples]
    inter = forward_intermediates(m, X)
    best = {}
    for layer in ["c1","c2","fc1","fc2"]:
        W = getattr(m, layer).W
        candidates = [("per_channel", True), ("per_tensor", True), ("per_tensor", False)]
        best_err, best_q = 1e30, None
        for scheme, sym in candidates:
            q = fit_weight_qparams(W, scheme=scheme, symmetric=sym)
            if layer == "c1":
                qs = QSimConvReLU(W, getattr(m, layer).b, q,
                                  fit_act_qparams(inter["inp"]),  # in
                                  fit_act_qparams(inter["a1"]))   # out
                y_hat = qs.forward(inter["inp"]);  y_ref = inter["a1"]
            elif layer == "c2":
                qs = QSimConvReLU(W, getattr(m, layer).b, q,
                                  fit_act_qparams(inter["p1"]),   # <-- in correcto (post-pool)
                                  fit_act_qparams(inter["a2"]))   # out
                y_hat = qs.forward(inter["p1"]);   y_ref = inter["a2"]
            elif layer == "fc1":
                qs = QSimLinear(W, getattr(m, layer).b, q,
                                fit_act_qparams(inter["flat"]),
                                fit_act_qparams(inter["a3"]))
                y_hat = qs.forward(inter["flat"]); y_ref = inter["a3"]
            else:  # fc2
                qs = QSimLinear(W, getattr(m, layer).b, q,
                                fit_act_qparams(inter["a3"]),
                                fit_act_qparams(inter["logits"], symmetric=True, signed=True))
                y_hat = qs.forward(inter["a3"]);  y_ref = inter["logits"]
            err = float(np.mean((y_ref - y_hat) ** 2))
            if err < best_err: best_err, best_q = err, q
        best[layer] = best_q
    return best


def init_activation_qparams(m, Xcal, p=99.0):
    inter = forward_intermediates(m, Xcal)
    aq = {}
    aq["inp"]  = fit_act_qparams(inter["inp"],  method="percentile", p=p, symmetric=False, signed=False)
    aq["a1"]   = fit_act_qparams(inter["a1"],   method="percentile", p=p, symmetric=False, signed=False)
    aq["p1"]   = fit_act_qparams(inter["p1"],   method="percentile", p=p, symmetric=False, signed=False)  # <- NUEVO
    aq["a2"]   = fit_act_qparams(inter["a2"],   method="percentile", p=p, symmetric=False, signed=False)
    aq["flat"] = fit_act_qparams(inter["flat"], method="percentile", p=p, symmetric=False, signed=False)
    aq["a3"]   = fit_act_qparams(inter["a3"],   method="percentile", p=p, symmetric=False, signed=False)
    aq["out"]  = fit_act_qparams(inter["logits"], method="percentile", p=p, symmetric=True,  signed=True)
    return aq

def tune_percentiles_SA(m, wq, Xcal, Xval, Yval, acc_fp32, iters=60, p0=99.0, step=0.7):
    rng = np.random.default_rng(0)
    aq = init_activation_qparams(m, Xcal, p=p0)
    inter = forward_intermediates(m, Xcal)  # cache

    src_map = {"inp":"inp","a1":"a1","p1":"p1","a2":"a2","flat":"flat","a3":"a3","out":"logits"}

    def rebuild(aq_in):
        b = {}
        for k, src in src_map.items():
            sym = (k == "out"); signed = sym
            b[k] = fit_act_qparams(inter[src], method="percentile", p=aq_in[k]["p"],
                                   symmetric=sym, signed=signed)
        return b

    def eval_acc(aq_in):
        qs = build_qsim_blocks(m, wq, aq_in)
        y  = predict_qsim(m, qs, Xval)
        return (y == Yval).mean() * 100.0

    best = aq.copy(); best_acc = curr_acc = eval_acc(aq)
    T = 1.0
    for _ in range(iters):
        cand = {k: v.copy() for k, v in aq.items()}
        for k in cand:
            cand[k]["p"] = float(np.clip(cand[k]["p"] + rng.uniform(-step, step), 90.0, 99.9))
        cand = rebuild(cand)
        a = eval_acc(cand)
        # restricción dura: no aceptar candidatos que caigan >2 pts vs FP32
        if (acc_fp32 - a) <= 2.0 and (a > curr_acc or np.exp((a - curr_acc) / max(T, 1e-8)) > rng.uniform()):
            aq, curr_acc = cand, a
            if a > best_acc:
                best, best_acc = cand, a
        T = max(T * 0.95, 1e-3)
    return best
