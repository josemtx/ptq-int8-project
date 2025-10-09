# src/qsim_engine.py
import numpy as np
from .quantizer import quantize, dequantize

def conv2d_int(x_q, w_q, z_x, z_w, bias_int32, stride=1, padding=0):
    # x_q: (N,C,H,W) int8/uint8 ; w_q: (OC,IC,KH,KW) int8 ; salida int32 sin reescala
    N,C,H,W = x_q.shape
    OC,IC,KH,KW = w_q.shape
    s, p = stride, padding
    xpad = np.pad(x_q, ((0,0),(0,0),(p,p),(p,p)), mode='constant', constant_values=z_x)
    Ho = (H + 2*p - KH)//s + 1
    Wo = (W + 2*p - KW)//s + 1
    y32 = np.zeros((N, OC, Ho, Wo), dtype=np.int32)
    for n in range(N):
        for oc in range(OC):
            for i in range(Ho):
                for j in range(Wo):
                    patch = xpad[n, :, i*s:i*s+KH, j*s:j*s+KW].astype(np.int32)
                    w = (w_q[oc].astype(np.int32) - z_w)
                    x = (patch - z_x)
                    y32[n, oc, i, j] = np.sum(x * w) + int(bias_int32[oc])
    return y32

def linear_int(x_q, w_q, z_x, z_w, bias_int32):
    # x_q: (N,Fin), w_q: (Fout,Fin)
    x = (x_q.astype(np.int32) - z_x)
    w = (w_q.astype(np.int32) - z_w)
    y32 = x @ w.T + bias_int32.astype(np.int32)
    return y32

def requantize_int32(y32, s_x, s_w, s_y, z_y, qmin, qmax, out_dtype):
    # y_q = round(y32 * (s_x*s_w)/s_y + z_y)
    scale = (s_x * s_w) / s_y
    y = np.round(y32.astype(np.float32) * scale) + z_y
    y = np.clip(y, qmin, qmax).astype(np.int32)
    return y.astype(out_dtype)

class QSimConvReLU:
    def __init__(self, W_fp32, b_fp32, w_qparams, act_qparams_in, act_qparams_out, fuse_relu=True):
        self.fuse_relu = fuse_relu
        self.w_q = None
        self.W_fp32 = W_fp32
        self.b_fp32 = b_fp32.astype(np.float32)
        self.w_qparams = w_qparams
        self.act_in = act_qparams_in   # dict con scale/zp/qmin/qmax/dtype
        self.act_out = act_qparams_out
        self._prepare()

    def _prepare(self):
        # cuantiza pesos
        from .quantizer import quantize_weights
        self.w_q = quantize_weights(self.W_fp32, self.w_qparams)
        # bias en int32 con escalado (s_w*s_x/s_y aplicado en requant; aquí lo dejamos en FP32->INT32 direct)
        self.bias_int32 = np.round(self.b_fp32 / self.act_out["scale"]).astype(np.int32) * 0

    def forward(self, x_fp32, stride=1, padding=0):
        # quant in
        x_q = quantize(x_fp32, self.act_in["scale"], self.act_in["zp"], self.act_in["qmin"], self.act_in["qmax"], dtype=self.act_in["dtype"])
        # conv int32
        z_x = self.act_in["zp"]; z_w = 0 if self.w_qparams.get("signed", True) else self.w_qparams["zp"]
        y32 = conv2d_int(x_q, self.w_q, z_x, z_w, bias_int32=self.bias_int32, stride=stride, padding=padding)
        # requant
        y_q = requantize_int32(
            y32,
            self.act_in["scale"],
            # ANTES: self.w_qparams["scale"][:,None,None,None]
            self.w_qparams["scale"] if self.w_qparams["scheme"]=="per_tensor"
            else self.w_qparams["scale"][None, :, None, None],   # (1, OC, 1, 1) ✅
            self.act_out["scale"], self.act_out["zp"],
            self.act_out["qmin"], self.act_out["qmax"],
            self.act_out["dtype"]
        )
        # ReLU en dtype destino
        if self.fuse_relu:
            y_q = np.maximum(y_q, self.act_out["zp"]).astype(self.act_out["dtype"])
        # dequant out
        y_fp32 = dequantize(y_q, self.act_out["scale"], self.act_out["zp"])
        return y_fp32

class QSimLinear:
    def __init__(self, W_fp32, b_fp32, w_qparams, act_qparams_in, act_qparams_out):
        self.W_fp32, self.b_fp32 = W_fp32, b_fp32.astype(np.float32)
        self.w_qparams = w_qparams
        from .quantizer import quantize_weights
        self.w_q = quantize_weights(self.W_fp32, self.w_qparams)
        self.act_in, self.act_out = act_qparams_in, act_qparams_out
        self.bias_int32 = np.round(self.b_fp32 / self.act_out["scale"]).astype(np.int32) * 0

    def forward(self, x_fp32):
        x_q = quantize(x_fp32, self.act_in["scale"], self.act_in["zp"], self.act_in["qmin"], self.act_in["qmax"], dtype=self.act_in["dtype"])
        z_x = self.act_in["zp"]; z_w = 0 if self.w_qparams.get("signed", True) else self.w_qparams["zp"]
        y32 = linear_int(x_q, self.w_q, z_x, z_w, self.bias_int32)
        y_q = requantize_int32(
            y32,
            self.act_in["scale"],
            # ANTES: self.w_qparams["scale"][:,None]
            self.w_qparams["scale"] if self.w_qparams["scheme"]=="per_tensor"
            else self.w_qparams["scale"][None, :],               # (1, Fout) ✅
            self.act_out["scale"], self.act_out["zp"],
            self.act_out["qmin"], self.act_out["qmax"],
            self.act_out["dtype"]
        )
        y_fp32 = dequantize(y_q, self.act_out["scale"], self.act_out["zp"])
        return y_fp32
