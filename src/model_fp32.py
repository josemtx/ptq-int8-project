# src/model_fp32.py
import numpy as np

def kaiming_uniform(shape, fan_in):
    bound = np.sqrt(6.0 / fan_in); return np.random.uniform(-bound, bound, size=shape).astype(np.float32)

class Conv2D:
    def __init__(self, in_c, out_c, k=5, stride=1, padding=0):
        self.stride, self.padding = stride, padding
        self.W = kaiming_uniform((out_c, in_c, k, k), in_c*k*k)
        self.b = np.zeros((out_c,), np.float32)
    def forward(self, x):
        N,C,H,W = x.shape; OC,_,KH,KW = self.W.shape; s,p = self.stride, self.padding
        xpad = np.pad(x, ((0,0),(0,0),(p,p),(p,p)))
        Ho, Wo = (H+2*p-KH)//s + 1, (W+2*p-KW)//s + 1
        y = np.zeros((N, OC, Ho, Wo), np.float32)
        for n in range(N):
            for oc in range(OC):
                for i in range(Ho):
                    for j in range(Wo):
                        patch = xpad[n, :, i*s:i*s+KH, j*s:j*s+KW]
                        y[n, oc, i, j] = np.sum(patch * self.W[oc]) + self.b[oc]
        return y

class ReLU:
    def forward(self, x): return np.maximum(x, 0)
class MaxPool2D:
    def __init__(self, k=2): self.k = k
    def forward(self, x):
        N,C,H,W = x.shape; k=self.k; Ho,Wo = H//k, W//k
        y = np.zeros((N,C,Ho,Wo), np.float32)
        for n in range(N):
            for c in range(C):
                for i in range(Ho):
                    for j in range(Wo):
                        y[n,c,i,j] = np.max(x[n,c,i*k:(i+1)*k, j*k:(j+1)*k])
        return y

class Linear:
    def __init__(self, in_f, out_f):
        self.W = kaiming_uniform((out_f, in_f), in_f)
        self.b = np.zeros((out_f,), np.float32)
    def forward(self, x): return x @ self.W.T + self.b

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True); e = np.exp(x, dtype=np.float32)
    return e / np.sum(e, axis=1, keepdims=True)

class LeNetLite:
    def __init__(self):
        self.c1=Conv2D(1,8,5); self.r1=ReLU(); self.p1=MaxPool2D(2)
        self.c2=Conv2D(8,16,5); self.r2=ReLU(); self.p2=MaxPool2D(2)
        self.fc1=Linear(16*4*4,64); self.r3=ReLU(); self.fc2=Linear(64,10)
    def forward(self,x):
        x=self.p1.forward(self.r1.forward(self.c1.forward(x)))
        x=self.p2.forward(self.r2.forward(self.c2.forward(x)))
        x=x.reshape(x.shape[0],-1); x=self.r3.forward(self.fc1.forward(x))
        return self.fc2.forward(x)
    def predict(self,x): return np.argmax(softmax(self.forward(x)), axis=1)
    def state_dict(self):
        return {"c1.W":self.c1.W,"c1.b":self.c1.b,"c2.W":self.c2.W,"c2.b":self.c2.b,
                "fc1.W":self.fc1.W,"fc1.b":self.fc1.b,"fc2.W":self.fc2.W,"fc2.b":self.fc2.b}
    def load_state_dict(self,sd):
        self.c1.W,self.c1.b=sd["c1.W"],sd["c1.b"]; self.c2.W,self.c2.b=sd["c2.W"],sd["c2.b"]
        self.fc1.W,self.fc1.b=sd["fc1.W"],sd["fc1.b"]; self.fc2.W,self.fc2.b=sd["fc2.W"],sd["fc2.b"]
