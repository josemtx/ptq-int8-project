# src/trainer.py
import numpy as np
from .model_fp32 import LeNetLite, softmax
from .data import batches

def one_hot(y, num_classes=10):
    oh = np.zeros((len(y), num_classes), np.float32)
    oh[np.arange(len(y)), y] = 1.0
    return oh

def cross_entropy(logits, y):
    probs = softmax(logits)
    return -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-9)), probs

def train(model, x_tr, y_tr, x_val, y_val, epochs=1, lr=0.05, batch=64):
    for ep in range(epochs):
        losses=[]
        for xb, yb in batches(x_tr, y_tr, batch_size=batch, shuffle=True, seed=ep):
            # forward (conv no entrenan para simplificar; actualizamos solo lineales)
            # c1->r1->p1->c2->r2->p2
            z1=model.c1.forward(xb); a1=np.maximum(z1,0); p1=model.p1.forward(a1)
            z2=model.c2.forward(p1); a2=np.maximum(z2,0); p2=model.p2.forward(a2)
            flat=p2.reshape(p2.shape[0],-1)
            z3=model.fc1.forward(flat); a3=np.maximum(z3,0)
            logits=model.fc2.forward(a3)
            loss, probs = cross_entropy(logits, yb); losses.append(loss)
            # backward solo lineales
            y1h = one_hot(yb)
            dlogits = (probs - y1h)/len(yb)              # (N,10)
            dW2 = dlogits.T @ a3; db2 = dlogits.sum(0)
            da3 = dlogits @ model.fc2.W
            dz3 = da3.copy(); dz3[z3<=0]=0
            dW1 = dz3.T @ flat; db1 = dz3.sum(0)
            # SGD
            model.fc2.W -= lr*dW2; model.fc2.b -= lr*db2
            model.fc1.W -= lr*dW1; model.fc1.b -= lr*db1
        acc = (model.predict(x_val)==y_val).mean()*100
        print(f"Epoch {ep+1}: loss={np.mean(losses):.4f}  val_acc={acc:.2f}%")
    return model

def evaluate(model, x, y, repeats=50):
    import time
    t0=time.perf_counter()
    for _ in range(repeats): model.predict(x[:64])
    ms = (time.perf_counter()-t0)/repeats*1000
    acc = (model.predict(x)==y).mean()*100
    return ms, acc