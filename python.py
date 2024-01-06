import time
import numpy as np
N = 1024
with open("mat.dat", 'rb') as f:
    a = np.fromfile(f,dtype=np.float32,
            count=N*N)
    b = np.fromfile(f,dtype=np.float32,
            count=N*N)

a = a.reshape((N,N))
b = b.reshape((N,N))
gflops = 0
for i in range(5):
    st = time.monotonic()
    c = a@a
    et = time.monotonic()
    sec = et-st
    gflop = N*N*N*2*1e-9
    gflops += gflop/sec
    if not(b.all() == c.all()):
        print("Fail")
        exit(1)
    print("GFLOPS : %f"%(gflop/sec))
print("Avg GFLOPS : %f"%(gflops/5))


