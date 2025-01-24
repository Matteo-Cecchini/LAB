import time
import numpy as np
from scipy.fft import rfft, fft
import matplotlib.pyplot as plt

a = np.sin(np.linspace(0,10,100, endpoint=False))

b = rfft(a)
d = fft(a).real

l = len(a)
j = l//2 + 1
c = np.zeros(l)
c[:j] = b
if l%2 == 0:
    c[j:] = np.conj(b[1:-1][::-1])
else:
    c[j:] = np.conj(b[1:][::-1])
print(a)
print(b)
print(len(a), len(b))
print(len(c))

for i,j,k in zip(c,d, range(l)):
    print(i==j, k)