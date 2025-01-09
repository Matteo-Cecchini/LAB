import time
import numpy as np
from scipy.fft import rfft
import matplotlib.pyplot as plt

def uno(cs, n):
    leff = len(cs)
    cost = np.pi / leff     # non 2pi perchè uso rfft, che calcola nel dominio [0;pi]
    ns = np.arange(0, leff)
    nn = np.arange(0, n)
    
    recos = np.array([np.sum(cs.real*np.cos(cost*ns*i)) for i in nn])
    resin = np.array([np.sum(cs.real*np.sin(cost*ns*i)) for i in nn])
    imcos = np.array([np.sum(cs.imag*np.cos(cost*ns*i)) for i in nn])
    imsin = np.array([np.sum(cs.imag*np.sin(cost*ns*i)) for i in nn])
    
    re = recos - imsin
    im = imcos + resin
    
    xs = ( re + 1j*im ) / leff
    return xs

def due(cs):
    n = len(cs)
    arg = ( np.arange(n) * np.pi ) / n
    ind = np.arange(n)
    



a = np.linspace(0,100,1000, endpoint=False)
a = np.sin(a)
b = rfft(a)

t = time.time()
c = uno(b, len(b)*2)
t1 = time.time() - t

print(t1)
plt.plot(a)
plt.plot(c)
plt.show()