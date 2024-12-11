import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from re import split
from scipy.fft import rfft, rfftfreq, irfft

stream = open("data1.txt", "r")
data = np.array([split(r'[\t\n\s]', i)[:2] for i in stream.readlines()], dtype=np.float64)
data = data.transpose()

coeff = rfft(data[1])
freq = rfftfreq(len(coeff), d=1/44100) / 2

'''
plt.plot(data[0], data[1], 'o-')
plt.show()
'''
plt.plot(freq, np.absolute(coeff[:len(freq)])**2, '*-', color="tomato")
#plt.xscale('log')
plt.yscale('log')
plt.show()

mask = np.absolute(coeff)**2 < 1e9
unmask = [not i for i in mask]
ucoeff = coeff.copy()
ucoeff[mask] = 0
ucoeff[unmask] = [0, ucoeff[unmask][1]]
print(ucoeff)
print(ucoeff[unmask])
print(freq[unmask[:len(freq)]])


ys = irfft(coeff)
uys = irfft(ucoeff)


plt.plot(data[0], ys, alpha=.2)
plt.plot(data[0], uys)
plt.show()
