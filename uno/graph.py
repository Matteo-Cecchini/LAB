import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

a = pd.read_csv("data.csv")

vinpp = .568
verr = .008

freq = a["freq"].values
vpps = a["Vpp"].values

# Vpp input è costante, quindi si può trovare la banda passante direttamente dai Vpp out
Vpass = .707 * 1.18
Vpass_err = .707 * verr
print(f"Il voltaggio picco-picco corrispondente alla banda passante vale {Vpass} +- {Vpass_err}")

mask = vpps < Vpass

guads = 20*np.log10(vpps / vinpp)
used_guads = 20*np.log10(vpps[mask] / vinpp)
guads_err = 20*np.sqrt(((vinpp/vpps[mask])**2) * (verr**2 + (verr/(vinpp**2))**2))
freqs = np.log10(freq)
used_freqs = np.log10(freq[mask])

def lin(x, a, b):
    return a*x + b

p, c = curve_fit(lin, used_freqs, used_guads, p0=(-20,100), sigma=guads_err, absolute_sigma=False)
print(p, np.sqrt(np.diag(c)))
y = lin(used_freqs, p[0], p[1])

norm = 183e3

plt.plot(freqs, guads)
plt.plot(used_freqs, used_guads)
plt.plot(used_freqs, y)

plt.show()