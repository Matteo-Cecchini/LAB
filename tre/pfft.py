import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq, irfft
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import ds1000z

# tentativo di analisi dei dati con fft, fallimentare

csvs = ("12sec1.csv", "12sec2.csv", "12sec3.csv", "12sec4.csv", "12sec5.csv", "12sec6.csv")

data, inc = ds1000z.csv_formatter(pd.read_csv("12sec1.csv"))

ccs = rfft(data["ch1_volt"].values)
ffs = rfftfreq(len(ccs), d=1) * .5

sq = np.absolute(ccs)**2
inds, dic = find_peaks(sq, height=np.mean(sq))

pps = dic["peak_heights"]
print(pps*inc)
ccp = np.where(np.isin(np.arange(len(ccs)),inds[0]), ccs, np.zeros(len(ccs)))
ffp = ffs[inds]

yys = irfft(ccp)

plt.plot(ffs, np.absolute(ccs[:len(ffs)])**2)
plt.plot(ffp, pps)
plt.show()

plt.plot(data["time"], data["ch1_volt"])
plt.plot(data["time"][:-1], yys)
plt.show()