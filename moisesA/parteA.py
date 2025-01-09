import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from re import split
from scipy.fft import rfft, rfftfreq, irfft
from scipy.signal import find_peaks
import os

def resynth(cs, n):
    leff = len(cs)
    cost = np.pi / leff     # non 2pi perchè uso rfft, che calcola nel dominio [0;pi]
    ns = np.arange(0, leff)
    nn = np.arange(0, n)
    
    re = np.array([np.sum(cs*np.cos(cost*ns*i)) for i in nn])
    im = np.array([np.sum(cs*np.sin(cost*ns*i)) for i in nn])
    
    xs = ( re + 1j*im ) / leff
    return xs

directory = "parteA"
colori = "darkturquoise", "limegreen", "tomato"
check = os.listdir(directory)

samplerate = 1 / 44100
arrlen = 10 * 44100 # lunghezza degli array sapendo che la freq. di campionamento è 44100 Hz
cutlen = arrlen//2 + 1

for i, j in zip(os.listdir(directory), colori):
    stream = open(os.path.join(directory,i), "r")
    data = np.array([split(r'[\t\n\s]', i)[:2] for i in stream.readlines()], dtype=np.float64)
    data = data.transpose()
    filename = i.strip(".txt")
    # grafici waveform
    '''
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5), sharey=True)
    ax1.plot(data[0], data[1], c=j)
    ax2.plot(data[0], data[1], c=j)
    ax3.plot(data[0], data[1], c=j)
    
    ax2.set_xlim(0,.1)
    ax3.set_xlim(0,.05)
    
    ax1.set_xlabel("Zoom 1x (tutto il segnale)", size=15)
    ax2.set_xlabel("Zoom 10x (fino a un $10^o$ del segnale)", size=15)
    ax3.set_xlabel("Zoom 20x (fino a un $20^o$ del segnale)", size=15)
    
    plt.suptitle("Waveform - " + i, size=20)
    plt.tight_layout()
    plt.show()
    '''
    coeff = rfft(data[1])
    freqs = rfftfreq(len(data[0]), d=samplerate)
    # grafici spettro
    '''
    fig = plt.figure(figsize=(10,5))
    left = fig.add_gridspec(1,1, left=.05, right=.45)
    right = fig.add_gridspec(2,1, left=.55, right=.95)
    
    ps = left.subplots()
    ps.plot(freqs, np.absolute(coeff[:cutlen])**2, c=j)
    ps.set_title("Spettro di potenza", size=15)
    ps.set_xlabel("Frequenza (Hz)", size = 11.5)
        
    re, im = right.subplots(sharex=True)
    re.plot(freqs, coeff[:cutlen].real, c="black")
    re.set_title("Parte Reale", size=13)
    im.plot(freqs, coeff[:cutlen].imag, c="gray")
    im.set_title("Parte Immaginaria", size=13)
    im.set_xlabel("Frequenza (Hz)", size = 11.5)
    
    plt.tight_layout()
    plt.show()
    '''
    # sintesi
    '''
    n = 1200
    pts = resynth(coeff, n)
    fft = irfft(coeff, arrlen)
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
    ax1.plot(data[1][:2*n], label="Segnale originale", c=j)
    ax1.plot(pts, label="Segnale ripristinato seni/coseni", c="plum")
    ax1.legend()
    
    ax2.plot(data[1][:2*n], label="Segnale originale", c=j)
    ax2.plot(fft[:n], label="Segnale ripristinato irfft", c="plum")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    '''
    pws = np.absolute(coeff[:cutlen])**2
    mn, std = np.mean(pws), np.std(pws)
    ind, pks = find_peaks(pws, height=mn)
    print(freqs[ind])
    m0 = np.max(pks["peak_heights"])
    
    cfs = coeff.copy()
    if i == check[0]:
        msk = [i != m0 for i in np.absolute(coeff)**2]
        cfs[msk] = 0
    elif i == check[1]:
        mm = m0 / np.arange(1,16, 2)**4
        print(pks["peak_heights"])
        print(mm)
        tolerance = 0.05  # tolleranza relativa del 5%
        msk = [all(not (m * (1 - tolerance) <= val <= m * (1 + tolerance)) for m in mm) for val in np.abs(coeff)**2]
        cfs[msk] = 0
    else:
        mm = m0 / np.arange(1,16, 2)**2
        print(pks["peak_heights"])
        print(mm)
        tolerance = 0.05  # tolleranza relativa del 5%
        msk = [all(not (m * (1 - tolerance) <= val <= m * (1 + tolerance)) for m in mm) for val in np.abs(coeff)**2]
        cfs[msk] = 0
    
    fft = irfft(cfs)
    plt.plot(data[0], data[1], label="segnale originale", c=j)
    plt.plot(data[0], fft, label="segnale filtrato")
    plt.xlim(0,.1)
    plt.legend()
    plt.show()