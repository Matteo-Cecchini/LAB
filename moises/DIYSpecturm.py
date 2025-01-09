from scipy.integrate import simpson
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import soundfile as sf


class DIYSpectrum:
    audio = np.array([])
    coefficients = np.array([])
    frequencies = np.array([])
    samplerate = 0
    peaks = np.array([])
    note = pd.DataFrame()
    
    def __init__(self, path):
        audio, samplerate = sf.read(path)
        
        self.audio = audio.transpose()[0]
        self.samplerate = samplerate
        
    def _fft(self):
        pass
        
    def analize(self):
        
        self.lenght = len(self.frequencies)
        
    def find_peaks(self):
        if self.coefficients.size == 0:
            self.analize()
        mn, std = np.mean(np.absolute(self.coefficients[:len(self.frequencies)])**2), np.std(np.absolute(self.coefficients[:len(self.frequencies)])**2)
        ind, props = find_peaks(np.absolute(self.coefficients[:len(self.frequencies)])**2, height=mn + std)
        self.peaks = np.array([self.frequencies[ind].copy(), props["peak_heights"]])
        
    def plot(self):     
        if self.coefficients.size == 0 or self.peaks.size == 0:
            self.find_peaks()
        
        fig = plt.figure(figsize=(14,6))
        gs1 = fig.add_gridspec(1,2, left=.08, right=.6)
        ax1, ax2 = gs1.subplots()
        ax1.plot(self.audio, label="audio originale")
        ax2.plot(self.frequencies, np.absolute(self.coefficients[:len(self.frequencies)])**2, color="tomato", label="spettro di potenza")
        ax2.plot(self.peaks[0], self.peaks[1], 's-', color="darkturquoise")
        ax1.set_xlabel("n° of sample")
        ax1.set_ylabel("sample amplitude")
        ax2.set_xlabel("frequency in Hz")
        ax2.set_ylabel("power of Fourier transform's coefficients' power")
        ax1.legend()
        ax2.legend()
        
        gs2 = fig.add_gridspec(2, left=.68, right=.94, top=.8, bottom=.2, hspace=0)
        ax3, ax4 = gs2.subplots(sharex=True, sharey=True)
        ax3.plot(self.frequencies, self.coefficients[:len(self.frequencies)].real, color="navy", label="parte reale trasformata")
        ax4.plot(self.frequencies, self.coefficients[:len(self.frequencies)].imag, color="orange", label="parte immaginaria trasformata")
        ax3.vlines(self.peaks[0], ymin=-3000, ymax=3000, colors="orange")
        ax4.vlines(self.peaks[0], ymin=-3000, ymax=3000, colors="navy")
        ax3.set_xlabel("frequency in Hz")
        ax3.set_ylabel("real part of Fourier transform's coefficient")
        ax4.set_xlabel("frequency in Hz")
        ax4.set_ylabel("imaginary part of Fourier transform's coefficient")
        ax3.set_xscale('log')
        ax4.set_xscale('log')
        ax3.legend()
        ax4.legend()
        
        plt.show()
    
    def find_notes(self):
        if self.peaks.size == 0:
            self.find_peaks()
        n = 12 * np.log2(self.peaks[0] / 440)
        n_rounded = np.array([round(i) for i in n])
        posizione = n_rounded % 12
        note = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
        ottava = 4 + (n_rounded // 12)
        nota = [note[i] + str(j) for i,j in zip(posizione, ottava)]
        powsum = self.peaks[1] / np.sum(self.peaks[1])
        
        df = pd.DataFrame({"Frequenze":self.peaks[0], "Note":nota, "Percentuale": powsum*100})
        self.note = df.groupby("Note", as_index=False).agg({"Frequenze" : "mean", "Percentuale" : "sum"})

        
    def drop_notes(self):
        if self.note.empty:
            self.find_notes()
        
        print(self.note)
        print("Tutte le note","\n")
        
        mnpcr, stdpcr = np.mean(self.note["Percentuale"]), np.std(self.note["Percentuale"])
        short = self.note.loc[self.note["Percentuale"] > mnpcr + .5*stdpcr]
        print(short)
        print("Note con percentuale filtrata rispetto la media delle percentuali (oltre mezza deviazione, ricordando che le percentuali sono associate al rapporto tra l'altezza del picco singolo e la somma di tutte le altezze dei picchi)")

    def mean_synthesis(self, file: str = None, first: bool = False, last: bool = False):
        mask = np.absolute(self.coefficients)**2 < np.mean(np.absolute(self.coefficients)*2) + 2*np.std(np.absolute(self.coefficients)*2)
        cofs = self.coefficients.copy()
        cofs[mask] = 0
        if first:
            cofs[0] = self.coefficients[0]
        if last:
            cofs[-1] = self.coefficients[-1]
        ys = irfft(cofs)
        
        plt.plot(self.audio, alpha=.3)
        plt.plot(ys, alpha=.3)
        plt.show()
        
        if file == None:
            return
        else:
            sf.write(file, ys, samplerate=self.samplerate)
    
    def _peak_int(self, bl, n, size):
        mn = min(0, bl - n)
        mx = max(size, bl + n + 1)
        return range(mn, mx)
        
    
    def peak_synthesis(self, n: int, file: str = None, first: bool = False, last: bool = False):
        mask = np.array([np.absolute(i)**2 not in self.peaks[1] for i in self.coefficients])
        false_indices = np.where(~mask)[0]
        for idx in false_indices:
            start = max(0, idx - n)
            end = min(len(mask), idx + n + 1)
            mask[start:end] = False

        cofs = self.coefficients.copy()
        cofs[mask] = 0
        if first:
            cofs[0] = self.coefficients[0]
        if last:
            cofs[-1] = self.coefficients[-1]
        ys = irfft(cofs)
        
        plt.plot(self.audio, alpha=.3)
        plt.plot(ys, alpha=.3)
        plt.show()
        
        if file == None:
            return
        else:
            sf.write(file, ys, samplerate=self.samplerate)