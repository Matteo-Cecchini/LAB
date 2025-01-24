from scipy.fft import rfft, rfftfreq, irfft
import scipy.fft as fft
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import soundfile as sf


class Spectrum:
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
        
    def _analize(self):
        self.coefficients = rfft(self.audio)
        self.frequencies = rfftfreq(len(self.coefficients), d=1/self.samplerate)*.5
        self.lenght = len(self.frequencies)
        
    def _find_peaks(self):
        if self.coefficients.size == 0:
            self._analize()
        mn, std = np.mean(np.absolute(self.coefficients[:len(self.frequencies)])**2), np.std(np.absolute(self.coefficients[:len(self.frequencies)])**2)
        ind, props = find_peaks(np.absolute(self.coefficients[:len(self.frequencies)])**2, height=mn + std)
        self.peaks = np.array([self.frequencies[ind].copy(), props["peak_heights"]])
        
    def plot(self, save: str = None):     
        if self.coefficients.size == 0 or self.peaks.size == 0:
            self._find_peaks()
        
        fig = plt.figure(figsize=(14,6))
        gs1 = fig.add_gridspec(1,2, left=.08, right=.6)
        ax1, ax2 = gs1.subplots()
        ax1.plot(self.audio, label="audio originale")
        ax2.plot(self.frequencies, np.absolute(self.coefficients[:len(self.frequencies)])**2, color="tomato", label="spettro di potenza")
        ax2.plot(self.peaks[0], self.peaks[1], 's', color="darkturquoise", alpha=.5, label="picchi oltre una deviazione")
        ax1.set_xlabel("n° di campionamento", size=15)
        ax1.set_ylabel("Ampiezza", size=15)
        ax2.set_xlabel("Frequenza in Hz", size=15)
        ax2.set_ylabel("Potenza dei coefficienti di Fourier", size=15)
        ax2.set_xlim(self.peaks[0][0] - 100, self.peaks[0][-1] + 100)
        ax1.legend()
        ax2.legend()
        
        gs2 = fig.add_gridspec(2, left=.68, right=.94, top=.8, bottom=.2, hspace=0)
        ax3, ax4 = gs2.subplots(sharex=True, sharey=True)
        ax3.plot(self.frequencies, self.coefficients[:len(self.frequencies)].real, color="navy", label="parte reale trasformata")
        ax4.plot(self.frequencies, self.coefficients[:len(self.frequencies)].imag, color="orange", label="parte immaginaria trasformata")
        #ax3.vlines(self.peaks[0], ymin=-3000, ymax=3000, colors="orange", alpha=.5)
        #ax4.vlines(self.peaks[0], ymin=-3000, ymax=3000, colors="navy", alpha=.5)
        ax3.set_xlabel("Frequenza in Hz", size=15)
        ax3.set_ylabel("Parte reale dei coefficienti", size=15)
        ax4.set_xlabel("Frequenza in Hz", size=15)
        ax4.set_ylabel("Parte immagianaria dei coefficienti", size=15)
        ax3.set_xlim(self.peaks[0][0] - 100, self.peaks[0][-1] + 100)
        ax4.set_xlim(self.peaks[0][0] - 100, self.peaks[0][-1] + 100)
        #ax3.set_xscale('log')
        #ax4.set_xscale('log')
        ax3.legend()
        ax4.legend()
        
        if save != None:
            plt.savefig(save)
        plt.show()
    
    def find_notes(self):
        if self.peaks.size == 0:
            self._find_peaks()
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
        mask = np.absolute(self.coefficients)**2 < np.mean(np.absolute(self.coefficients)**2) + 2*np.std(np.absolute(self.coefficients)**2)
        cofs = self.coefficients.copy()
        cofs[mask] = 0
        if first:
            cofs[0] = self.coefficients[0]
        if last:
            cofs[-1] = self.coefficients[-1]
        ys = irfft(cofs)
        
        plt.plot(self.audio, alpha=.3, label="audio originale")
        plt.plot(ys, alpha=.3, label="audio sintetizzato")
        plt.title("Sintesi con media e deviazione", size=15)
        plt.xlabel("n° campione", size=15)
        plt.ylabel("Ampiezza", size=15)
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
        if self.note.empty:
            self.find_notes()
        
        mask = np.isin(np.absolute(self.coefficients)**2, self.peaks[1], invert=True)
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
        ys = irfft(np.append(cofs, np.zeros(1000)))
        ys = ys[:-1000]
        plt.plot(self.audio, alpha=.3, label="audio originale")
        plt.plot(ys, alpha=.3, label="audio sintetizzato")
        plt.title("Sintesi con picco ed elementi vicini", size=15)
        plt.xlabel("N° campione", size=15)
        plt.ylabel("Ampiezza", size=15)
        plt.show()
        
        if file == None:
            return
        else:
            sf.write(file, ys, samplerate=self.samplerate)
    
    def __synth(self, n):
        if self.note.empty:
            self.find_notes()
        
        mask = np.isin(np.absolute(self.coefficients)**2, self.peaks[1], invert=True)
        false_indices = np.where(~mask)[0]
        for idx in false_indices:
            start = max(0, idx - n)
            end = min(len(mask), idx + n + 1)
            mask[start:end] = False

        cofs = self.coefficients.copy()
        cofs[mask] = 0
        cofs[0] = self.coefficients[0]
        cofs[-1] = self.coefficients[-1]
        
        return irfft(cofs)

            
    def all_synths(self, save: str = None, dir: str = None):
        a = [0, 2, 5]
        
        fig = plt.figure(figsize=(13,8))
        gs = fig.add_gridspec(3,1, hspace=0)
        axs = gs.subplots(sharex=True)
        for i,j in zip(axs,a):
            i.plot(self.audio, alpha=.2, label="segnale originale")
            i.plot(self.__synth(j), alpha=.6, label=f"sintesi con picco + {j} elementi per lato")            
            i.legend()
        fig.supxlabel("n° campione", size=15)
        fig.supylabel("ampiezza", size=15)
        fig.suptitle("Sintesi segnale con diverse estensioni del picco centrale", size=15)
        
        if save != None:
            if dir != None:
                save = dir + "/syn_" + save
            else:
                save = "syn_" + save
            plt.savefig(save)
        plt.show()
            
    def DIY_synthesis(self):
        pass
        
    def song_syn(self, n: int, file: str = None, wavdir: str = None, pngdir: str = None):
        if self.note.empty:
            self.find_notes()
        
        # indici per coefficienti
        b = np.where(self.frequencies > n)[0]
        bb = np.absolute(self.coefficients[b])**2
        
        # maschere per i coefficienti
        mb = np.isin(np.absolute(self.coefficients)**2, bb)
        mc = ~mb
        c1 = self.coefficients.copy()
        c2 = self.coefficients.copy()
        c1[mb] = 0 # elimina tutte le frequenze sopra soglia "n"
        c2[mc] = 0 # elimina tutte le frequenze sotto soglia "n"
        
        # sintesi discriminante
        yy1 = irfft(c1) # segnale basso
        yy2 = irfft(c2) # segnale chitarra
        
        # plot forme discriminate
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(13,6))
        ax1.plot(self.audio, alpha=.3)
        ax1.plot(yy1, alpha=.6)
        ax1.set_xlabel("N° campione", size=15)
        ax1.set_ylabel("Ampiezza", size=15)
        ax1.set_title("Basso", size=15)
        
        ax2.plot(self.audio, alpha=.3)
        ax2.plot(yy2, alpha=.6)
        ax2.set_xlabel("N° campione", size=15)
        ax2.set_ylabel("Ampiezza", size=15)
        ax2.set_title("Chitarra", size=15)
        
        if pngdir != None:  
            if file != None:
                path = pngdir + "/song_" + file
                plt.savefig(path)
        plt.show()
        
        # salvataggio file wav
        if file != None:
            if wavdir != None:
                path = wavdir + "/" + file
            sf.write(path + "_basso.wav", yy1, samplerate=self.samplerate)
            sf.write(path + "_chitarra.wav", yy2, samplerate=self.samplerate)