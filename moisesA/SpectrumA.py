from scipy.fft import rfft, rfftfreq, irfft
import scipy.fft as fft
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import soundfile as sf
import soundcard as sc
from time import time
from numba import njit
import multiprocessing
from re import split

@njit
def DIY_ifft(indexes, nonzeros, coefficients, tot):
    """
    Funzione ottimizzata per ricostruire il segnale usando la formula dell'IFFT.
    """
    result = np.zeros(len(indexes), dtype=np.float64)
    for i in range(len(indexes)):
        # Calcolo dell'esponenziale per ogni elemento
        args = 2j * np.pi * indexes[i] * nonzeros / tot
        exps = np.exp(args) * coefficients
        result[i] = np.sum(exps).real / tot  # Solo la parte reale
    return result

class Spectrum:
    timer = np.array([])
    audio = np.array([])
    coefficients = np.array([])
    frequencies = np.array([])
    samplerate = 0
    peaks = np.array([])
    note = pd.DataFrame()
    
    def __init__(self, path: str = None, rec: int = None):
        if path != None:
            self.samplerate = 44100
            lines = open(path, "r").readlines()
            data = np.array([split(r'[\t\n\s]', i)[:2] for i in lines], dtype=np.float64).transpose()
            self.timer = data[0]
            self.audio = data[1]

        elif rec != None:
            self.samplerate = 44100
            with sc.default_microphone().recorder(samplerate=self.samplerate, channels=2, blocksize=1024) as mic:
                print("Recording...")
                a = time()
                array = mic.record(int(self.samplerate * rec))
                b = time()
                print(f"Recording finished. {b-a} seconds recorded.")
                self.audio = array.transpose()[0] / np.max(array.transpose()[0])
    
    def listen(self, save: str = None):
        with sc.default_speaker().player(samplerate=self.samplerate, channels=2, blocksize=1024) as player:
            print("Listen...")
            player.play(self.audio)
            print("End.")
        if save != None:
            sf.write(save + ".wav", self.audio, samplerate=self.samplerate)
        
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
        
    def plot(self, name: str = None, title: str = None, c:str = None, zoom:int = None):     
        if self.coefficients.size == 0 or self.peaks.size == 0:
            self._find_peaks()
            
        f1, f2 = plt.figure(figsize=(12,5)), plt.figure(figsize=(12,5))
        g1 = f1.add_gridspec(3,1)
        g21, g22 = f2.add_gridspec(1,1, right=.45), f2.add_gridspec(2,1, left=.55)
        
        axs1 = g1.subplots(sharey=True).flatten()
        zms = np.array([1,10,20])
        if zoom != None:
            zms *= zoom
        for i,j in zip(axs1, 1/zms):
            if j == 1:
                i.plot(self.timer, self.audio, c=c, label="Primo secondo del segnale")
            else:
                i.plot(self.timer, self.audio, c=c, label=f"Fino a un ${int(1/j)}^o$ di secondo del segnale")
            i.set_xlim(0,j)
            i.legend(loc="upper left")
        f1.suptitle(f"Waveform di {name}", size=15)
        f1.supxlabel("Tempo in secondi", size=15)
        f1.supylabel("Ampiezza in unità arbitrarie", size=15)
        
        ax2 = g21.subplots()
        ax2.plot(self.frequencies, np.absolute(self.coefficients[:len(self.frequencies)])**2, c=c)
        ax2.plot(self.peaks[0], self.peaks[1], 's', color="darkturquoise", alpha=.5, label="picchi oltre una deviazione")
        ax2.set_xlabel("Frequenza in Hz", size=15)
        ax2.set_ylabel("Potenza dei coefficienti di Fourier", size=15)
        ax2.set_xlim(self.peaks[0][0] - 100, self.peaks[0][-1] + 100)
        ax2.set_title("Spettro di potenza dei coefficienti", size=15)
        ax2.legend()
        
        ax3, ax4 = g22.subplots(sharex=True, sharey=True)
        ax3.plot(self.frequencies, self.coefficients[:len(self.frequencies)].real, color="navy", label="parte reale trasformata")
        ax4.plot(self.frequencies, self.coefficients[:len(self.frequencies)].imag, color="orange", label="parte immaginaria trasformata")
        #ax3.vlines(self.peaks[0], ymin=-3000, ymax=3000, colors="orange", alpha=.5)
        #ax4.vlines(self.peaks[0], ymin=-3000, ymax=3000, colors="navy", alpha=.5)
        ax3.set_xlabel("Frequenza in Hz", size=15)
        ax3.set_ylabel("Ampiezza", size=15)
        ax4.set_xlabel("Frequenza in Hz", size=15)
        ax4.set_ylabel("Ampiezza", size=15)
        ax3.set_title("Spettro ampiezza dei coefficienti", size=15)
        ax3.set_xlim(self.peaks[0][0] - 100, self.peaks[0][-1] + 100)
        ax4.set_xlim(self.peaks[0][0] - 100, self.peaks[0][-1] + 100)
        #ax3.set_xscale('log')
        #ax4.set_xscale('log')
        ax3.legend()
        ax4.legend()
    
        plt.tight_layout()
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
        
    
    def peak_synthesis(self, n: int, file: str = None, first: bool = False, last: bool = False, c:str = None):
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
        ys = irfft(cofs)
        plt.plot(self.audio, alpha=.3, label="audio originale", c=c)
        plt.plot(ys, alpha=.3, label="audio sintetizzato")
        plt.title(f"Sintesi {file} con ", size=15)
        plt.xlabel("N° campione", size=15)
        plt.ylabel("Ampiezza", size=15)
        plt.show()

    
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
    
    def DIY_synthesis(self, n: int, file: str = None, c:str=None, zoom:int = None):
        if self.note.empty:
            self.find_notes()
            
            
        newcf = fft.fft(self.audio)
        N = len(self.audio)
        inds = np.arange(N)
        
        if file == "data1":
            pk = self.peaks[1][1]
        elif file == "data2":
            pk = self.peaks[1][:-1]
        elif file == "data3":
            pk = self.peaks[1][1:]
        
        mask = np.isin(np.absolute(newcf)**2, pk)
        false_indices = np.where(mask)[0]
        for idx in false_indices:
            start = max(0, idx - n)
            end = min(len(mask), idx + n + 1)
            mask[start:end] = True

        cofs = newcf[mask].copy()
        suminds = inds[mask].copy()
        
        yys = np.zeros(N, dtype=complex)
        yys[mask] = cofs
        yys = fft.ifft(yys)
        ylines = np.max(yys), np.min(yys)
        print("start")
        nums = []
        chunk_size = N // 4 # 4 core
        chunks = []
        for i in range(4):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < 4 - 1 else len(inds)
            chunks.append((inds[start:end], suminds, cofs, N))
        print("chunks")
            
        with multiprocessing.Pool(processes=4) as pool:
            nums = pool.starmap(DIY_ifft, chunks)
        nums = np.concatenate(nums)
        #nums = DIY_ifft(inds, suminds, cofs, N)
        print("done")
        
        zms = 20
        if zoom != None:
            zms *= zoom
        
        fig = plt.figure()
        gs = fig.add_gridspec(2,1, hspace=0)
        ax1, ax2 = gs.subplots(sharex=True, sharey=True)
        
        ax1.plot(self.timer, self.audio, alpha=.3, label="audio originale", c=c)
        ax1.plot(self.timer, nums, alpha=.6, label="sintesi seni/coseni")
        ax1.legend(loc="upper left")
        
        ax2.plot(self.timer, self.audio, alpha=.3, label="audio originale", c=c)
        ax2.plot(self.timer, yys, alpha=.6, label="sintesi ifft")
        ax2.legend(loc="upper left")
        
        ax1.hlines(ylines, xmin=0, xmax=.05, linestyles="dashdot", color="gray", alpha=.3)
        ax2.hlines(ylines, xmin=0, xmax=.05, linestyles="dashdot", color="gray", alpha=.3)
        
        ax1.set_xlim(0,1/zms)
        ax2.set_xlim(0,1/zms)

        fig.suptitle("Sintesi seni/coseni (sopra) e ifft (sotto)", size=15)
        fig.supxlabel("Tempo in secondi (fino a un $20^o$ di segnale)", size=15)
        fig.supylabel("Ampiezza", size=15)

        plt.show()
        
    
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