import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy.fft import rfft, rfftfreq, irfft
import argparse

def parser():
    parse = argparse.ArgumentParser(usage="python3 waveform.py [file.wav]", 
                                    description="dai su è ovvio")
    parse.add_argument("file", type=str, help="è il file da metterci")
    return parse.parse_args()


def main():
    file = parser()
    if file == None:
        print("Non è stato immesso nessun file audio")
    
    audio_data, sample_rate = sf.read(file.file)
    audio_data = audio_data.transpose()[0]
    print(audio_data)

    plt.plot(audio_data)
    plt.show()

    sd.play(audio_data, sample_rate)
    sd.wait()
    
    cff = rfft(audio_data, axis=0)
    frq = rfftfreq(len(cff)) / 2
    fig, axs = plt.subplots(1,2, figsize=(15,5))
    axs[0].plot(frq, np.absolute(cff[:len(frq)])**2)
    axs[0].set_yscale("log")
    axs[1].plot(frq, cff[:len(frq)].real)
    axs[1].plot(frq, cff[:len(frq)].imag)
    plt.show()
    

if __name__ == "__main__":
    main()