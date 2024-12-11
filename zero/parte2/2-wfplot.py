import numpy as np
import soundcard as sc
import matplotlib.pyplot as plt

#parametri registrazione
seconds = 11
samplerate = 48000
numframes = 1024
iterator = range(0, (samplerate * seconds) // numframes)
#definizione oggetti soundcard
mic = sc.default_microphone()
spk = sc.default_speaker()
#registrazione
array = []
with mic.recorder(samplerate=samplerate, channels=2) as rec:
    for i in iterator:
        data = rec.record(numframes=numframes)
        array.append(data)
#costruzione array numpy
audio = np.concatenate(array)
print(audio)
print(audio.size)
print(len(audio))
print(audio.shape)
#ri-ascolto della registrazione
with spk.player(samplerate=samplerate, channels=2) as player:
    player.play(audio / np.max(np.abs(audio)))

#asse delle ascisse del grafico
x = range(len(audio))
#costruzione grafico e visualizzazione
plt.rcParams["figure.figsize"] = [12, 7.5]
plt.rcParams["figure.autolayout"] = True

plt.plot(x, audio, color="blue")
plt.title("Random graph")

plt.show()