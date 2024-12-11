import numpy as np
import soundcard as sc

#parametri per la registrazione
seconds = 11
samplerate = 48000
numframes = 1024
blocksize = 4096
iterator = range(0, (samplerate * seconds) // numframes)

#definizione oggetti soundcard
mic = sc.default_microphone()
spk = sc.default_speaker()

#registrazione
array = []
with mic.recorder(samplerate=samplerate, channels=2, blocksize=blocksize) as rec:
    for i in iterator:
        data = rec.record(numframes=numframes)
        array.append(data)

#concatenazione array
audio = np.concatenate(array)
print(audio)
print(audio.size)

#ascolto
with spk.player(samplerate=samplerate, channels=2, blocksize=blocksize) as player:
    player.play(audio / np.max(np.abs(audio)))