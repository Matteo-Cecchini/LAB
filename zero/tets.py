import numpy as np
import soundcard as sc
import struct

a = sc.default_microphone().record(numframes=240000, samplerate=48000)

sc.default_speaker().play(a/np.max(np.abs(a)), samplerate=48000)
