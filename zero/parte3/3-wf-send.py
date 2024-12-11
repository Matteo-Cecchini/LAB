import socket
import struct
import numpy as np
import soundcard as sc
import matplotlib.pyplot as plt

#parametri di registrazione
seconds = 11
samplerate = 48000
numframes = 1024
iterator = range(0, (samplerate * seconds) // numframes)

# funzione di protocollo
def protocol(arr):
    bin = str(arr.dtype).encode() + b'DIV'
    bin += len(arr.shape).to_bytes(1) + b'DIV'
    bin += struct.pack('i'*len(arr.shape), *arr.shape)
    bin += b'DIV' + arr.tobytes()
    bin += b'EOT'   # End Of Transmission
    return bin

mic = sc.default_microphone()
input() # recording all'invio

print("Recording...")
# soundcard tratta dati float32
array = []
with mic.recorder(samplerate=samplerate, channels=2) as rec:
    for i in iterator:
        data = rec.record(numframes=numframes)
        array.append(data)
print("End")

# packing dati per la trasmissione
data = np.concatenate(array)
stream = protocol(data)
# connessione, invio dati e ricezione esito
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    input() #connessione all'invio
    s.connect(('localhost', 42069))
    s.sendall(stream)
    # esito
    msg = s.recv(1024).decode()
    print("Risposta del server: ", msg)
    # chiusura socket
    s.close()

# check invio corretto
x = range(len(data))
plt.rcParams["figure.figsize"] = [12, 7.5]
plt.rcParams["figure.autolayout"] = True

plt.plot(x, data, color="blue")
plt.title("Grafico inviato")

plt.show()