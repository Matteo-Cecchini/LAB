import socket
import struct
import numpy as np
import soundcard as sc
import matplotlib.pyplot as plt

# funzione di spacchettamento
def unprotocol_inator(bin):
    print(len(bin.split(b'DIV')))
    tp, ln, shp, dt = bin.split(b'DIV')
    ln = int.from_bytes(ln)
    shp = struct.unpack('i'*ln, shp)
    print(shp)
    dt = np.frombuffer(dt, dtype=tp.decode()).reshape(shp)
    return dt

# funzione loop di ricezione dati
def stream_loop(c):
    s = b''
    while True:
        p = c.recv(1024)
        if b'EOT' in p:
            s += p.split(b'EOT')[0]
            break
        s += p
    return s

# binding, ricezione e chiusura socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #binding
    s.bind(('localhost', 42069))
    s.listen()
    c, a = s.accept()
    print(f"Connessione stabilita con {a}")
    #ricezione
    try:
        stream = stream_loop(c)
        c.sendall("Messaggio ricevuto!".encode())
    except socket.error:
        print("Errore")
    #chiusura
    s.close()

#spacchettamento
data = unprotocol_inator(stream)
with sc.default_speaker().player(samplerate=48000) as player:
    player.play(data/np.max(np.abs(data)))
    
#grafico waveform
x = range(len(data))
plt.rcParams["figure.figsize"] = [12, 7.5]
plt.rcParams["figure.autolayout"] = True

plt.plot(x, data, color="red")
plt.title("Grafico ricevuto")

plt.show()