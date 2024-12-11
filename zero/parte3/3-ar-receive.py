import socket
import numpy as np

# la funzione decodifica dividendo il messaggio prendendo
# la forma dell'array, il tipo di dato che porta e infine
# ricostruisce l'array
def unprotocol_inator(bin):
    tp, shp, dt = bin.split(b'D')
    shp = [i for i in shp]
    dt = np.frombuffer(dt, dtype=tp.decode()).reshape(shp)
    return dt

# Definizione del socket e binding
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('localhost', 65432))

# ricezione dei dati
s.listen()
conn, addr = s.accept()
print("Connessione stabilita con {:}".format(addr))

data = b''
try:
    # il loop garantisce la trasmissione dei dati
    # senza saperne la dimensione
    while True:
        packet = conn.recv(4096)
        if b'EOT' in packet:
            data += packet.split(b'EOT')[0]
            break
        data += packet
    conn.sendall("Messaggio ricevuto!".encode())
except socket.error:
    conn.sendall("Errore".encode())

# spacchettamento, print e chiusura socket
data = unprotocol_inator(data)
print(data)
s.close()