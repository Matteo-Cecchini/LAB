import socket
import random as rd
import numpy as np

# il protocollo divide gli argomenti con D
# e dichiara la fine trasmissione con EOT
def protocol(arr):
    bin = str(arr.dtype).encode() + b'D'
    print(arr.shape)
    for i in arr.shape:
        bin += i.to_bytes(1)
    bin += b'D' + arr.tobytes()
    bin += b'EOT'
    return bin

# definizione socket e connessione
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('localhost', 65432))

# array da inviare
arr_to_send = np.array(rd.choices(range(10), k=10))
print(arr_to_send)

# invio array
data_to_send = protocol(arr_to_send)
s.sendall(data_to_send)

# ricezione esito trasmissione e chiusura socket
print(s.recv(1024))
s.close()