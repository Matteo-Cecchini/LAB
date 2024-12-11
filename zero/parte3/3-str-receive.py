import socket

#definizione oggetto socket e binding
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('localhost', 42069))
#ascolto e connessione
s.listen()
conn, addr = s.accept()
print("Connessione stabilita con {:}".format(addr))
#ricezione "Hello World!"
try:
    msg = conn.recv(1024).decode()
    print(msg)
    conn.sendall("Messaggio ricevuto!".encode())
except socket.error:
    conn.sendall("Errore".encode())
#chiusura socket
s.close()