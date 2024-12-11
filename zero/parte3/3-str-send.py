import socket

#definizione oggetto socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('localhost', 42069))
#invio "Hello World!"
s.sendall("Hello World!".encode())
print("Risposta del server: ", s.recv(1024).decode())
#chiusura socket
s.close()