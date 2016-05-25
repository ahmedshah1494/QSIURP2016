import socket
import os
import threading

clients = []

def task():
	while True:
		t = raw_input("press Enter to start recording")
		for c in clients:
			c.send("#")

T = threading.Thread(target=task)
T.start()

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
host = '0.0.0.0'
print host
port = 9999
s.bind((host,port))
s.listen(5)
recvCount = 0
while True:
    c, addr = s.accept()
    clients.append(c)
    print 'Got connection from', addr