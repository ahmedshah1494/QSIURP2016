import socket
import os
import threading
import os

clients = []
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

def task():
	global end
	while True:
		t = raw_input("press Enter to start recording")
		if "exit" in t:
			print "in if"
			
			for c in clients:
				c.close
			s.close()
			os._exit(0)
			return	
		for c in clients:
			try:
				c.send("#")
			except:
				clients.remove(c)
				c.close()

def pingTask():
	while True:
		print len(clients), "connected"
		for c in clients:
			r = c.recv(1);
			print "received ", r
			if r == ".":
				try:
					c.send(".")
				except:
					clients.remove(c)
					c.close()


# T2 = threading.Thread(target=pingTask)
# T2.start()

T1 = threading.Thread(target=task)
T1.start()

host = '0.0.0.0'
print host
port = 9998
s.bind((host,port))
s.listen(5)
recvCount = 0
while True:
    c, addr = s.accept()
    clients.append(c)
    print 'Got connection from', addr
