import socket
import os
from stat import *

def sendFile(filepath):
	filename = filepath.split('/')[-1]
	filesize = os.stat(filepath)[ST_SIZE]
	f = open(filepath,'r')
	fileData = f.read()
	f.close()
	data = "%s\n%s\n\n%s" % (filename,filesize,fileData)

	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
	s.connect(('makoshark.ics.cs.cmu.edu',9999))
	s.send(data)
	s.close()

sendFile('1461404088264.wav')
