import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
host = '0.0.0.0'
print host
port = 9999
s.bind((host,port))
s.listen(5)
while True:
    c, addr = s.accept()
    print 'Got connection from', addr
    data = c.recv(512)

    [headers,body] = data.split('\n\n')
    [filename, size] = headers.split('\n')
    size = int(size)
    while (len(body) < size):
    	body += c.recv(1)
    f = open("files/"+filename,'w')
    f.write(body)
    f.close()
    print filename +' saved'
    c.close()
