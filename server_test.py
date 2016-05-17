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
    data = ""
    filename = c.recv(512)
    size = c.recv(512)
    body = c.recv(int(size))

    print filename, size, len(body)
    # [headers,body] = data.split('\n\n')
    # [filename, size] = headers.split('\n')
    # size = int(size)
    # (bytes, addr) = c.recvfrom(size - len(body))
    # print data
    print "size recv = " + str(len(body))
    f = open("files/"+filename,'w')
    f.write(body)
    f.close()
    print filename +' saved'
    c.close()
