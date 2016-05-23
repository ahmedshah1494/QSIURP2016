import socket
import os

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
    r = c.recv(8192)
    while r:
        data += r
        r = c.recv(8192)
    # print data
    [headers,body] = data.split('\n\n')
    # print headers, len(body)
    [filename, size] = headers.split('\n')
    filename = filter(lambda x : ord(x) >= 33 and ord(x) <= 126, filename)
    size = filter(lambda x : ord(x) >= 33 and ord(x) <= 126, size)
    print filename
    print size
    size = eval(size.strip())

    path = filename[:len(filename) - len(filename.split('/')[-1])]
    if not os.path.exists(path):
        print path
        os.makedirs(path)
    # (bytes, addr) = c.recvfrom(size - len(body))
    # print data
    body = body[:size]
    print "size recv = " + str(len(body))
    f = open("files/"+filename,'w')
    f.write(body)
    f.close()
    print filename +' saved'
    c.close()
