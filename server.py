import socket


TCP_IP = '192.168.0.103'
TCP_PORT = 5005
BUFFER_SIZE = 20  # Normally 1024, but we want fast response

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)

conn, addr = s.accept()
print ('Connection address:', addr)
data = conn.recv(BUFFER_SIZE)
data=data.decode()
print (data)
#conn.send(data)  # echo
conn.close()
