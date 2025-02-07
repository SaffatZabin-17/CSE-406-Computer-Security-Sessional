# first of all import the socket library
import socket			
#import diffie_hellman_1805022 as dh
#import aes_1805022 as aes
import _1905060_f2 as f2
import _1905060_f1 as f1
import secrets

# P-256 Curve Parameters
p = 2**256 - 2**224 + 2**192 + 2**96 - 1
a = -3
b = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B
Gx = 0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296
Gy = 0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5
n = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551

AES_LEN = 128

# next create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)		
print ("Socket successfully created")

# reserve a port on your computer in our
# case it is 12345 but it can be anything
port = 12345			

# Next bind to the port
# we have not typed any ip in the ip field
# instead we have inputted an empty string
# this makes the server listen to requests
# coming from other computers on the network
s.bind(('', port))		
print ("socket binded to %s" %(port))

# put the socket into listening mode
s.listen()	
print ("socket is listening")		

# a forever loop until we interrupt it or
# an error occurs


while True:

# Establish connection with client.
    c, addr = s.accept()	
    print ('Got connection from', addr )
    print(c)

    private_key_A = f2.generate_private_key()

    public_key_A = f2.generate_public_key(private_key_A)

    public_key_msg = str(public_key_A[0]) + "," + str(public_key_A[1])
    #c.send(msg.encode())
    c.send(public_key_msg.encode())
    
    public_key_B = c.recv(1024).decode()



    B_x = int(public_key_B.split(",")[0])
    B_y = int(public_key_B.split(",")[1])
   
    shared_key = f2.ecdh_key_exchange(private_key_A, (B_x, B_y))
    
    print("s = ",shared_key)
    
    key_bin = int(bin(shared_key)[2:], 2)

    key_bin = key_bin % (1 << 128)

    key_bin = bin(key_bin)[2:]

    text = "Two One Nine Two Three Four Five Six"

    IV = secrets.randbits(128)

    print("IV = ",IV)

    c.send(str(IV).encode())

    [encrypted_hex,encrypted_text,encryption_time,key_scheduling_time_encrypt] = f1.ENCRYPT_AES(text,key_bin, IV)

    c.send(encrypted_text.encode())

    # Close the connection with the client
    c.close()

    # Breaking once connection closed
    break

