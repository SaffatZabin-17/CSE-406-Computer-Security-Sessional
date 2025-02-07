# Import socket module
import socket			
#import diffie_hellman_1805022 as dh
import _1905060_f2 as f2
import _1905060_f1 as f1
#import aes_1805022 as aes

# P-256 Curve Parameters
p = 2**256 - 2**224 + 2**192 + 2**96 - 1
a = -3
b = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B
Gx = 0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296
Gy = 0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5
n = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551

AES_LEN = 128
# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)		

# Define the port on which you want to connect
port = 12345			

# connect to the server on local computer
s.connect(('127.0.0.1', port))

private_key_B = f2.generate_private_key()

public_key_B = f2.generate_public_key(private_key_B)

public_key_msg = str(public_key_B[0]) + "," + str(public_key_B[1])

public_key_A = s.recv(1024).decode()

s.send(public_key_msg.encode())

# receive data from the server and decoding to get the string.
#public_key_A = s.recv(1024).decode()
#[p,g,A] = msg.split(",")

A_x = public_key_A.split(",")[0]
A_y = public_key_A.split(",")[1]

shared_key = f2.ecdh_key_exchange(private_key_B, (int(A_x), int(A_y)))

print("Shared Key: ", shared_key)

key_bin = int(bin(shared_key)[2:], 2)

key_bin = key_bin % (1 << 128)

key_bin = bin(key_bin)[2:]

IV = s.recv(1024).decode()

encrypted_text = s.recv(1024).decode()

[decrypted_hex,decrypted_text,decryption_time,key_scheduling_time_decrypt] = f1.DECRYPT_AES(encrypted_text,key_bin, IV)

print("Decrypted Text: ",decrypted_text)

s.close()

	
