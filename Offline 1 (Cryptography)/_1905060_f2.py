import random
import time

# P-256 Curve Parameters
p = 2**256 - 2**224 + 2**192 + 2**96 - 1
a = -3
b = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B
Gx = 0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296
Gy = 0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5
n = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551

# Point Addition on Elliptic Curve
def point_addition(P, Q):
    x_p, y_p = P
    x_q, y_q = Q

    if P != Q:
        m = (y_q - y_p) * pow(x_q - x_p, -1, p) % p
    else:
        m = (3 * x_p**2 + a) * pow(2 * y_p, -1, p) % p

    x_r = (m**2 - x_p - x_q) % p
    y_r = (m * (x_p - x_r) - y_p) % p

    return (x_r, y_r)

# Scalar Multiplication on Elliptic Curve
def scalar_multiply(k, P):
    result = None
    for bit in bin(k)[2:]:
        result = point_addition(result, result)
        if bit == '1':
            result = point_addition(result, P)
    return result

# Generate a random private key
def generate_private_key():
    return random.randint(1, n - 1)

# Generate the public key corresponding to a private key
def generate_public_key(private_key):
    return scalar_multiply(private_key, (Gx, Gy))

# Perform ECDH key exchange
def ecdh_key_exchange(private_key_A, public_key_B):
    shared_secret = scalar_multiply(private_key_A, public_key_B)
    print("Shared Secret: ", shared_secret)
    return shared_secret[0]  # Return x-coordinate as the shared secret

# Example: Generate shared keys with different bit lengths
bit_lengths = [128, 192, 256]

if __name__ == "__main__":
    print("Computational Time for ECDH Key Exchange (P-256 Curve):")
    for bit_length in bit_lengths:
        private_key_A = generate_private_key()
        private_key_B = generate_private_key()
    
        temp_time =time.time()
        public_key_A = generate_public_key(private_key_A)
        public_key_A_time = time.time() - temp_time

        temp_time =time.time()
        public_key_B = generate_public_key(private_key_B)
        public_key_B_time = time.time() - temp_time

        temp_time =time.time()
        shared_key_A = ecdh_key_exchange(private_key_A, public_key_B) % (1 << bit_length)
        shared_key_B = ecdh_key_exchange(private_key_B, public_key_A) % (1 << bit_length)
        assert shared_key_A == shared_key_B
        shared_key_time = (time.time() - temp_time)/2

        print("k : ",bit_length)
        print("A computation time: ", public_key_A_time*1000, " ms")
        print("B computation time: ", public_key_B_time*1000, " ms")
        print("Shared key computation time: ", shared_key_time*1000, " ms")
        print()