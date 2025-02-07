from lib2to3.pytree import convert
from BitVector import *
import time
import secrets

Sbox = (
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
)

InvSbox = (
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D,
)

Mixer = [
    [BitVector(hexstring="02"), BitVector(hexstring="03"), BitVector(hexstring="01"), BitVector(hexstring="01")],
    [BitVector(hexstring="01"), BitVector(hexstring="02"), BitVector(hexstring="03"), BitVector(hexstring="01")],
    [BitVector(hexstring="01"), BitVector(hexstring="01"), BitVector(hexstring="02"), BitVector(hexstring="03")],
    [BitVector(hexstring="03"), BitVector(hexstring="01"), BitVector(hexstring="01"), BitVector(hexstring="02")]
]

InvMixer = [
    [BitVector(hexstring="0E"), BitVector(hexstring="0B"), BitVector(hexstring="0D"), BitVector(hexstring="09")],
    [BitVector(hexstring="09"), BitVector(hexstring="0E"), BitVector(hexstring="0B"), BitVector(hexstring="0D")],
    [BitVector(hexstring="0D"), BitVector(hexstring="09"), BitVector(hexstring="0E"), BitVector(hexstring="0B")],
    [BitVector(hexstring="0B"), BitVector(hexstring="0D"), BitVector(hexstring="09"), BitVector(hexstring="0E")]
]

round_constants = [
    BitVector(hexstring="01"),
    BitVector(hexstring="02"),
    BitVector(hexstring="04"),
    BitVector(hexstring="08"),
    BitVector(hexstring="10"),
    BitVector(hexstring="20"),
    BitVector(hexstring="40"),
    BitVector(hexstring="80"),
    BitVector(hexstring="1B"),
    BitVector(hexstring="36")
]

AES_MODULUS = BitVector(bitstring='100011011')

def xor_ascii_strings(str1, str2):
    # Ensure both strings are of the same length
    # XOR each pair of corresponding ASCII values
    xor_result = [chr(ord(char1) ^ ord(char2)) for char1, char2 in zip(str1, str2)]

    # Convert the list of characters back to a string
    result_string = ''.join(xor_result)

    return result_string

def xor_string_and_integer(str_val, int_val):
    # Convert the string to an integer
    str_val = convert_to_binary(str_val)

    str_int = int(str_val, 2)

    # Perform XOR
    result = str_int ^ int_val

    # Convert the result back to a binary string
    result_str = bin(result)[2:].zfill(128)

    return result_str

def print_array(array):
    for i in range(len(array)):
        print(array[i] + " ", end="")
    print()

# convert ascii to binary string
def convert_to_binary(text):
    binary_string = ""
    i = 0
    while i < len(text):
        binary_string += bin(ord(text[i]))[2:].zfill(8)
        i += 1
    return binary_string

#convert to Hex String 
def convert_to_hex_string(text):
    hex_string = ""
    index = 0

    while index < len(text):
        bit = hex(ord(text[index]))[2:]
        if len(bit) == 1:
            bit = "0" + bit
        hex_string += bit
        index += 1
    
    # merge 4 byte into 1 word
    hex_list = []
    index = 0
    while index < len(hex_string):
        hex_pair = hex_string[index:index+2]
        hex_list.append(hex_pair)
        index += 2
    #hex_string = [hex_string[i:i+2] for i in range(0,len(hex_string),2)]
    return hex_list

# convert to hex string from binary string
def convert_to_hex_array_from_binary(binary_string):
    hex_string = []
    index = 0
    while index < len(binary_string):
        binary_chunk = binary_string[index:index + 8]
        hex_value = hex(int(binary_chunk, 2))[2:]
        hex_string.append(hex_value)
        index += 8
    return hex_string

# convert to ascii from hex matrix
def convert_to_ascii(matrix):
    ascii_string = ""
    i = 0
    while i < len(matrix):
        j = 0
        while j < len(matrix[i]):
            bitvector_hex = matrix[i][j].get_bitvector_in_hex()
            decimal_value = int(bitvector_hex, 16)
            ascii_string += chr(decimal_value)
            j += 1
        i += 1
    return ascii_string

def flatten_matrix(matrix):
    flat_string = ""
    outer_index = 0
    matrix_len = len(matrix)
    while outer_index < matrix_len:
        inner_index = 0
        inner_len = len(matrix[outer_index])
        while inner_index < inner_len:
            flat_string += (matrix[outer_index][inner_index].get_bitvector_in_hex() + " ")
            inner_index += 1
        outer_index += 1

    return flat_string

def truncate_key(key,key_len):
    truncated_key = []
    for i in range(min(len(key), key_len)):
        truncated_key.append(key[i])

    return truncated_key

# trim or pad key to make it 128 bits
def resize_key(key):
    key_len = 128
    if key_len >= len(key):
        key = key + "0"*(key_len-len(key))
    else:
        key = truncate_key(key,key_len)
    return convert_to_hex_array_from_binary(key)

#convert array to 2D array and then transpose it
def convert_to_2D_array(array):
    array = [array[i:i+4] for i in range(0,len(array),4)]
    # convert to bitvector
    new_array = []
    for j in range(len(array[0])):
        temp_column = []
        for i in range(len(array)):
            temp_column.append(BitVector(hexstring=array[i][j]))
        new_array.append(temp_column)
    array = new_array
    array = transpose_matrix(array)
    return array

def substitute_bytes_list(list,inv):
    i = 0
    result_list = []
    if not inv:
        while i < len(list):
            result_list.append(BitVector(intVal=Sbox[list[i].intValue()], size=8))
            i += 1
    else:
        while i < len(list):
            result_list.append(BitVector(intVal=InvSbox[list[i].intValue()], size=8))
            i += 1
    return result_list


def substitute_bytes_matrix(matrix,inv):
    i = 0
    while i < len(matrix):
        matrix[i] = substitute_bytes_list(matrix[i], inv)
        i += 1
    return matrix

def func_g(word,round_constant):
    # one-byte left circular rotation
    word = left_shift_list(word,1)
    # substitute bytes
    word = substitute_bytes_list(word, False)
    # XOR with round constant
    word[0] = word[0] ^ round_constant
    return word

# XOR 2 list
def xor_list(list1,list2):
    i = 0
    result_list = []
    while i < len(list1):
        result_list.append(list1[i] ^ list2[i])
        i += 1
    return result_list

def xor_matrix(matrix1, matrix2):
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        raise ValueError("Matrices must have the same dimensions for XOR operation.")

    result_matrix = []
    for i in range(len(matrix1)):
        row = []
        for j in range(len(matrix1[0])):
            row.append(matrix1[i][j] ^ matrix2[i][j])
        result_matrix.append(row)

    return result_matrix

# Transpose a matrix
def transpose_matrix(matrix):
    # Get the number of rows and columns in the original matrix
    rows = len(matrix)
    cols = len(matrix[0])

    # Create a new matrix with swapped rows and columns
    transposed_matrix = []

    i = 0
    while i < cols:
        j = 0
        new_row = []
        while j < rows:
            new_row.append(matrix[j][i])
            j += 1
        transposed_matrix.append(new_row)
        i += 1

    return transposed_matrix

# AES key scheduling
def key_scheduling(key):
    all_keys = []
    key_round_0 = convert_to_2D_array(key)
    all_keys.append(key_round_0)
    column_length = len(key_round_0)
    round_constant = BitVector(hexstring="01")
    round = 1
    while round <= 10:
        key_new_round = []
        list1 = all_keys[round - 1][0]
        list2 = func_g(all_keys[round - 1][column_length - 1], round_constant)
        list3 = xor_list(list1, list2)
        key_new_round.append(list3)
        i = 1
        while i < len(all_keys[round - 1]):
            list1 = all_keys[round - 1][i]
            list2 = key_new_round[i - 1]
            list3 = xor_list(list1, list2)
            key_new_round.append(list3)
            i += 1
        all_keys.append(key_new_round)
        round_constant = round_constant.gf_multiply_modular(BitVector(hexstring="02"), AES_MODULUS, 8)
        round += 1

    # transpose all_keys
    k = 0
    while k < len(all_keys):
        all_keys[k] = transpose_matrix(all_keys[k])
        k += 1
   
    return all_keys

def mix_columns(matrix,inv):
    if not inv:
        mixer = Mixer
    else:
        mixer = InvMixer
    ret = []
    i = 0
    while(i < len(matrix)):
        ret.append([BitVector(intVal=0, size=8)] * len(matrix[0]))
        i += 1
    i = 0
    while i < len(matrix):
        j = 0
        while j < len(matrix[0]):
            k = 0
            while k < len(matrix):
                ret[i][j] = ret[i][j] ^ mixer[i][k].gf_multiply_modular(matrix[k][j], AES_MODULUS, 8)
                k += 1
            j += 1
        i += 1
    return ret

def left_shift_list(list,shift):
    shifted_list = []
    shifted_list = list[shift:] + list[:shift]
    return shifted_list

def left_shift_matrix(matrix):
    i = 0
    while i < len(matrix):
        matrix[i] = left_shift_list(matrix[i], i)
        i += 1
    return matrix

def right_shift_list(list,shift):
    shifted_list = []
    shifted_list = list[-shift:] + list[:-shift]
    return shifted_list

def right_shift_matrix(matrix):
    i = 0
    while i < len(matrix):
        matrix[i] = right_shift_list(matrix[i],i)
        i += 1
    return matrix


def AES_ENCRYPT_ROUNDS(state,keys):

    state = xor_matrix(state,keys[0])

    round = 1
    while round <= 10:
        # substitute bytes
        state = substitute_bytes_matrix(state, False)
        # shift rows
        state = left_shift_matrix(state)
        # mix columns
        if round != 10:
            state = mix_columns(state, False)
        # add round key
        state = xor_matrix(state, keys[round])
        round += 1

    return state

def ENCRYPT_AES(text,key, IV):
    # key might not be of the required length, so resize it
    key = resize_key(key)
    # AES key scheduling
    temp_time = time.time()
    key_schedule = key_scheduling(key)
    key_scheduling_time = time.time() - temp_time

    while(len(text)%(128//8) != 0 ):
        text += " "
    
    encrypted_text_list = ""
    encrypted_hex_list = ""
    encrypted_texts = []

    start_time = time.time()

    for i in range(0,len(text),128//8):
        split_text = text[i:i+128//8]
        #print(len(split_text))
        temp_text = convert_to_binary(split_text)
        #print(len(temp_text))
        
        if i == 0:
            temp_text = xor_string_and_integer(temp_text,IV)
        else:
            temp_text = xor_ascii_strings(temp_text,convert_to_binary(encrypted_texts[i//16 - 1]))
            #print(len(temp_text))
            #print(len(convert_to_binary(encrypted_texts[i//16 - 1])))
        
        #print(len(temp_text))
        #print(temp_text)
        #print(len(split_text))
        split_text = convert_to_2D_array(convert_to_hex_string(split_text))
        split_text = transpose_matrix(split_text)
        encrypted = AES_ENCRYPT_ROUNDS(split_text,key_schedule)
        encrypted = transpose_matrix(encrypted)
        encrypted_text = convert_to_ascii(encrypted)
        encrypted_texts += [encrypted_text]
        encrypted_hex = flatten_matrix(encrypted)
        encrypted_text_list+=(encrypted_text)
        encrypted_hex_list+=(encrypted_hex)
    
    encryption_time = time.time() - start_time
    
    return encrypted_hex_list,encrypted_text_list,encryption_time,key_scheduling_time


def AES_DECRYPT_ROUNDS(state,keys):

    state = xor_matrix(state,keys[0])
    
    round = 1
    while round <= 10:
        # shift rows
        state = right_shift_matrix(state)
        # substitute bytes
        state = substitute_bytes_matrix(state, True)
        # add round key
        state = xor_matrix(state, keys[round])
        # mix columns
        if round != 10:
            state = mix_columns(state, True)
        round += 1

        
    return state

def DECRYPT_AES(text,key, IV):
    # key might not be of the required length, so resize it
    
    key = resize_key(key)
    # AES key scheduling
    temp_time = time.time()
    key_schedule = key_scheduling(key)
    key_scheduling_time = time.time() - temp_time
    key_schedule = key_schedule[::-1]
  
    start_time = time.time()
    decrypted_text = ""
    decrypted_hex = ""
    cipher_texts = []

    for i in range(0,len(text),128//8):
        split_text = text[i:i+128//8]
        cipher_texts += [split_text]
        split_text = convert_to_2D_array( convert_to_hex_string(split_text) )
        split_text = transpose_matrix(split_text)
        decrypted = AES_DECRYPT_ROUNDS(split_text,key_schedule)
        decrypted = transpose_matrix(decrypted)
        decrypted_text += convert_to_ascii(decrypted)

        temp_text = convert_to_binary(decrypted_text)
        if i == 0:
            temp_text = xor_string_and_integer(temp_text,IV)
        else:
            temp_text = xor_ascii_strings(temp_text,convert_to_binary(cipher_texts[i//16 - 1]))
        decrypted_hex += flatten_matrix(decrypted)

    decryption_time = time.time() - start_time
    
    return decrypted_hex,decrypted_text,decryption_time,key_scheduling_time

if __name__ == "__main__":
    key = "Thats my Kung Fu"
    text = "Two One Nine Two Three Four Five Six"

    text_hex = convert_to_hex_string(text)

    key_hex = convert_to_hex_string(key)

    key_bin = convert_to_binary(key)

    #Generate a 128-bit IV
    IV = secrets.randbits(128)
    print("IV: ",IV)
   
    [encrypted_hex,encrypted_text,encryption_time,key_scheduling_time_encrypt] = ENCRYPT_AES(text,key_bin, IV)
    [decrypted_hex,decrypted_text,decryption_time,key_scheduling_time_decrypt] = DECRYPT_AES(encrypted_text,key_bin, IV)
    
    key_scheduling_time = (key_scheduling_time_encrypt + key_scheduling_time_decrypt)/2

    print("Key:")
    print("In ASCII: ",key)
    print("In HEX: ",end="")
    print_array(key_hex)
    print()
    print("Plain Text:")
    print("In ASCII: ",text)
    print("In HEX: ", end="")
    print_array(text_hex)
    print()
    print("Ciphered Text:")
    print("In HEX: ",encrypted_hex)
    print("In ASCII: ",encrypted_text)
    print()
    print("Deciphered Text:")
    print("In HEX: ",decrypted_hex)
    print("In ASCII: ",decrypted_text)
    print()
    print("Execution time details")
    print("Key Scheduling: ",key_scheduling_time*1000," ms")
    print("Encryption time: ",encryption_time*1000," ms")
    print("Decryption time: ",decryption_time*1000," ms")


