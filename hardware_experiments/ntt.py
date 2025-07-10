import math
import numpy as np

def ntt(a, q, omegas):
    n = len(a)
    out = [0] * n
    
    for i in range(n):
        for j in range(n):
            out[i] = (out[i] + a[j] * omegas[(i * j) % n]) % q
    return out

def reverse_bits(number, bit_length):
    # Reverses the bits of `number` up to `bit_length`.
    reversed = 0
    for i in range(0, bit_length):
        if (number >> i) & 1: 
            reversed |= 1 << (bit_length - 1 - i)
    return reversed

def bit_rev_shuffle(input):
    
    out = input.copy()
    ntt_len = len(input)
    for i in range(ntt_len):
        rev_i = reverse_bits(i, ntt_len.bit_length() - 1)
        if rev_i > i:
            out[i]     = input[rev_i]
            out[rev_i] = input[i]
    return out

def ntt_dif_nr(a, q, omegas):
    n = len(a)
    out = a.copy()
    
    log2n = math.log2(n)

    # The length of the input array `a` should be a power of 2.
    assert log2n.is_integer()

    M = n
    for _ in range(int(log2n)):
        M >>= 1
        g = 0
        for k in range(M):
            for j in range(0, n, 2*M):
                U = out[j + k]
                V = out[j + k + M]
                
                out[j + k] = (U + V) % q
                out[j + k + M] = ((U - V) * omegas[g]) % q
                # print("g =", g)
                # print("upper =", j + k, "lower =", j + k + M)

            g += n // (2*M)

    return out

def ntt_dit_rn(a, q, omegas):
    n = len(a)
    out = a.copy()
    
    log2n = math.log2(n)

    # The length of the input array `a` should be a power of 2.
    assert log2n.is_integer()
    
    iterations = int(log2n)
    # iterations = 2
    M = 2
    for p in range(iterations):

        for i in range(0, n, M):

            g = 0
            for j in range(0, M >> 1):
                k = i + j + (M >> 1)
                U = out[i + j]
                V = out[k] * omegas[g]
                
                out[i + j] = (U + V) % q
                out[k] = (U - V) % q
                
                # print("g =", g)
                # print("upper =", i + j, "lower =", k)

                g = g + n // M
                
                
        M <<= 1

    return out

