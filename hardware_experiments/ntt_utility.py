import math
import sympy
import random
from sympy import ntheory

def closest_powers_of_two(exponent):
    # If the exponent is even, it can be exactly divided by two
    if exponent % 2 == 0:
        return 2**(exponent // 2), 2**(exponent // 2)
    # If the exponent is odd, we return the two closest powers of 2
    else:
        return 2**((exponent // 2) + 1), 2**(exponent // 2)

def find_a_modulus(n, bit_width):
    """
    Find a modulus of the form m*n+1 which is a prime.
    """
    # bit_width = 128
    max_value = pow(2, bit_width)-1
    max_m = (max_value-1)//n
    # If we make first_m too big it gets slow
    first_m = random.randint(1, 10)
    for offset in range(max_m):
        m = (first_m + offset) % (max_m) + 1
        modulus = m * n + 1
        is_prime = sympy.isprime(modulus)
        if is_prime:
            break
    return modulus


def get_root_of_unity(n, modulus):
    """
    Returns the smallest n'th root of unity for modulus `modulus`.
    """
    # We only handle `n` beging a power of two.
    assert math.log(n, 2).is_integer()
    b = (modulus - 1) // n
    primitive_root = ntheory.primitive_root(modulus)
    omega = pow(primitive_root, b) % modulus

    # `omega ^ s` is another primitive root if s and n are coprime.
    # Since `n` is a power of two all odd values of s are coprime.
    # So we keep multiplying omega by omega^2 until we loop back round to
    # omega and that will give us all the roots of unity.
    value = omega
    min_rou = omega
    omega2 = (omega * omega) % modulus
    while True:
        value = (value * omega2) % modulus
        if value < min_rou:
            min_rou = value
        if value == omega:
            break
    return min_rou

def generate_twiddle_factors(n, q, precompute=False, x=0, b=0, omega=0):
    # Produces `n` omegas (or twiddle factors) 
    # given the generator: x^b (mod q) of the 
    # prime field of q.

    # we know q is prime
    # assert isprime(q)
    if not precompute:
        x = ntheory.primitive_root(q)
        b = (q - 1) // n
        omega = (x ** b) % q

    omegas = [1]
    for i in range(n-1):
        # Multiply (mod q) by the previous value.
        omegas.append((omegas[i] * omega) % q)

    return omegas    

if __name__ == "__main__":
     n = pow(2,16)
     modulus = find_a_modulus(n)
     omega = get_root_of_unity(n, modulus)
     print(modulus, omega)