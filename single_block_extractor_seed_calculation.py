from math import ceil, log2, e
from sympy import isprime

def get_seed_length(input_length, error_prob, target_bits=512):
    # the following part is according to Eq. (S29) of arXiv:1803.06219 with the
    # replacement of epsilon there by extractor_error^2/2 here.
    t=target_bits
    w=2*ceil(log2(4*input_length*t**2) + log2(4 / error_prob))
    w=w+1
    while not isprime(w):
        w=w+2
    return int(w**2 * max(2, 1+ceil((log2(t-e)-log2(w-e))/(log2(e)-log2(e-1)))))


input_length = 11924208
error_prob = 1.147943701974890e-43 * 512
#error_prob = 0.2 * 2**-64
print(get_seed_length(input_length, error_prob))
#Should be 398161
