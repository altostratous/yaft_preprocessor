import math
import vbcode
import bitstring

GAMMA = 'gamma'
VARIABLE_BYTE = 'varbyte'

COMPRESSION_TYPES = (
    GAMMA,
    VARIABLE_BYTE,
)

MAGIC = '01111110'


def compress_using_gamma(integers: list):
    bits = bitstring.BitArray()
    for integer in integers:
        bit_length = int(math.log(integer, 2))
        for _ in range(bit_length):
            bits.append('0b1')
        bits.append('0b0')
        integer_bits = []
        while integer > 0:
            integer_bits.append(integer % 2)
            integer //= 2
        for i in list(reversed(integer_bits))[1:]:
            bits.append('0b1' if i == 1 else '0b0')
    binary = bits.bin
    binary = '0b' + ((8 - (len(binary) % 8)) * '0') + MAGIC + binary
    return bitstring.BitArray(binary).hex


def decompress_using_gamma(compressed_value: str):
    bits = bitstring.BitArray(hex=compressed_value).bin
    magic_index = bits.index(MAGIC)
    bits = bits[magic_index + 8:]
    integers = []
    while bits:
        integer_length = bits.index('0')
        bits = bits[integer_length + 1:]
        integers.append(int('1' + bits[:integer_length], 2))
        bits = bits[integer_length:]
    return integers


def compress_using_variable_byte(integers: list):
    return vbcode.encode(integers).hex()


def decompress_using_variable_byte(compressed_value: str):
    return vbcode.decode(bytes.fromhex(compressed_value))


COMPRESSORS = {
    GAMMA: compress_using_gamma,
    VARIABLE_BYTE: compress_using_variable_byte,
}

DECOMPRESSORS = {
    GAMMA: decompress_using_gamma,
    VARIABLE_BYTE: decompress_using_variable_byte,
}


def integers_to_gaps(integers: list):
    if not integers:
        raise ValueError('integers must be of positive length.')
    return [
        integers[0], *[
            integers[i] - integers[i - 1] for i in range(1, len(integers))
        ]
    ]


def gaps_to_integers(gaps: list):
    if not gaps:
        raise ValueError('gaps must be of positive length.')
    result = [gaps[0]]
    for i in range(1, len(gaps)):
        result.append(gaps[i] + result[-1])
    return result


def compress(integers: list, compression_type):
    return COMPRESSORS[compression_type](integers_to_gaps(integers))


def decompress(compressed_value: str, compression_type):
    return gaps_to_integers(DECOMPRESSORS[compression_type](compressed_value))


def decompress_values(compressed_values_dict, compression_type):
    return {
        key: [i - 1 for i in decompress(compressed_value, compression_type)] for key, compressed_value in compressed_values_dict.items()
    }


def compress_lists(integers_dict, compression_type):
    return {
        key: compress([i + 1 for i in integer_list], compression_type) for key, integer_list in integers_dict.items()
    }
