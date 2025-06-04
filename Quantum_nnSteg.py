import numpy as np
from PIL import Image
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler, Session
import zlib
import hashlib

PASSWORD = "password"

def deterministic_random_gates(password, block_index, num_qubits):
    hash_input = f"{password}-{block_index}".encode('utf-8')
    digest1 = hashlib.sha256(hash_input).digest()
    digest2 = hashlib.sha256(digest1).digest()
    bits1 = [(digest1[i // 8] >> (7 - (i % 8))) & 1 for i in range(num_qubits)]
    bits2 = [(digest2[i // 8] >> (7 - (i % 8))) & 1 for i in range(num_qubits)]
    return bits1, bits2

def generate_quantum_bits_deterministic(n, password):
    bits = []
    num_qubits = 28
    num_blocks = (n + num_qubits - 1) // num_qubits
    backend = AerSimulator()
    circuits = []
    for block_index in range(num_blocks):
        bits_h, bits_x = deterministic_random_gates(password, block_index, num_qubits)
        qc = QuantumCircuit(num_qubits, num_qubits)
        for i in range(num_qubits):
            if bits_h[i]: qc.h(i)
            if bits_x[i]: qc.x(i)
        qc.measure(range(num_qubits), range(num_qubits))
        circuits.append(qc)
    transpiled_circuits = transpile(circuits, backend=backend, optimization_level=0)
    result = backend.run(transpiled_circuits, shots=1).result()
    for qc_result in result.results:
        counts = qc_result.data.counts
        raw_key = list(counts.keys())[0]
        if raw_key.startswith('0x'):
            value = int(raw_key, 16)
            bitstring = format(value, f'0{num_qubits}b')
        else:
            bitstring = raw_key
        if len(bitstring) != num_qubits:
            raise ValueError(f"Bitstring length {len(bitstring)} != expected {num_qubits}")
        block_bits = [int(bit) for bit in bitstring[::-1]]
        bits.extend(block_bits)
    return bits[:n]

def hamming_encode_block(data_bits_4):
    d = data_bits_4
    p1 = d[0] ^ d[1] ^ d[3]
    p2 = d[0] ^ d[2] ^ d[3]
    p3 = d[1] ^ d[2] ^ d[3]
    return [p1, p2, d[0], p3, d[1], d[2], d[3]]

def hamming_decode_block(code_bits_7):
    p1, p2, d0, p3, d1, d2, d3 = code_bits_7
    s1 = p1 ^ d0 ^ d1 ^ d3
    s2 = p2 ^ d0 ^ d2 ^ d3
    s3 = p3 ^ d1 ^ d2 ^ d3
    syndrome = (s1 << 2) | (s2 << 1) | s3
    corrected = list(code_bits_7)
    if syndrome != 0:
        error_pos = syndrome - 1
        if error_pos < 7:
            corrected[error_pos] ^= 1
    return corrected[2], corrected[4], corrected[5], corrected[6]

def bytes_to_bits(data):
    return [(byte >> i) & 1 for byte in data for i in reversed(range(8))]

def bits_to_bytes(bits):
    return bytes([sum(b << (7 - i) for i, b in enumerate(bits[n:n+8])) for n in range(0, len(bits), 8)])

def xor_bits(bits1, bits2):
    return [int(b1) ^ int(b2) for b1, b2 in zip(bits1, bits2)]

def embed_lsb_rgb_random(image, bits, password):
    img = image.copy().convert("RGBA")
    pixels = np.array(img).copy()
    flat_pixels = pixels.reshape(-1, 4)

    total_bits = len(bits)
    total_pixels_needed = (total_bits + 2) // 3
    if total_pixels_needed > len(flat_pixels):
        raise ValueError("Not enough pixels")

    # Deterministic shuffle
    seed = int(hashlib.sha256(password.encode()).hexdigest(), 16)
    rng = np.random.default_rng(seed)
    pixel_indices = rng.permutation(len(flat_pixels))

    bit_idx = 0
    for idx in pixel_indices[:total_pixels_needed]:
        r_bit = bits[bit_idx] if bit_idx < total_bits else 0
        g_bit = bits[bit_idx + 1] if bit_idx + 1 < total_bits else 0
        b_bit = bits[bit_idx + 2] if bit_idx + 2 < total_bits else 0
        bit_idx += 3

        rgb = flat_pixels[idx, :3]
        rgb = (rgb & 0xFE) | np.array([r_bit, g_bit, b_bit], dtype=np.uint8)
        flat_pixels[idx, :3] = rgb

    return Image.fromarray(flat_pixels.reshape(pixels.shape), mode='RGBA')

def extract_lsb_rgb_random(image, password):
    pixels = np.array(image)
    flat_pixels = pixels.reshape(-1, 4)

    seed = int(hashlib.sha256(password.encode()).hexdigest(), 16)
    rng = np.random.default_rng(seed)
    pixel_indices = rng.permutation(len(flat_pixels))

    def get_bits_from_pixels(indices, count):
        bits = []
        for idx in indices:
            r, g, b = flat_pixels[idx][:3]
            bits.extend([r & 1, g & 1, b & 1])
            if len(bits) >= count:
                return bits[:count]
        return bits

    # Step 1: Get header
    header_bits = get_bits_from_pixels(pixel_indices, 28)
    length_decoded = decode_bits(header_bits)[:16]
    num_blocks = sum(int(b) << (15 - i) for i, b in enumerate(length_decoded))
    total_bits = 28 + num_blocks * 7

    # Step 2: Get total payload
    payload_bits = get_bits_from_pixels(pixel_indices, total_bits)
    return payload_bits


def encode_bits(data_bits):
    encoded_bits = []
    for i in range(0, len(data_bits), 4):
        block = data_bits[i:i+4]
        if len(block) < 4:
            block += [0] * (4 - len(block))
        encoded_bits.extend(hamming_encode_block(block))
    return encoded_bits

def decode_bits(encoded_bits):
    decoded_bits = []
    for i in range(0, len(encoded_bits), 7):
        block = encoded_bits[i:i+7]
        if len(block) < 7:
            break
        decoded_bits.extend(hamming_decode_block(block))
    return decoded_bits

def encode_with_length_and_crc(secret_bits):
    secret_len = len(secret_bits)
    secret_len_bits = [(secret_len >> (15 - i)) & 1 for i in range(16)]
    crc_val = zlib.crc32(bits_to_bytes(secret_bits)) & 0xFFFF
    crc_bits = [(crc_val >> i) & 1 for i in reversed(range(16))]
    termination_data_bits = [0, 0, 0, 0]
    data_bits = secret_len_bits + secret_bits + crc_bits + termination_data_bits
    num_blocks = (len(data_bits) + 3) // 4
    length_bits = [(num_blocks >> (15 - i)) & 1 for i in range(16)]
    return encode_bits(length_bits) + encode_bits(data_bits)

def decode_with_length(encoded_bits):
    length_encoded_bits = encoded_bits[:28]
    length_decoded = decode_bits(length_encoded_bits)[:16]
    num_blocks = sum(int(bit) << (15 - i) for i, bit in enumerate(length_decoded))
    start = 28
    end = start + num_blocks * 7
    data_encoded = encoded_bits[start:end]
    data_decoded = decode_bits(data_encoded)
    if data_decoded[-4:] == [0, 0, 0, 0]:
        data_decoded = data_decoded[:-4]
    if len(data_decoded) < 32:
        return None
    secret_len = sum(int(bit) << (15 - i) for i, bit in enumerate(data_decoded[:16]))
    if len(data_decoded) < 16 + secret_len + 16:
        return None
    secret_bits = data_decoded[16:16 + secret_len]
    crc_bits = data_decoded[16 + secret_len:16 + secret_len + 16]
    calc_crc = zlib.crc32(bits_to_bytes(secret_bits)) & 0xFFFF
    calc_crc_bits = [(calc_crc >> i) & 1 for i in reversed(range(16))]
    return secret_bits if crc_bits == calc_crc_bits else None

def decode_from_stego_image(stego_image):
    max_bits = stego_image.width * stego_image.height * 3
    raw_extracted_bits = extract_lsb_rgb_random(stego_image, PASSWORD)
    length_header_decoded = decode_bits(raw_extracted_bits[:28])[:16]
    num_blocks = sum(int(bit) << (15 - i) for i, bit in enumerate(length_header_decoded))
    total_bits = 28 + num_blocks * 7
    extracted_bits = raw_extracted_bits[:total_bits]
    return decode_with_length(extracted_bits)

def main():
    cover = Image.open("SpongeBob_SquarePants_character.jpg").convert("RGBA")
    secret = "Quantum secured by password!"
    secret_bits = bytes_to_bits(secret.encode('utf-8'))
    key_bits = generate_quantum_bits_deterministic(len(secret_bits), PASSWORD)
    masked_bits = xor_bits(secret_bits, key_bits)
    encoded_bits = encode_with_length_and_crc(masked_bits)
    stego = embed_lsb_rgb_random(cover, encoded_bits, PASSWORD)
    stego.save("quantum_stego_password_protected_hamming.png")
    print("Stego image saved as 'quantum_stego_password_protected_hamming.png'")
    recovered_masked_bits = decode_from_stego_image(stego)
    if recovered_masked_bits is None:
        print("Data integrity check failed. Cannot decode secret reliably.")
        return
    recovered_masked_bits = recovered_masked_bits[:len(secret_bits)]
    recovered_secret_bits = xor_bits(recovered_masked_bits, key_bits)
    recovered_secret_bytes = bits_to_bytes(recovered_secret_bits)
    try:
        recovered_secret = recovered_secret_bytes.decode('utf-8')
    except UnicodeDecodeError:
        recovered_secret = recovered_secret_bytes.decode('utf-8', errors='replace')
    print(f"Recovered secret: {recovered_secret}")

if __name__ == "__main__":
    main()
