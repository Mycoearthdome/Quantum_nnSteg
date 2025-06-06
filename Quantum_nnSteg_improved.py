import numpy as np
from PIL import Image
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import zlib
import hashlib
import cv2
import bz2

PASSWORD = "password"

# --- Noise map functions --
def check_noise_map_integrity(noise_map):
    print(f"[DEBUG] Noise map shape before embedding: {noise_map.shape}")
    print(f"[DEBUG] Noise map dtype: {noise_map.dtype}")
    print(f"[DEBUG] Noise map min, max values: {noise_map.min()}, {noise_map.max()}")
    print(f"[DEBUG] Noise map total bytes: {noise_map.nbytes}")
    print(f"[DEBUG] Noise map preview (first 10 values): {noise_map.flatten()[:10]}")
    return noise_map

def compute_noise_map(image):
    try:
        gray = np.array(image.convert("L"), dtype=np.uint8)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        norm_grad = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-10)

        norm_grad = norm_grad.astype(np.float32)
        
        print(f"[DEBUG] Computed noise map with shape: {norm_grad.shape}")
        print(f"[DEBUG] Noise map min, max values: {norm_grad.min()}, {norm_grad.max()}")
        print(f"[DEBUG] Noise map preview (first 10 values): {norm_grad.flatten()[:10]}")
        
        return norm_grad.astype(np.float32)
    except Exception as e:
        print(f"[ERROR] Failed to compute noise map: {e}")
        raise

def serialize_file_data(file_path):
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    compressed_bytes = bz2.compress(file_bytes)

    file_size_bits = len(compressed_bytes) * 8  # match what extractor reads
    print(f"[DEBUG] Original file size: {len(file_bytes)} bytes")
    print(f"[DEBUG] Compressed file size: {len(compressed_bytes)} bytes")
    print(f"[DEBUG] Bit length for embedding: {file_size_bits} bits")

    # Create an 8-byte header: height (2 bytes), width (2 bytes), bit_length (4 bytes)
    # Use 0 as placeholder for height/width
    header_bytes = (0).to_bytes(2, 'big') + (0).to_bytes(2, 'big') + file_size_bits.to_bytes(4, 'big')

    return header_bytes + compressed_bytes

def deserialize_noise_map(data):
    try:
        # Extract header info
        height = int.from_bytes(data[0:2], 'big')
        width = int.from_bytes(data[2:4], 'big')
        bit_length = int.from_bytes(data[4:8], 'big')
        
        # Extract compressed data of the specified length in bytes
        compressed_data = data[8:8 + (bit_length // 8)]
        print(f"[DEBUG] Compressed data size: {len(compressed_data)}")

        decompressed_data = bz2.decompress(compressed_data)
        print(f"[DEBUG] Decompressed data size: {len(decompressed_data)}")

        noise_map = np.frombuffer(decompressed_data, dtype=np.uint8).reshape((height, width)).astype(np.float32) / 255.0
        print(f"[DEBUG] Extracted noise map shape: {noise_map.shape}")

        return noise_map

    except Exception as e:
        print(f"[ERROR] Failed to deserialize noise map: {e}")
        raise


# --- Permutation helpers ---
def get_permutation_indices(length, password):
    """
    Generate a permutation of indices based on the length and password.
    The permutation is deterministic for each password.
    """
    # Create a hash of the password to base the permutation on
    hash_input = password.encode('utf-8')
    hash_output = hashlib.sha256(hash_input).digest()

    # Generate a list of indices from 0 to length-1
    indices = list(range(length))

    # Shuffle indices based on the hash of the password
    for i in range(length - 1):
        # Use the hash to determine the swap index
        swap_index = (hash_output[i % len(hash_output)] + i) % length
        indices[i], indices[swap_index] = indices[swap_index], indices[i]
    
    return indices

def permute_bits(bits, indices):
    return [bits[i] for i in indices]

def unpermute_bits(bits, indices):
    output = [0] * len(bits)
    for i, idx in enumerate(indices):
        output[idx] = bits[i]
    return output


# --- Embedding map with optional noise map ---
def generate_embedding_map(bit_count, pixel_count, password, tag="embed"):
    seed = int(hashlib.sha256((password + f"-{tag}").encode()).hexdigest(), 16)
    rng = np.random.default_rng(seed)

    replace = bit_count > pixel_count

    pixel_indices = rng.choice(pixel_count, size=bit_count, replace=replace)
    channels = rng.integers(0, 3, size=bit_count)
    bit_planes = rng.integers(0, 3, size=bit_count)

    return pixel_indices, channels, bit_planes


# --- Quantum bits ---
def deterministic_random_gates(password, block_index, num_qubits):
    hash_input = f"{password}-{block_index}".encode('utf-8')
    digest1 = hashlib.sha256(hash_input).digest()
    digest2 = hashlib.sha256(digest1).digest()
    bits_h = [(digest1[i // 8] >> (7 - (i % 8))) & 1 for i in range(num_qubits)]
    bits_x = [(digest2[i // 8] >> (7 - (i % 8))) & 1 for i in range(num_qubits)]
    return bits_h, bits_x

def generate_quantum_bits_deterministic(n, password):
    bits = []
    num_qubits = 28
    num_blocks = (n + num_qubits - 1) // num_qubits
    backend = AerSimulator()

    for block_index in range(num_blocks):
        bits_h, bits_x = deterministic_random_gates(password, block_index, num_qubits)
        qc = QuantumCircuit(num_qubits, num_qubits)
        for i in range(num_qubits):
            if bits_h[i]: qc.h(i)
            if bits_x[i]: qc.x(i)
        qc.measure(range(num_qubits), range(num_qubits))
        
        transpiled = transpile(qc, backend=backend, optimization_level=0)
        result = backend.run(transpiled, shots=1).result()
        counts = result.get_counts(qc)

        raw_key = list(counts.keys())[0]
        bitstring = raw_key if not raw_key.startswith('0x') else format(int(raw_key, 16), f'0{num_qubits}b')
        block_bits = [int(b) for b in bitstring[::-1]]
        bits.extend(block_bits)

    return bits[:n]


# --- Hamming code ---
def hamming_encode_block(d):
    p1 = d[0] ^ d[1] ^ d[3]
    p2 = d[0] ^ d[2] ^ d[3]
    p3 = d[1] ^ d[2] ^ d[3]
    return [p1, p2, d[0], p3, d[1], d[2], d[3]]

def hamming_decode_block(c):
    p1, p2, d0, p3, d1, d2, d3 = c
    s1 = p1 ^ d0 ^ d1 ^ d3
    s2 = p2 ^ d0 ^ d2 ^ d3
    s3 = p3 ^ d1 ^ d2 ^ d3
    syndrome = (s1 << 2) | (s2 << 1) | s3
    if syndrome != 0 and 0 < syndrome <= 7:
        c[syndrome - 1] ^= 1
    return c[2], c[4], c[5], c[6]


# --- Length + CRC + Hamming ---
def encode_bits(data_bits):
    return [bit for i in range(0, len(data_bits), 4) for bit in hamming_encode_block(data_bits[i:i+4] + [0]*(4 - len(data_bits[i:i+4])) )]

def decode_bits(encoded_bits):
    return [bit for i in range(0, len(encoded_bits), 7) for bit in hamming_decode_block(encoded_bits[i:i+7])]

def encode_with_length_and_crc(bits):
    # Calculate the length of the bits
    bit_length = len(bits)
    
    # Calculate the CRC32 of the bits (for integrity check)
    crc32 = zlib.crc32(bits_to_bytes(bits))  # Ensure the bits are converted to bytes for CRC calculation
    
    # Convert the length to bits and concatenate with the CRC and original bits
    length_bits = int_to_bits(bit_length, 32)
    crc_bits = int_to_bits(crc32, 32)
    
    encoded_bits = length_bits + crc_bits + bits  # The final bitstream
    
    return encoded_bits


def decode_with_length_and_crc(encoded_bits):
    length_bits = encoded_bits[:32]
    crc_bits = encoded_bits[32:64]
    secret_bits = encoded_bits[64:]

    decoded_length = bits_to_int(length_bits)
    decoded_crc = bits_to_int(crc_bits)
    actual_crc = zlib.crc32(bits_to_bytes(secret_bits))

    print(f"[DEBUG] Expected bit length: {decoded_length}")
    print(f"[DEBUG] Actual bit length:   {len(secret_bits)}")
    print(f"[DEBUG] Expected CRC:         {decoded_crc}")
    print(f"[DEBUG] Actual CRC:           {actual_crc}")

    if actual_crc != decoded_crc:
        print("CRC mismatch, data is corrupted!")
        return None

    if decoded_length != len(secret_bits):
        print(f"Length mismatch: expected {decoded_length}, but got {len(secret_bits)} bits.")
        return None

    return secret_bits

# --- Bit utils ---
def bytes_to_bits(byte_data):
    return [int(bit) for byte in byte_data for bit in bin(byte)[2:].zfill(8)]

def bits_to_bytes(bits):
    byte_data = []
    for i in range(0, len(bits), 8):
        byte_data.append(int(''.join(str(bit) for bit in bits[i:i+8]), 2))
    return bytes(byte_data)

def xor_bits(a, b):
    return [i ^ j for i, j in zip(a, b)]

def bits_to_int(bits):
    return int(''.join(str(bit) for bit in bits), 2)

def int_to_bits(n, bit_length):
    return [int(x) for x in bin(n)[2:].zfill(bit_length)]

# --- Embedding & extraction ---
def embed_lsb_rgb_random(image, bits, password, noise_map=None, permute=True, tag="embed"):
    img = image.copy().convert("RGBA")
    pixels = np.array(img, dtype=np.uint8).copy()
    flat_pixels = pixels.reshape(-1, 4)

    capacity = len(flat_pixels) * 3
    print(f"[DEBUG] Embedding capacity: {capacity} bits")
    print(f"[DEBUG] Bits to embed: {len(bits)}")

    if len(bits) > capacity:
        raise ValueError(f"Not enough capacity in image: bits={len(bits)} capacity={capacity}")

    pixel_indices, channels, bit_planes = generate_embedding_map(len(bits), len(flat_pixels), password, tag=tag)

    if permute:
        indices = get_permutation_indices(len(bits), password)
        bits_to_embed = permute_bits(bits, indices)
    else:
        bits_to_embed = bits

    for i, (idx, channel, bit_plane) in enumerate(zip(pixel_indices, channels, bit_planes)):
        original_value = flat_pixels[idx, channel]
        bit_value = bits_to_embed[i]
        if ((original_value >> bit_plane) & 1) != bit_value:
            flat_pixels[idx, channel] ^= (1 << bit_plane)

    pixels = flat_pixels.reshape(pixels.shape)
    stego_img = Image.fromarray(pixels, "RGBA")
    return stego_img

def extract_header_bits(image, password):
    pixels = np.array(image, dtype=np.uint8)
    flat_pixels = pixels.reshape(-1, 4)
    pixel_count = len(flat_pixels)

    header_bit_length = 64  # 8 bytes = 64 bits

    # Generate deterministic embedding map (no noise map)
    pixel_indices, channels, bit_planes = generate_embedding_map(
        header_bit_length, pixel_count, password, tag="header")

    # Direct extraction without permutation
    extracted_bits = [
        (flat_pixels[idx, channel] >> bit_plane) & 1
        for idx, channel, bit_plane in zip(pixel_indices, channels, bit_planes)
    ]

    print(f"[DEBUG] Raw extracted header bits: {extracted_bits}")
    return extracted_bits

def embed_lsb_rgb_separate_header(image, full_data_bytes, password, noise_map=None):
    header_bytes = full_data_bytes[:8]
    payload_bytes = full_data_bytes[8:]

    header_bits = bytes_to_bits(header_bytes)
    payload_bits = bytes_to_bits(payload_bytes)

    # Embed header WITHOUT permutation
    intermediate_img = embed_lsb_rgb_random(image, header_bits, password, noise_map=None, permute=False, tag="header")

    # Embed payload WITH permutation and noise map
    final_img = embed_lsb_rgb_random(intermediate_img, payload_bits, password, noise_map=noise_map, permute=True, tag="payload")

    return final_img


def extract_payload_bits(image, password, total_bits):
    pixels = np.array(image, dtype=np.uint8)
    flat_pixels = pixels.reshape(-1, 4)
    pixel_count = len(flat_pixels)


    pixel_indices, channels, bit_planes = generate_embedding_map(total_bits, pixel_count, password, tag="payload")
    indices = get_permutation_indices(total_bits, password)

    extracted_bits = [
        (flat_pixels[idx, channel] >> bit_plane) & 1
        for idx, channel, bit_plane in zip(pixel_indices, channels, bit_planes)
    ]

    unpermuted_bits = unpermute_bits(extracted_bits, indices)
    return unpermuted_bits


def extract_lsb_rgb_random(image, password):
    # Skip parsing header — just extract a fixed number of bits first
    header_bits = extract_header_bits(image, password)
    header_bytes = bits_to_bytes(header_bits)

    # Just try a large enough number of bits; we'll truncate later using embedded length
    # Or: assume rest of image is payload
    payload_capacity = image.width * image.height * 9 - 64
    payload_bits = extract_payload_bits(image, password, payload_capacity)

    return header_bits + payload_bits


def compare_noise_data(original_data, extracted_data):
    if len(original_data) != len(extracted_data):
        print(f"[ERROR] Length mismatch: original({len(original_data)}) vs extracted({len(extracted_data)})")
    else:
        # Compare byte-by-byte
        for i in range(len(original_data)):
            if original_data[i] != extracted_data[i]:
                print(f"[ERROR] Data mismatch at index {i}: original={original_data[i]}, extracted={extracted_data[i]}")
                break
        else:
            print("[DEBUG] All data matched successfully.")


def deserialize_file_data(data_bytes):
    try:
        # Skip first 8 bytes (2 bytes height, 2 bytes width, 4 bytes bit length)
        header = data_bytes[:8]
        bit_length = int.from_bytes(header[4:8], 'big')
        byte_length = bit_length // 8

        compressed_data = data_bytes[8:8 + byte_length]
        decompressed_data = bz2.decompress(compressed_data)
        return decompressed_data
    except Exception as e:
        print(f"[ERROR] Failed to deserialize file data: {e}")
        raise

def embed_file_in_image(image, file_path, password):
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    compressed_bytes = bz2.compress(file_bytes)
    data_bits = bytes_to_bits(compressed_bytes)

    print(f"[DEBUG] Original file size: {len(file_bytes)} bytes")
    print(f"[DEBUG] Compressed file size: {len(compressed_bytes)} bytes")

    # Step 1: Hamming encode the data bits
    encoded_bits = encode_bits(data_bits)

    # Step 2: Add length and CRC to the encoded bits
    encoded_bits_with_crc = encode_with_length_and_crc(encoded_bits)

    # Step 3: XOR the entire encoded bitstream with quantum-generated key
    key_bits = generate_quantum_bits_deterministic(len(encoded_bits_with_crc), password)
    masked_bits = xor_bits(encoded_bits_with_crc, key_bits)

    print(f"[DEBUG] Embedding file of {len(masked_bits)} bits")

    # Split header (first 64 bits) and payload
    header_bits = masked_bits[:64]  # 32-bit length + 32-bit CRC
    payload_bits = masked_bits[64:]

    # Convert to bytes for embedding
    header_bytes = bits_to_bytes(header_bits)
    payload_bytes = bits_to_bytes(payload_bits)
    full_data_bytes = header_bytes + payload_bytes

    # Capacity check
    capacity = image.width * image.height * 9
    if len(full_data_bytes) * 8 > capacity:
        raise ValueError(f"File too large: {len(full_data_bytes)*8} bits > {capacity} bits")

    print(f"[DEBUG] Embedding capacity: {capacity} bits")
    print(f"[DEBUG] Bits to embed: {len(full_data_bytes) * 8}")

    # Embed header without permutation, payload with permutation
    stego_img = embed_lsb_rgb_separate_header(image, full_data_bytes, password, noise_map=None)
    print("[DEBUG] File data embedded into image.")
    return stego_img



def extract_file_from_image(image, password):
    # Step 1: Extract just the header (64 bits)
    header_bits = extract_header_bits(image, password)

    # These are already masked, so we extract them as-is
    payload_capacity = image.width * image.height * 9 - 64

    # Step 2: Extract the rest of the bits (payload), as many as the image can hold
    payload_bits = extract_payload_bits(image, password, payload_capacity)

    # Combine header + payload (both are masked at this point)
    all_masked_bits = header_bits + payload_bits

    # Step 3: Generate quantum key and XOR to unmask
    key_bits = generate_quantum_bits_deterministic(len(all_masked_bits), password)
    unmasked_bits = xor_bits(all_masked_bits, key_bits)

    # Step 4: Extract and validate length + CRC
    decoded_bits = decode_with_length_and_crc(unmasked_bits)
    if decoded_bits is None:
        print("[ERROR] Failed CRC or length mismatch.")
        return None

    # Step 5: Hamming decode
    data_bits = decode_bits(decoded_bits)
    data_bytes = bits_to_bytes(data_bits)

    # Step 6: Decompress the data
    try:
        decompressed = bz2.decompress(data_bytes)
        print(f"[DEBUG] Extracted file content size: {len(decompressed)} bytes")
        return decompressed
    except Exception as e:
        print(f"[ERROR] Failed to decompress: {e}")
        return None



# --- Main ---
def main():
    cover = Image.open("bird.jpeg").convert("RGBA")

    # Example: embed a file (e.g., "secret.txt") into the image
    file_to_hide = "secret.txt"
    stego_image = embed_file_in_image(cover, file_to_hide, PASSWORD)
    stego_image.save("stego_output.png")

    # Later, extract file back
    extracted_content = extract_file_from_image(stego_image, PASSWORD)
    if extracted_content is not None:
        with open("extracted_secret.txt", "wb") as f:
            f.write(extracted_content)
        print("[DEBUG] File extracted and saved as extracted_secret.txt")
    else:
        print("[ERROR] Extraction failed. No data written.")

if __name__ == "__main__":
    main()