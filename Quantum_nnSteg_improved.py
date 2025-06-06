import numpy as np
from PIL import Image
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import zlib
import hashlib
import cv2
import bz2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

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

def generate_bits_fast(n, password):
    num_bytes = (n + 7) // 8
    key_material = b""
    counter = 0

    while len(key_material) < num_bytes:
        hasher = hashlib.sha512()
        hasher.update(password.encode())
        hasher.update(counter.to_bytes(4, 'big'))
        key_material += hasher.digest()
        counter += 1

    bits = []
    for byte in key_material[:num_bytes]:
        bits.extend([(byte >> i) & 1 for i in reversed(range(8))])

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
    # Ensure the bitstream is 8-bit aligned
    if len(bits) % 8 != 0:
        pad_len = 8 - (len(bits) % 8)
        bits += [0] * pad_len
        print(f"[DEBUG] Added {pad_len} bits of padding to align CRC input to byte boundary.")

    bit_length = len(bits)
    crc32 = zlib.crc32(bits_to_bytes(bits))
    length_bits = int_to_bits(bit_length, 32)
    crc_bits = int_to_bits(crc32, 32)

    return length_bits + crc_bits + bits


def decode_with_length_and_crc(encoded_bits):
    length_bits = encoded_bits[:32]
    crc_bits = encoded_bits[32:64]
    secret_bits = encoded_bits[64:]

    decoded_length = bits_to_int(length_bits)
    decoded_crc = bits_to_int(crc_bits)
    
    # Truncate to actual declared length
    if len(secret_bits) < decoded_length:
        print(f"[ERROR] Not enough bits in payload. Expected {decoded_length}, got {len(secret_bits)}")
        return None

    secret_bits = secret_bits[:decoded_length]
    actual_crc = zlib.crc32(bits_to_bytes(secret_bits))

    print(f"[DEBUG] Expected bit length: {decoded_length}")
    print(f"[DEBUG] Actual bit length:   {len(secret_bits)}")
    print(f"[DEBUG] Expected CRC:         {decoded_crc}")
    print(f"[DEBUG] Actual CRC:           {actual_crc}")

    if actual_crc != decoded_crc:
        print("CRC mismatch, data is corrupted!")
        return None

    return secret_bits

# --- Bit utils ---
def bytes_to_bits(byte_data):
    return [int(bit) for byte in byte_data for bit in bin(byte)[2:].zfill(8)]

def bits_to_bytes(bits):
    if len(bits) % 8 != 0:
        print(f"[WARN] bits_to_bytes received non-byte-aligned bits: {len(bits)}")
        bits += [0] * (8 - len(bits) % 8)  # Pad with zeros
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

    pixel_indices, channels, bit_planes = generate_embedding_map(
        header_bit_length, pixel_count, password, tag="header")

    extracted_bits = [
        int((flat_pixels[idx, channel] >> bit_plane) & 1)
        for idx, channel, bit_plane in zip(pixel_indices, channels, bit_planes)
    ]

    print(f"[DEBUG] Extracted header bits: {extracted_bits[:64]}")  # Check the first 64 bits
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
    print(f"[DEBUG] Starting payload extraction for {total_bits} bits")
    pixels = np.array(image, dtype=np.uint8)
    flat_pixels = pixels.reshape(-1, 4)
    pixel_count = len(flat_pixels)
    print(f"[DEBUG] Pixel count: {pixel_count}")

    pixel_indices, channels, bit_planes = generate_embedding_map(total_bits, pixel_count, password, tag="payload")
    indices = get_permutation_indices(total_bits, password)

    extracted_bits = [
        (flat_pixels[idx, channel] >> bit_plane) & 1
        for idx, channel, bit_plane in zip(pixel_indices, channels, bit_planes)
    ]

    unpermuted_bits = unpermute_bits(extracted_bits, indices)
    print(f"[DEBUG] Payload bits extracted: {len(unpermuted_bits)} bits")
    return unpermuted_bits



def extract_lsb_rgb_random(image, password):
    header_bits = extract_header_bits(image, password)
    print(f"[DEBUG] Header bits: {header_bits}")
    decoded_length = bits_to_int(header_bits[:32])
    print(f"[DEBUG] Decoded length from header: {decoded_length}")
    
    if decoded_length <= 0:
        print("[ERROR] Decoded length is zero or negative, aborting payload extraction.")
        return header_bits  # or None
    
    payload_bits = extract_payload_bits(image, password, decoded_length)
    print("PAYLOAD EXTRACTED!")

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

    print(f"[DEBUG] Data bits before Hamming: {len(data_bits)}")
    print(f"[DEBUG] Bits after Hamming:       {len(encoded_bits)}")
    print(f"[DEBUG] Bits after CRC + length:  {len(encoded_bits_with_crc)}")

    # Step 3: Generate XOR key from password
    key_bits = generate_bits_fast(len(encoded_bits_with_crc), password)

    # Step 4: XOR only the payload bits (skip header for masking)
    header_bits = encoded_bits_with_crc[:64]  # 32-bit length + 32-bit CRC
    payload_bits = encoded_bits_with_crc[64:]
    masked_payload_bits = xor_bits(payload_bits, key_bits[64:])

    # Combine header (clear) + masked payload
    final_bits = header_bits + masked_payload_bits

    print(f"[DEBUG] Embedding file of {len(final_bits)} bits")

    # Convert to bytes for embedding
    final_bytes = bits_to_bytes(final_bits)

    # Capacity check
    capacity = image.width * image.height * 9
    if len(final_bytes) * 8 > capacity:
        raise ValueError(f"File too large: {len(final_bytes)*8} bits > {capacity} bits")

    print(f"[DEBUG] Embedding capacity: {capacity} bits")
    print(f"[DEBUG] Bits to embed: {len(final_bytes) * 8}")

    # Embed header without permutation, payload with permutation
    stego_img = embed_lsb_rgb_separate_header(image, final_bytes, password, noise_map=None)
    print("[DEBUG] File data embedded into image.")
    return stego_img

def extract_file_from_image(image, password):
    # Step 1: Extract the header bits (first 64 bits)
    all_masked_bits = extract_lsb_rgb_random(image, password)
    masked_header_bits = all_masked_bits[:64]
    print(f"[DEBUG] UNMasked header bits: {masked_header_bits}")  # Check masked header
    unmasked_header_bits = masked_header_bits

    # Step 3: Parse the length and CRC from the header
    length_bits = unmasked_header_bits[:32]  # First 32 bits for length
    crc_bits = unmasked_header_bits[32:]    # Next 32 bits for CRC
    decoded_length = bits_to_int(length_bits)

    print(f"[DEBUG] Decoded length from header: {decoded_length}")  # Check decoded length

    # If the length is greater than the image's pixel capacity, something went wrong
    if decoded_length <= 0 or decoded_length > image.width * image.height * 9:  # Check max image capacity
        print("[ERROR] Invalid decoded length")
        return None

    print(f"[DEBUG] Decoded bit length from header: {decoded_length}")

    # Step 4: Re-extract full masked bitstream (header + payload)
    total_bit_length = 64 + decoded_length
    all_masked_bits = all_masked_bits[:total_bit_length]
    
    # Step 5: Generate full quantum key matching full bitstream length
    full_key_bits = generate_bits_fast(total_bit_length, password)

    # Step 6: Unmask the entire bitstream
    # Leave header untouched, only unmask the payload
    unmasked_bits = all_masked_bits[:64] + xor_bits(all_masked_bits[64:], full_key_bits[64:])


    print(f"[DEBUG] Unmasked total bit length: {len(unmasked_bits)}")
    
    # Step 7: Validate CRC and extract payload
    decoded_bits = decode_with_length_and_crc(unmasked_bits)
    if decoded_bits is None:
        print("[ERROR] Failed CRC or length mismatch.")
        return None

    trunc_length = (len(decoded_bits) // 7) * 7
    decoded_bits = decoded_bits[:trunc_length]

    # Step 8: Hamming decode the payload
    data_bits = decode_bits(decoded_bits)

    print(f"[DEBUG] Bits after Hamming decode: {len(data_bits)}")

    data_bytes = bits_to_bytes(data_bits)

    

    # Step 9: Decompress to recover original file
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