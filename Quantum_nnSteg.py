import numpy as np
from PIL import Image
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import SamplerV2 as Sampler, Session
from qiskit_aer import Aer

PASSWORD = "password"  # Your quantum key password

def hash_password_to_seed(password, block_index):
    # Simple deterministic hash function to generate a seed from password and block index
    return abs(hash(f"{password}-{block_index}")) % (2**32)

def generate_quantum_bits_deterministic(n, password):
    bits = []
    num_qubits = 28  # max qubits due to backend limit
    num_blocks = (n + num_qubits - 1) // num_qubits
    backend = Aer.get_backend('aer_simulator')

    circuits = []
    for block_index in range(num_blocks):
        seed = hash_password_to_seed(password, block_index)
        np.random.seed(seed)

        qc = QuantumCircuit(num_qubits, num_qubits)
        for i in range(num_qubits):
            if np.random.rand() > 0.5:
                qc.h(i)
            if np.random.rand() > 0.5:
                qc.x(i)
        qc.measure(range(num_qubits), range(num_qubits))

        circuits.append(qc)

    transpiled_circuits = transpile(circuits, backend=backend, optimization_level=0)

    with Session(backend=backend) as session:
        sampler = Sampler(mode=session)
        job = sampler.run(transpiled_circuits, shots=1)  # Run all circuits at once
        results = job.result()

    # results is a list of Result objects, one per circuit
    for i, pub_result in enumerate(results):
        bit_array = pub_result.data.c
        bitstrings = bit_array.get_bitstrings()
        block_bits = [int(bit) for bitstring in bitstrings for bit in bitstring]
        bits.extend(block_bits)

    return bits[:n]


def bytes_to_bits(data):
    return [(byte >> i) & 1 for byte in data for i in reversed(range(8))]

def bits_to_bytes(bits):
    return bytes([sum(b << (7 - i) for i, b in enumerate(bits[n:n+8])) for n in range(0, len(bits), 8)])

def xor_bits(bits1, bits2):
    return [b1 ^ b2 for b1, b2 in zip(bits1, bits2)]

def embed_lsb_rgb(image, bits):
    """Embed bits into the LSB of RGB channels (3 bits per pixel)."""
    img = image.copy().convert("RGBA")
    pixels = np.asarray(img, dtype=np.uint8).copy()
    height, width = pixels.shape[:2]

    flat_pixels = pixels.reshape(-1, 4)
    num_pixels_needed = (len(bits) + 2) // 3
    if num_pixels_needed > len(flat_pixels):
        raise ValueError("Not enough pixels to embed all bits.")

    bit_idx = 0
    for idx in range(num_pixels_needed):
        r_bit = bits[bit_idx] if bit_idx < len(bits) else 0
        g_bit = bits[bit_idx + 1] if bit_idx + 1 < len(bits) else 0
        b_bit = bits[bit_idx + 2] if bit_idx + 2 < len(bits) else 0
        bit_idx += 3

        # Sanity check: ensure bits are 0 or 1 integers
        r_bit = 1 if r_bit else 0
        g_bit = 1 if g_bit else 0
        b_bit = 1 if b_bit else 0

        rgb = flat_pixels[idx, :3]
        rgb &= 0xFE
        rgb |= np.array([r_bit, g_bit, b_bit], dtype=np.uint8)
        flat_pixels[idx, :3] = rgb

    pixels = flat_pixels.reshape((height, width, 4))
    return Image.fromarray(pixels, mode='RGBA')


def extract_lsb_rgb(image, num_bits):
    """Extract bits from the LSB of RGB channels (3 bits per pixel)."""
    pixels = np.array(image)
    flat_pixels = pixels.reshape(-1, 4)  # RGBA

    bits = []
    for idx in range(len(flat_pixels)):
        if len(bits) >= num_bits:
            break

        r_bit = flat_pixels[idx][0] & 1
        g_bit = flat_pixels[idx][1] & 1
        b_bit = flat_pixels[idx][2] & 1

        bits.extend([r_bit, g_bit, b_bit])

    return bits[:num_bits]

def main():
    # 1. Load cover image
    cover = Image.open("SpongeBob_SquarePants_character.jpg").convert("RGBA")

    # 2. Convert secret to bits
    secret = b"Quantum secured by password!"  #The hidden message
    secret_bits = bytes_to_bits(secret)

    # 3. Generate pseudo-quantum key using password
    key_bits = generate_quantum_bits_deterministic(len(secret_bits), PASSWORD)

    # 4. XOR secret with key
    masked_bits = xor_bits(secret_bits, key_bits)

    # 5. Embed into image (RGB LSBs)
    stego = embed_lsb_rgb(cover, masked_bits)
    stego.save("quantum_stego_password_protected.png")
    print("Stego image saved as 'quantum_stego_password_protected.png'")

    # 6. Extract and decode
    extracted = extract_lsb_rgb(stego, len(secret_bits))
    recovered_bits = xor_bits(extracted, key_bits)
    recovered_secret = bits_to_bytes(recovered_bits)

    print(f"Recovered secret: {recovered_secret.decode(errors='replace')}")
    with open("recovered_secret.txt", "wb") as f:
        f.write(recovered_secret)

if __name__ == "__main__":
    main()
