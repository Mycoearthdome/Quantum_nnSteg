import numpy as np
from PIL import Image
import os
import time
import logging

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import Aer
from qiskit_experiments.library.characterization import LocalReadoutError
from qiskit_experiments.framework import ExperimentData
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_aer.noise import NoiseModel
from qiskit.circuit import Parameter
from collections import Counter

service = QiskitRuntimeService(channel="ibm_quantum", token="YOUR_API_KEY_HERE") # "ibm_quantum" becomes "ibm_cloud", "ibm_quantum_platform" after 1st July 2025
#backend = service.least_busy(backend_filter=lambda b: b.num_qubits >= 5 and b.simulator is False and b.operational)

backend = service.backend('ibm_brisbane') #ibm_sherbrooke


# --- Shared Simulation Setup ---
backend_sim = Aer.get_backend('aer_simulator')
noise_model = NoiseModel.from_backend(backend_sim)

def build_bell_rgb_encoder(r, g, b, secret_bit):
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr)

    if secret_bit == 1:
        qc.x(qr[1])  # Create |Ψ+> from |00>

    qc.h(qr[0])
    qc.cx(qr[0], qr[1])

    # Optional RGB encoding (can skip if you want minimal disturbance)
    qc.rx(np.pi * r, qr[0])
    qc.ry(np.pi * g, qr[1])
    qc.rz(np.pi * b, qr[0])

    qc.barrier()
    qc.measure_all()
    return qc

def prepare_bell_circuits(pixels, bits):
    circuits = []
    for idx, bit in enumerate(bits):
        y, x = divmod(idx, pixels.shape[1])
        r, g, b = pixels[y, x]
        qc = build_bell_rgb_encoder(r, g, b, bit)
        circuits.append(qc)
    return circuits


def get_measurement_fitter(backend, shots=1024):
    """
    Calibrates measurement error using Qiskit Experiments.
    Returns a measurement fitter object that can be used to correct raw counts.
    """
    mem_exp = LocalReadoutError(backend=backend)
    exp_data: ExperimentData = mem_exp.run(shots=shots)
    exp_data.block_for_results()  # Wait for results to be available

    fitter = exp_data.analysis_results("meas_cal_fitter").value
    return fitter

# --- Backend job submission with retry and queue wait ---
def run_with_retry(backend, circuits, meas_fitter=None, shots=1024, max_retries=5, wait_time=30):
    for attempt in range(max_retries):
        try:
            transpiled_circuits = transpile(circuits, backend)
            sampler = Sampler(backend=backend)
            job = sampler.run(transpiled_circuits, shots=shots)
            print(f"Job submitted successfully on attempt {attempt + 1}")
            results = job.result()

            bitstream = ""

            for i, pub_result in enumerate(results):

                bit_array = pub_result.data.meas  # This is BitArray print(dir(bit_array)) #DEBUG

                try:
                    bitstrings = bit_array.get_bitstrings()

                    if not bitstrings:
                        print(f"No bitstrings for circuit {i}")
                        continue

                    #Pseudo-Counts
                    bitstrings_counts = Counter(bitstrings)

                    # Apply measurement error mitigation if fitter is provided
                    mitigated_bitstrings = meas_fitter.filter(bitstrings_counts) if meas_fitter else bitstrings_counts

                    # Pick most common bitstring
                    most_common_str = max(mitigated_bitstrings.items(), key=lambda x: x[1])[0]

                    # Read bit from appropriate qubit (adjust qubit index as needed)
                    encoded_bit_index = 1
                    bitstream += most_common_str[-(encoded_bit_index + 1)]


                except Exception as e:
                    logging.error(f"Failed to decode circuit {i}: {e}")
                    continue

            if not bitstream:
                print("Bitstream is empty. All circuits failed?")
            return bitstream

        except Exception as e:
            print(f"Run attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Waiting {wait_time}s before retrying...")
                time.sleep(wait_time)
            else:
                logging.error("Max retries reached. Returning None.")
                return None

            
def decode_bell_counts(counts_list):
    bitstream = ""
    for counts in counts_list:
        if not counts:
            bitstream += "0"
            continue
        most_common = max(counts.items(), key=lambda x: x[1])[0]
        if most_common in ["00", "11"]:
            bitstream += "0"
        else:
            bitstream += "1"
    return bitstream            

# --- Measurement Error Mitigation ---
def execute(circuits, backend, qubits, batch_size=20, shots=1024):
    bitstream = ''
    total_circuits = len(circuits)
    meas_fitter = get_measurement_fitter(backend, shots=1024)
    # Run circuits in batches
    for i in range(0, total_circuits, batch_size):
        batch = circuits[i:i + batch_size]
        print(f"Running batch {i // batch_size + 1} with {len(batch)} circuits...")
        result = run_with_retry(backend, batch, shots=shots, meas_fitter=meas_fitter)
        if result is None:
            continue
        bitstream += result

        time.sleep(5)  # avoid flooding backend queue

    return bitstream

# --- Image Normalization Functions ---
def image_to_normalized_pixels(img):
    pixels = np.array(img.convert("RGBA"))
    norm_pixels = pixels[:, :, :3] / 255.0
    alpha_channel = pixels[:, :, 3]
    return norm_pixels, alpha_channel

def normalized_pixels_to_image(norm_pixels, alpha_channel):
    pixels = (norm_pixels * 255).clip(0, 255).astype(np.uint8)
    rgba = np.dstack((pixels, alpha_channel))
    return Image.fromarray(rgba, mode='RGBA')

def embed_bits_bell(img, bits, backend):
    norm_pixels, alpha = image_to_normalized_pixels(img)
    circuits = prepare_bell_circuits(norm_pixels, bits)

    print(f"Sending {len(circuits)} Bell-state encoding circuits to backend {backend.name}...")
    bitstream = execute(circuits, backend, qubits=list(range(2)))
    
    # Process bitstream to alter image (same as original method)
    for idx, bit in enumerate(bitstream):
        y, x = divmod(idx, norm_pixels.shape[1])
        result_bit = int(bit)
        norm_pixels[y, x, 0] = min(1.0, norm_pixels[y, x, 0] + 0.05) if result_bit else max(0.0, norm_pixels[y, x, 0] - 0.05)

    return normalized_pixels_to_image(norm_pixels, alpha)

def decode_bits_bell(img, bits, backend):
    norm_pixels, _ = image_to_normalized_pixels(img)
    circuits = prepare_bell_circuits(norm_pixels, bits)
    print(f"Sending {len(circuits)} Bell-state decoding circuits to backend {backend.name}...")
    meas_fitter = get_measurement_fitter(backend)
    raw_counts = []
    for i in range(0, len(circuits), 20):
        batch = circuits[i:i+20]
        result = run_with_retry(backend, batch, shots=1024, meas_fitter=meas_fitter)
        if result:
            raw_counts.extend(result)
        time.sleep(3)

    return [int(bit) for bit in raw_counts]

# --- Bit/Byte Conversions ---
def bytes_to_bits(data):
    return [(byte >> i) & 1 for byte in data for i in reversed(range(8))]

def bits_to_bytes(bits):
    return bytes([sum(b << (7 - i) for i, b in enumerate(bits[n:n+8])) for n in range(0, len(bits), 8)])

# --- Main ---
def main():
    # Load cover image and secret file
    cover_img = Image.open("SpongeBob_SquarePants_character.jpg").convert("RGBA")
    with open("secret.txt", "rb") as f:
        secret = f.read()

    bits = bytes_to_bits(secret)
    if len(bits) > cover_img.width * cover_img.height:
        raise ValueError("Secret too large for the cover image.")

    print(f"Selected backend {backend.name} for embedding/decoding.")

    # stego_img = embed_bits(encode_weights, decode_weights, cover_img, bits, backend)
    stego_img = embed_bits_bell(cover_img, bits, backend)
    stego_img.save("stego_image_bell.png")

    # decoded_bits = decode_bits(encode_weights, decode_weights, stego_img, bits, backend)
    decoded_bits = decode_bits_bell(stego_img, bits, backend)

    # Convert bits to bytes
    recovered_secret = bits_to_bytes(decoded_bits)

    # Save recovered secret
    with open("recovered_secret.txt", "wb") as f:
        f.write(recovered_secret)
    print("Recovered secret saved as recovered_secret.txt.")

if __name__ == "__main__":
    main()
