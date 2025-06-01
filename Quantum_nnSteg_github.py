import numpy as np
from PIL import Image
import os
import time
import logging

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import Aer
from qiskit_experiments.library.characterization import LocalReadoutError
from qiskit_experiments.framework import AnalysisResultData
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_aer.noise import NoiseModel
from scipy.optimize import minimize
from collections import Counter
from tqdm import tqdm
from sklearn.utils import shuffle
from multiprocessing.dummy import Pool as ThreadPool
from qiskit.circuit import Parameter
import nevergrad as ng

service = QiskitRuntimeService(channel="ibm_quantum", token="YOUR_API_KEY_HERE") # "ibm_quantum" becomes "ibm_cloud", "ibm_quantum_platform" after 1st July 2025
#backend = service.least_busy(backend_filter=lambda b: b.num_qubits >= 5 and b.simulator is False and b.operational)

backend = service.backend('ibm_sherbrooke')


# --- Shared Simulation Setup ---
backend_sim = Aer.get_backend('aer_simulator')
noise_model = NoiseModel.from_backend(backend_sim)

# --- Parameterized Circuit Setup ---
encode_params = [Parameter(f"e{i}") for i in range(12)]
decode_params = [Parameter(f"d{i}") for i in range(9)]

# --- Training Dataset Generation ---
def generate_training_data(n_samples=500):
    data = []
    for _ in range(n_samples):
        r, g, b = np.random.uniform(0, 1, 3)
        bit = np.random.randint(0, 2)
        data.append((r, g, b, bit))
    return data

def build_parameterized_encoding(r, g, b, bit):
    qr = QuantumRegister(4)
    qc = QuantumCircuit(qr)
    qc.rx(np.pi * r, qr[0])
    qc.ry(np.pi * g, qr[1])
    qc.rz(np.pi * b, qr[2])
    qc.rx(np.pi * bit, qr[3])

    for i in range(4):
        qc.rx(encode_params[3*i + 0], qr[i])
        qc.ry(encode_params[3*i + 1], qr[i])
        qc.rz(encode_params[3*i + 2], qr[i])

    qc.cz(qr[0], qr[1])
    qc.cz(qr[1], qr[2])
    qc.cz(qr[2], qr[3])

    return qc

def build_parameterized_decoding(r, g, b):
    qr = QuantumRegister(3)
    qc = QuantumCircuit(qr)
    qc.rx(np.pi * r, qr[0])
    qc.ry(np.pi * g, qr[1])
    qc.rz(np.pi * b, qr[2])

    for i in range(3):
        qc.rx(decode_params[3*i + 0], qr[i])
        qc.ry(decode_params[3*i + 1], qr[i])
        qc.rz(decode_params[3*i + 2], qr[i])

    qc.cz(qr[0], qr[1])
    qc.cz(qr[1], qr[2])

    return qc

def build_combined_bound_circuit(encode_w, decode_w, r, g, b, bit):
    qc1 = build_parameterized_encoding(r, g, b, bit)
    qc2 = build_parameterized_decoding(r, g, b)

    combined = QuantumCircuit(qc1.num_qubits + qc2.num_qubits)
    combined.compose(qc1, qubits=range(qc1.num_qubits), inplace=True)
    combined.compose(qc2, qubits=range(qc1.num_qubits, combined.num_qubits), inplace=True)

    param_dict = dict(zip(encode_params + decode_params,np.concatenate([encode_w, decode_w])))

    bound_circuit = combined.assign_parameters(param_dict)
    bound_circuit.measure_all()
    return bound_circuit

# --- Optimized Evaluation Function ---
def evaluate_sample(args):
    sample, weights, shots = args
    r, g, b, bit = sample

    # Ensure weights are split correctly even if misaligned
    encode_w = weights[:12]
    decode_w = weights[12:21]

    circuit = build_combined_bound_circuit(encode_w, decode_w, r, g, b, bit)
    transpiled = transpile(circuit, backend_sim)
    job = backend_sim.run(transpiled, shots=shots, noise_model=noise_model)
    counts = job.result().get_counts()
    prob = counts.get('1', 0) / shots
    return (prob - bit) ** 2

# --- Optimized Training Loop ---
def train_offline_model(max_epochs=5000, batch_size=100, save_every=100, shots=2048):
    print("Starting accelerated training on simulator...")

    full_data = generate_training_data(n_samples=2000)
    weights = np.random.uniform(0, 2 * np.pi, len(encode_params) + len(decode_params))
    best_loss = float("inf")

    def loss(weights, data_batch):
        # Flatten and force as NumPy array
        weights = np.array(weights).flatten()

        args_list = [(sample, weights, shots) for sample in data_batch]
        with ThreadPool() as pool:
            losses = pool.map(evaluate_sample, args_list)
        return sum(losses) / len(losses)

    # Create optimizer for the full dimensionality of weights
    optimizer = ng.optimizers.SPSA(parametrization=len(encode_params) + len(decode_params), budget=max_epochs)


    for epoch in tqdm(range(1, max_epochs + 1), desc="Training Progress"):
        full_data = shuffle(full_data)
        batch = full_data[:batch_size]

        # Ask for a candidate weights vector
        candidate = optimizer.ask()

        # Evaluate loss on this candidate
        candidate_loss = loss(candidate.value, batch)

        # Tell optimizer the loss
        optimizer.tell(candidate, candidate_loss)

        if candidate_loss < best_loss:
            best_loss = candidate_loss
            weights = candidate.value
            np.save("encode_weights.npy", weights[:12])
            np.save("decode_weights.npy", weights[12:])

        if epoch % save_every == 0 or epoch == max_epochs:
            print(f"Epoch {epoch}: Loss = {candidate_loss:.6f}")

    print(f"Training complete. Final loss: {best_loss:.6f}")


# --- Backend job submission with retry and queue wait ---
def run_with_retry(backend, circuits, shots=1024, max_retries=5, wait_time=30):
    """
    Runs the given list of quantum circuits with retries upon failure.
    Transpiles each circuit for the specified backend before running.

    Parameters:
    - backend: The quantum backend to run the job on.
    - circuits: A list of QuantumCircuit objects to be executed.
    - shots: The number of shots to run for each circuit.
    - max_retries: Maximum number of retry attempts in case of failure.
    - wait_time: The time to wait between retries (in seconds).

    Returns:
    - result: The result object from the job or None if all retries fail.
    """
    for attempt in range(max_retries):
        try:
            # Transpile each circuit for the backend before running
            transpiled_circuits = [transpile(circuit, backend) for circuit in circuits]
            
            # Create the sampler object
            sampler = Sampler(backend)
            
            # Execute the quantum circuits
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

                    # Count frequency of bitstrings
                    most_common_str, _ = Counter(bitstrings).most_common(1)[0]
                    bitstream += most_common_str


                except Exception as e:
                    logging.error(f"Failed to decode circuit {i}: {e}")
                    continue

            if not bitstream:
                print("Bitstream is empty. All circuits failed?")

            return bitstream
        
        except Exception as e:
            print(f"Run attempt {attempt + 1} failed: {e}")
            
            # If there are retries left, wait and try again
            if attempt < max_retries - 1:
                print(f"Waiting {wait_time}s before retrying...")
                time.sleep(wait_time)
            else:
                logging.error("Max retries reached. Returning None.")
                return None

# --- Measurement Error Mitigation ---
def execute(circuits, backend, batch_size=20, shots=1024):
    bitstream = ''
    total_circuits = len(circuits)
    
    # Run circuits in batches
    for i in range(0, total_circuits, batch_size):
        batch = circuits[i:i + batch_size]
        print(f"Running batch {i // batch_size + 1} with {len(batch)} circuits...")
        result = run_with_retry(backend, batch, shots=shots)
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

# --- Prepare Circuits ---
def prepare_embedding_circuits(encode_weights, pixels, bits):
    circuits = []
    for idx, bit in enumerate(bits):
        y, x = divmod(idx, pixels.shape[1])
        r, g, b = pixels[y, x]
        qc = build_parameterized_encoding(r, g, b, bit)
        param_dict = {encode_params[i]: encode_weights[i] for i in range(len(encode_params))}
        qc = qc.assign_parameters(param_dict)
        circuits.append(qc)
    return circuits

def prepare_decoding_circuits(decode_weights, pixels, n_bits):
    circuits = []
    for idx in range(n_bits):
        y, x = divmod(idx, pixels.shape[1])
        r, g, b = pixels[y, x]
        qc = build_parameterized_decoding(r, g, b)
        param_dict = {decode_params[i]: decode_weights[i] for i in range(len(decode_params))}
        qc = qc.assign_parameters(param_dict)
        circuits.append(qc)
    return circuits

def embed_bits(encode_weights, img, bits, backend):
    """
    Embed secret bits into the image using quantum encoding circuits.
    Each bit is encoded into the quantum state of the image pixels.
    """
    # Convert image to normalized pixel values (RGB)
    norm_pixels, alpha = image_to_normalized_pixels(img)
    circuits = prepare_embedding_circuits(encode_weights, norm_pixels, bits)
    
    print(f"Sending {len(circuits)} encoding circuits to backend {backend.name}...")
    
    # Execute the circuits on the backend with mitigation
    bitstream = execute(circuits, backend)
    
    # Process the result of each circuit to modify the image pixels
    for idx, bit in enumerate(bitstream):
        y, x = divmod(idx, norm_pixels.shape[1])
        
        # Assuming each pixel corresponds to one quantum circuit (RGB values)
        # Extract the quantum measurement result (0 or 1)
        if bit == "1":  # If '1' was measured in the quantum circuit
            result_bit = 1
        else:
            result_bit = 0
        
        # Use the quantum result to encode the bit into the image
        # Modify the pixel value based on the result: shift or slightly adjust color channels
        if result_bit == 1:
            # Example: Adjust red channel if the encoded bit is 1
            norm_pixels[y, x, 0] = min(1.0, norm_pixels[y, x, 0] + 0.05)  # Increase red channel
        else:
            # If the encoded bit is 0, leave the pixel unaltered or slightly reduce red
            norm_pixels[y, x, 0] = max(0.0, norm_pixels[y, x, 0] - 0.05)  # Decrease red channel

    # Convert back to image
    return normalized_pixels_to_image(norm_pixels, alpha)

# --- Decode Bits from Image ---
def decode_bits(decode_weights, img, n_bits, backend):
    norm_pixels, _ = image_to_normalized_pixels(img)
    circuits = prepare_decoding_circuits(decode_weights, norm_pixels, n_bits)
    print(f"Sending {len(circuits)} decoding circuits to backend {backend.name}...")
    bitstream = execute(circuits, backend)
    bitstream_int = [int(bit) for bit in bitstream]
    return [1 if val > 0 else 0 for val in bitstream_int]

# --- Bit/Byte Conversions ---
def bytes_to_bits(data):
    return [(byte >> i) & 1 for byte in data for i in reversed(range(8))]

def bits_to_bytes(bits):
    return bytes([sum(b << (7 - i) for i, b in enumerate(bits[n:n+8])) for n in range(0, len(bits), 8)])

# --- Main ---
def main():
    # Load trained weights
    if not (os.path.exists("encode_weights.npy") and os.path.exists("decode_weights.npy")):
         # Train model offline on simulator with noise (reduce epochs for testing)
        train_offline_model() #1000-3000 max_epochs hopefully.

    encode_weights = np.load("encode_weights.npy")
    decode_weights = np.load("decode_weights.npy")
    print("Loaded trained model weights.")

    # Load cover image and secret file
    cover_img = Image.open("SpongeBob_SquarePants_character.jpg").convert("RGBA")
    with open("secret.txt", "rb") as f:
        secret = f.read()

    bits = bytes_to_bits(secret)
    if len(bits) > cover_img.width * cover_img.height:
        raise ValueError("Secret too large for the cover image.")

    print(f"Selected backend {backend.name} for embedding/decoding.")

    # Embed secret bits into the image using quantum encoding circuits
    stego_img = embed_bits(encode_weights, cover_img, bits, backend)
    stego_img.save("stego_image.png")
    print("Stego image saved as stego_image.png.")

    # Decode bits back from the stego image
    decoded_bits = decode_bits(decode_weights, stego_img, len(bits), backend)

    # Convert bits to bytes
    recovered_secret = bits_to_bytes(decoded_bits)

    # Save recovered secret
    with open("recovered_secret.txt", "wb") as f:
        f.write(recovered_secret)
    print("Recovered secret saved as recovered_secret.txt.")

if __name__ == "__main__":
    main()
