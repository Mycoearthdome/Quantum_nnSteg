import numpy as np
from PIL import Image
import os
import time
import logging

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import Aer
from qiskit_experiments.library.characterization import LocalReadoutError
from qiskit_experiments.framework import AnalysisResultData
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit_aer.noise import NoiseModel
from scipy.optimize import minimize

service = QiskitRuntimeService(channel="ibm_quantum", token="YOUR_API_KEY_HERE") # "ibm_quantum" becomes "ibm_cloud", "ibm_quantum_platform" after 1st July 2025
#backend = service.least_busy(backend_filter=lambda b: b.num_qubits >= 5 and b.simulator is False and b.operational)

backend = service.backend('ibm_sherbrooke')

logging.basicConfig(level=logging.INFO)

# --- Quantum Circuit Builders ---
def create_encoding_circuit(params, r, g, b, bit):
    """
    Encoding circuit: 4 qubits with parameterized rotations plus entangling gates,
    embedding RGB and one bit.
    """
    qr = QuantumRegister(4)
    qc = QuantumCircuit(qr)
    # Embed normalized RGB as rotation angles
    qc.rx(np.pi * r, qr[0])
    qc.ry(np.pi * g, qr[1])
    qc.rz(np.pi * b, qr[2])
    # Embed the bit as RX rotation on last qubit
    qc.rx(np.pi * bit, qr[3])

    # Apply learned rotations from params
    for i in range(4):
        qc.rx(params[3 * i + 0], qr[i])
        qc.ry(params[3 * i + 1], qr[i])
        qc.rz(params[3 * i + 2], qr[i])

    # Entangle qubits for encoding correlations
    qc.cz(qr[0], qr[1])
    qc.cz(qr[1], qr[2])
    qc.cz(qr[2], qr[3])

    qc.measure_all()
    return qc

def create_decoding_circuit(params, r, g, b):
    """
    Decoding circuit: 3 qubits, parameterized rotations plus entanglement,
    used to decode bit from RGB-encoded qubits.
    """
    qr = QuantumRegister(3)
    qc = QuantumCircuit(qr)
    # Embed RGB info as rotations
    qc.rx(np.pi * r, qr[0])
    qc.ry(np.pi * g, qr[1])
    qc.rz(np.pi * b, qr[2])

    # Apply learned decoding rotations
    for i in range(3):
        qc.rx(params[3 * i + 0], qr[i])
        qc.ry(params[3 * i + 1], qr[i])
        qc.rz(params[3 * i + 2], qr[i])

    # Entangling gates to decode correlations
    qc.cz(qr[0], qr[1])
    qc.cz(qr[1], qr[2])

    qc.measure_all()
    return qc

# --- Training Dataset Generation ---
def generate_training_data(n_samples=500):
    data = []
    for _ in range(n_samples):
        r, g, b = np.random.uniform(0, 1, 3)
        bit = np.random.randint(0, 2)
        data.append((r, g, b, bit))
    return data

# --- Training ---
def train_offline_model(max_epochs=3000, save_every=100):
    logging.info("Starting offline training with simulated backend...")

    # Set up backend and noise model
    backend_sim = Aer.get_backend('aer_simulator')
    noise_model = NoiseModel.from_backend(backend_sim)

    # Generate training data
    training_data = generate_training_data()

    # Initialize weights
    encode_weights = np.random.uniform(0, 2 * np.pi, 12)
    decode_weights = np.random.uniform(0, 2 * np.pi, 9)

    # Define the loss function
    def loss(weights):
        encode_w = weights[:12]
        decode_w = weights[12:]
        bit_loss = 0.0
        shots = 50 #250

        for r, g, b, bit in training_data:
            # Create the encoding and decoding circuits
            qc1 = create_encoding_circuit(encode_w, r, g, b, bit)
            qc2 = create_decoding_circuit(decode_w, r, g, b)
            
            # Combine both circuits
            combined = QuantumCircuit(qc1.num_qubits + qc2.num_qubits)
            combined.compose(qc1, qubits=range(qc1.num_qubits), inplace=True)
            combined.compose(qc2, qubits=range(qc1.num_qubits, qc1.num_qubits + qc2.num_qubits), inplace=True)

            # Transpile and run the job
            transpiled = transpile(combined, backend_sim)
            job = backend_sim.run(transpiled, shots=shots, noise_model=noise_model)
            result = job.result()
            counts = result.get_counts()
            bitstr = max(counts, key=counts.get)
            output = 1 if bitstr[-1] == '1' else 0
            bit_loss += (output - bit) ** 2

        # Return the average loss over all training data
        return bit_loss / len(training_data)

    # Initialize the optimizer (using SciPy's COBYLA as an example)
    init_point = np.concatenate([encode_weights, decode_weights])
    best_weights = init_point.copy()
    best_loss = float("inf")

    # Optimization loop using scipy.optimize.minimize
    for epoch in range(1, max_epochs + 1):
        try:
            # Optimize the loss function with SciPy's COBYLA (non-gradient-based)
            result = minimize(loss, best_weights, method='COBYLA', options={'maxiter': 1})

            # Extract new weights and loss value from the result
            new_weights = result.x
            loss_val = result.fun

            # Update best weights and loss if we found a better solution
            if loss_val < best_loss:
                best_loss = loss_val
                best_weights = new_weights

            # Save the weights at specified intervals
            if epoch % save_every == 0 or epoch == max_epochs:
                np.save("encode_weights.npy", best_weights[:12])
                np.save("decode_weights.npy", best_weights[12:])
                logging.info(f"Epoch {epoch}: Loss={loss_val:.6f}")
        
        except Exception as e:
            logging.error(f"Epoch {epoch} failed: {e}")

    # Final save after training is complete
    np.save("encode_weights.npy", best_weights[:12])
    np.save("decode_weights.npy", best_weights[12:])
    logging.info("Training complete.")

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
            result = sampler.run(transpiled_circuits, shots=shots).result()
            
            logging.info(f"Job submitted successfully on attempt {attempt + 1}")
            return result
        except Exception as e:
            logging.warning(f"Run attempt {attempt + 1} failed: {e}")
            
            # If there are retries left, wait and try again
            if attempt < max_retries - 1:
                logging.info(f"Waiting {wait_time}s before retrying...")
                time.sleep(wait_time)
            else:
                logging.error("Max retries reached. Returning None.")
                return None

# --- Measurement Error Mitigation ---
def execute_with_mitigation(circuits, backend, batch_size=20):
    mitigated_counts_all = []
    total_circuits = len(circuits)

    qubit_list = list(range(circuits[0].num_qubits))
    logging.info(f"Starting measurement calibration on {backend.name}...")
    
    # Run the calibration experiment
    calibration_exp = LocalReadoutError(physical_qubits=qubit_list)
    calibration_data = calibration_exp.run(backend).block_for_results()

    # Extract the mitigator from the analysis results
    analysis_results = calibration_data.analysis_results()
    mitigator = None
    for result in analysis_results:
        if hasattr(result, 'value'):
            mitigator = result.value
            break

    if mitigator is None:
        raise ValueError("Mitigator not found in calibration analysis results.")

    # Run circuits in batches and apply mitigation
    for i in range(0, total_circuits, batch_size):
        batch = circuits[i:i + batch_size]
        logging.info(f"Running batch {i // batch_size + 1} with {len(batch)} circuits...")
        result = run_with_retry(backend, batch)
        if result is None:
            logging.error(f"Batch {i // batch_size + 1} failed. Returning empty counts for this batch.")
            mitigated_counts_all.extend([{} for _ in batch])
            continue
        try:
            for circ in batch:
                raw_counts = result.get_counts(circ)
                mitigated_counts = mitigator.mitigate(raw_counts)
                mitigated_counts_all.append(mitigated_counts)
        except Exception as e:
            logging.error(f"Mitigation application failed: {e}")
            # Fall back to raw counts if mitigation fails
            for circ in batch:
                mitigated_counts_all.append(result.get_counts(circ))
        time.sleep(5)  # avoid flooding backend queue

    return mitigated_counts_all

# --- Counts to Expectation Value ---
def counts_to_expectation(counts, shots=1024):
    exp_vals = []
    for count in counts:
        p0 = count.get('0', 0) / shots
        p1 = count.get('1', 0) / shots
        exp_vals.append(p1 - p0)
    return exp_vals

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
        qc = create_encoding_circuit(encode_weights, r, g, b, bit)
        circuits.append(qc)
    return circuits

def prepare_decoding_circuits(decode_weights, pixels, n_bits):
    circuits = []
    for idx in range(n_bits):
        y, x = divmod(idx, pixels.shape[1])
        r, g, b = pixels[y, x]
        qc = create_decoding_circuit(decode_weights, r, g, b)
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
    
    logging.info(f"Sending {len(circuits)} encoding circuits to backend {backend.name}...")
    
    # Execute the circuits on the backend with mitigation
    mitigated_counts = execute_with_mitigation(circuits, backend)
    
    # Process the result of each circuit to modify the image pixels
    for idx, (bit, count) in enumerate(zip(bits, mitigated_counts)):
        y, x = divmod(idx, norm_pixels.shape[1])
        
        # Assuming each pixel corresponds to one quantum circuit (RGB values)
        # Extract the quantum measurement result (0 or 1)
        if '1' in count:  # If '1' was measured in the quantum circuit
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
    logging.info(f"Sending {len(circuits)} decoding circuits to backend {backend.name}...")
    mitigated_counts = execute_with_mitigation(circuits, backend)
    exp_vals = counts_to_expectation(mitigated_counts)
    return [1 if val > 0 else 0 for val in exp_vals]

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
        train_offline_model(max_epochs=100, save_every=25) #3000 max_epochs originally

    encode_weights = np.load("encode_weights.npy")
    decode_weights = np.load("decode_weights.npy")
    logging.info("Loaded trained model weights.")

    # Load cover image and secret file
    cover_img = Image.open("SpongeBob_SquarePants_character.jpg").convert("RGBA")
    with open("secret.txt", "rb") as f:
        secret = f.read()

    bits = bytes_to_bits(secret)
    if len(bits) > cover_img.width * cover_img.height:
        raise ValueError("Secret too large for the cover image.")

    logging.info(f"Selected backend {backend.name} for embedding/decoding.")

    # Embed secret bits into the image using quantum encoding circuits
    stego_img = embed_bits(encode_weights, cover_img, bits, backend)
    stego_img.save("stego_image.png")
    logging.info("Stego image saved as stego_image.png.")

    # Decode bits back from the stego image
    decoded_bits = decode_bits(decode_weights, stego_img, len(bits), backend)

    # Convert bits to bytes
    recovered_secret = bits_to_bytes(decoded_bits)

    # Save recovered secret
    with open("recovered_secret.txt", "wb") as f:
        f.write(recovered_secret)
    logging.info("Recovered secret saved as recovered_secret.txt.")

if __name__ == "__main__":
    main()
