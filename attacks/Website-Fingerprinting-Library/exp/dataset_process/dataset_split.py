import numpy as np
import os
import random
import argparse

from tqdm import tqdm 
from scapy.all import rdpcap
from pipelinetools import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics

# Set a fixed seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)
np.random.seed(fix_seed)

# Set up argument parser to get dataset name from command line arguments
# parser = argparse.ArgumentParser(description="WFlib")
# parser.add_argument("--dataset", type=str, required=True, default="Undefended", help="Dataset name")
# parser.add_argument("--use_stratify", type=str, default="True", help="Whether to use stratify")

# # Parse arguments
# args = parser.parse_args()
# infile = os.path.join("./datasets", f"{args.dataset}.npz")
# dataset_path = os.path.join("./datasets", args.dataset)
# os.makedirs(dataset_path, exist_ok=True)

# assert os.path.exists(infile), f"{infile} does not exist!"

# Adding code to convert pcaps to data

dataset_path = os.path.join(".", "datasets")

os.makedirs(dataset_path, exist_ok=True)

def convert_pcap_to_fingerprint(pcap_file, user_addr, pad, max_length=5000):
    # Read the pcap file
    packets = rdpcap(pcap_file)
    
    # List to store the fingerprint
    fingerprint = []
    
    # Get the timestamp of the first packet
    first_packet_time = packets[0].time
    
    # Iterate through each packet
    for packet in packets:
        
        # Calculate the relative time
        relative_time = packet.time - first_packet_time
        
        packet_raw = bytes(packet)

        # Get the source and destination IP addresses
        src_ip = ".".join(str(byte) for byte in packet_raw[32:36])
        dst_ip = ".".join(str(byte) for byte in packet_raw[36:40])


        # Determine if the packet is incoming or outgoing
        if src_ip == user_addr:
            # Outgoing packet
            fingerprint.append(relative_time)
        elif dst_ip == user_addr:
            # Incoming packet
            fingerprint.append(-1 * relative_time)
    
    # Apply padding if requested
    if pad:
        if len(fingerprint) < max_length:
            fingerprint.extend([0] * (max_length - len(fingerprint)))
        else:
            fingerprint = fingerprint[:max_length]
    
    # Convert to a NumPy array
    fingerprint_array = np.array(fingerprint, dtype=np.float32)
    
    return fingerprint_array


# Differnt lengths in X and y
def pad_traces(traces, desired_length):
    padded_traces = []
    for trace in traces:
        if len(trace) < desired_length:
            padded_trace = np.pad(trace, (0, desired_length - len(trace)), 'constant')
        else:
            padded_trace = trace[:desired_length]
        padded_traces.append(padded_trace)
    return padded_traces


output_dir = "./test-output"
num_samples = 100
num_classes = 9
user_addr = '10.0.0.45'

traces = {i: [[] for _ in range(num_samples)] for i in range(num_classes)}

for sample_id in tqdm(range(num_samples)):
    for class_id in range(num_classes):
        pcap_filename = f'../../../../output/{sample_id}/website_{class_id}.pcap'
        
        fingerprint = convert_pcap_to_fingerprint(pcap_filename, user_addr, pad=False)
        
        traces[class_id][sample_id] = fingerprint

traces_kfp = process_traces(traces, 'kfp')

X, y = traces_to_xy(traces)

max_length = max(len(trace) for trace in X)

# Pad all traces to have the same length
X_padded = pad_traces(X, max_length)

# Convert the list of padded traces to a NumPy array
X_array = np.array(X_padded)


# Load dataset from the specified .npz file
# print("loading...", infile)
# data = np.load(infile)
# X = data["X"]
# y = data["y"]

# Ensure labels are continuous
y = np.array(y)
num_classes = len(np.unique(y))
assert num_classes == y.max() + 1, "Labels are not continuous"


# if args.use_stratify == "True":
#     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=fix_seed, stratify=y)
#     X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.9, random_state=fix_seed, stratify=y_train)
# else:
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=fix_seed)
    # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.9, random_state=fix_seed)

# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=fix_seed)
# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.9, random_state=fix_seed)

X_train, X_test, y_train, y_test = train_test_split(X_array, y, test_size=0.2, random_state=42)

# Print dataset information
print(f"Train: X = {X_train.shape}, y = {y_train.shape}")
# print(f"Valid: X = {X_valid.shape}, y = {y_valid.shape}")
print(f"Test: X = {X_test.shape}, y = {y_test.shape}")


# Save the split datasets into separate .npz files
np.savez(os.path.join(dataset_path, "train.npz"), X = X_train, y = y_train)
# np.savez(os.path.join(dataset_path, "valid.npz"), X = X_valid, y = y_valid)
np.savez(os.path.join(dataset_path, "test.npz"), X = X_test, y = y_test)