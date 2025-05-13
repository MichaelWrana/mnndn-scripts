import numpy as np
import time
import pickle

from tqdm import tqdm 
from scapy.all import rdpcap
from pipelinetools import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics


# partially written by deepseek
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
        pcap_filename = f'../../output/{sample_id}/website_{class_id}.pcap'
        
        fingerprint = convert_pcap_to_fingerprint(pcap_filename, user_addr, pad=False)
        
        traces[class_id][sample_id] = fingerprint

traces_kfp = process_traces(traces, 'kfp')

X, y = traces_to_xy(traces)
# X = np.array(X)
# y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_jobs=2, n_estimators=100, oob_score = True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# score = metrics.accuracy_score(y_test, y_pred)

if X_train.ndim < 3:
    X_train = X_train[:, np.newaxis, :]

print("train dataset shape: {}, label shape:{}".format(X_train.shape, y_train.shape))
    
# Saving data X_train and y_train: - New addition
data_to_save = {'dataset': X_train, 'label': y_train}
np.save('rf_training_data.npy', data_to_save)


# New addition - For test dataset
if X_test.ndim < 3:
    X_test = X_test[:, np.newaxis, :]

print("test dataset shape: {}, label shape:{}".format(X_test.shape, y_test.shape))
    
# Saving data X_train and y_train: - New addition
data = {'dataset': X_test, 'label': y_test}
np.save('rf_testing_data.npy', data)


