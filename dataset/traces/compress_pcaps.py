import os
import time
import pickle
import numpy as np
import argparse

from pathlib import Path
from tqdm import tqdm
from scapy.all import rdpcap

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process PCAP files for a specific website ID.")
    parser.add_argument("--in_dir", type=str, default="output/") # Optional input directory with default value
    parser.add_argument("website_id",type=int,help="ID of the website to process") # Mandatory integer website ID
    parser.add_argument("--out_file",type=str) # Optional output file, default based on website_id
    args = parser.parse_args()

    in_dir = args.in_dir
    website_id = args.website_id
    website_name = f"website_{website_id}"
    # Handle default value for out_file based on website_id
    if not args.out_file:
        out_file = f"{website_name}_processed.npz"
    user_addr = '10.0.0.6'

    # Get all folder names in in_dir that are integers
    sample_ids = sorted([
        int(name) for name in os.listdir(in_dir)
        if os.path.isdir(os.path.join(in_dir, name)) and name.isdigit()
    ])

    traces = {website_id: []}
    processed_count = 0

    for sample_id in tqdm(sample_ids):
        pcap_filename = os.path.join(in_dir, str(sample_id), f'{website_name}.pcap')

        # Skip files smaller than 1MB
        if not os.path.isfile(pcap_filename) or os.path.getsize(pcap_filename) < 600_000:
            continue

        fingerprint = convert_pcap_to_fingerprint(pcap_filename, user_addr, pad=False)
        traces[website_id].append(fingerprint)
        processed_count += 1

    print(f"Total folders processed: {processed_count}")

    with open(out_file, 'wb') as f:
        pickle.dump(traces, f)