import os
import pickle
import random
import re
import time
import numpy as np

from mininet.log import setLogLevel, info
from minindn.minindn import Minindn
from minindn.util import MiniNDNCLI
from minindn.apps.app_manager import AppManager
from minindn.apps.nfd import Nfd
from minindn.apps.nlsr import Nlsr

from experiment_configs import single_website_nodefense

def get_next_folder_number(base_dir):
    max_num = -1
    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if os.path.isdir(full_path) and name.isdigit():
            num = int(name)
            if num > max_num:
                max_num = num
    return max_num + 1

def perturb_link_delays(input_path, min_delta, max_delta):
    """
    Adjusts the delay of each link in a topology file by a random amount within [min_delta, max_delta] ms.
    The modified file is saved with '_temp' appended before the file extension.

    Parameters:
        input_path (str): Path to the input topology config file.
        min_delta (int): Minimum amount to change the delay.
        max_delta (int): Maximum amount to change the delay.
    """
    # Derive the output path
    base, ext = os.path.splitext(input_path)
    temp_output_path = f"{base}-temp{ext}"

    with open(input_path, "r") as infile:
        lines = infile.readlines()

    new_lines = []
    in_links_section = False

    for line in lines:
        stripped = line.strip()
        if stripped == "[links]":
            in_links_section = True
            new_lines.append(line)
            continue
        elif stripped.startswith("[") and stripped.endswith("]"):
            in_links_section = False
            new_lines.append(line)
            continue

        if in_links_section and ':' in stripped and 'delay=' in stripped:
            parts = stripped.split()
            link_part = parts[0]
            delay_str = parts[1]  # e.g., delay=15ms
            loss_str = parts[2]   # e.g., loss=3

            current_delay = int(delay_str.split('=')[1].replace("ms", ""))
            delta = random.uniform(min_delta, max_delta)
            if random.choice([True, False]):
                new_delay = max(0, current_delay + round(delta))
            else:
                new_delay = max(0, current_delay - round(delta))

            new_line = f"{link_part} delay={new_delay}ms {loss_str}\n"
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    with open(temp_output_path, "w") as outfile:
        outfile.writelines(new_lines)
    print(f"Modified file saved to: {temp_output_path}")

    return temp_output_path

if __name__ == '__main__':
    experiment_dir = '/home/wrana_michael/experiment_data' # experiment data home directory
    topo_file = f'{experiment_dir}/topologies/ba-topo-unstable.conf' # Mininet topology file directory

    website_dir = f'{experiment_dir}/websites' # folder containing website HTML files
    config_dir = f'{website_dir}/config' # website and experimental configuration file

    image_dir = f'{experiment_dir}/resources' # folder containing image files

    wid = "website_0"

    with open(config_dir,'rb') as f:
        config = pickle.load(f)

    iteration = get_next_folder_number(f'{experiment_dir}/output')

    print(f'Beginning iteration: {iteration}...')
    start_time = time.time()

    output_dir = f'{experiment_dir}/output/{iteration}' # output folder for resulting PCAP files

    if not os.path.exists(output_dir):
        os.makedirs(output_dir) # Create the folder if it doesn't exist

    temp_topo_file = perturb_link_delays(topo_file, min_delta=0, max_delta=4)

    # setup mini-ndn network
    setLogLevel('info')
    Minindn.cleanUp()
    Minindn.verifyDependencies()
    ndn = Minindn(topoFile = temp_topo_file)
    ndn.start()

    # Iterate through each host and write its name and IP address to the file (for processing pcap dataset)
    
    with open(f'{output_dir}/config', "w") as file:
        for name,obj in ndn.net.items():
            file.write(f"{obj.name}:{obj.IP()}\n")

    # Configure NDN Nodes
    info('Starting nfd and nlsr on nodes')
    nfds = AppManager(ndn, ndn.net.hosts, Nfd)
    nlsrs = AppManager(ndn, ndn.net.hosts, Nlsr)

    single_website_nodefense(ndn, config, website_dir, image_dir, output_dir, wid)

    # record runtime
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Iteration {iteration} complete. Execution time: {execution_time} seconds")

    ndn.stop()

