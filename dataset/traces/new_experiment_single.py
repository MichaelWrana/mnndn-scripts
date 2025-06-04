import sys
import os
import pickle
import random
import time
import numpy as np
import argparse

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

if __name__ == '__main__':
    experiment_dir = '/home/wrana_michael/experiment_data' # experiment data home directory
    topo_file = f'{experiment_dir}/topologies/ba-topo.conf' # Mininet topology file directory

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

    # setup mini-ndn network
    setLogLevel('info')
    Minindn.cleanUp()
    Minindn.verifyDependencies()
    ndn = Minindn(topoFile = topo_file)
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

