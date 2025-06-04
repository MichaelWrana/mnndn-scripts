import sys
import os
import shutil
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

from utils_watchdog import *
from experiment_configs import single_website_andana

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
    topo_file = f'{experiment_dir}/topologies/default-topology.conf' # Mininet topology file directory
    website_dir = f'{experiment_dir}/websites' # folder containing website HTML files

    # setup mini-ndn network
    setLogLevel('info')
    Minindn.cleanUp()
    Minindn.verifyDependencies()
    ndn = Minindn(topoFile = topo_file)
    ndn.start()

    # Configure NDN Nodes
    info('Starting nfd and nlsr on nodes\n')
    nfds = AppManager(ndn, ndn.net.hosts, Nfd)
    nlsrs = AppManager(ndn, ndn.net.hosts, Nlsr)

    clients = {'a':ndn.net['a']}
    chosen_relays = {
        'b':ndn.net['b'],
        'c':ndn.net['c'],
    }
    chosen_server = {'d':ndn.net['d']}
    
    for name, obj in {**clients, **chosen_relays, **chosen_server}.items():
        create_logs(obj)
        advertise_prefix(host=obj, prefix=f'/{name}')

    chosen_client = clients['a']
    relay_order = ['c','b']

    put_file(
        host=chosen_server['d'], 
        address=f'd/website_0/index.html',
        data=f'{website_dir}/website_0/index.html'
    )

    get_file(
        host=chosen_client,
        address=f'd/website_0/index.html',
        dest=f'index.html'
    )

    for relay_name, relay_obj in chosen_relays.items():
        share_symmetric_key('a', chosen_client, relay_name, relay_obj)

    start_packet_recording(chosen_client,f'/home/wrana_michael/experiment_data/website_0.pcap')

    get_file_andana(
        client_name='a',
        client_obj=chosen_client,
        ars=chosen_relays,
        ar_order=relay_order,
        interest=f'd/website_0/index.html',
        dest_file=f'index.html'
    )

    stop_packet_recording(chosen_client)

    MiniNDNCLI(ndn.net)

    #single_website_andana(ndn, config, website_dir, image_dir, output_dir, wid)

    ndn.stop()

