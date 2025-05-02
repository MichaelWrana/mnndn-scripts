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

from utils_watchdog import *

def perform_single_experiment(ndn, config, website_dir, image_dir, output_dir):
    
    ''' website_to_image: mapping of all website index files to the images they contain
        server_to_website: mapping of all server names to the websites they host
        server_to_image: mapping of all server names to the images they host '''
    website_to_image,server_to_website,server_to_image = config

    # get servers, DNS, and chosen user, map to mininet refs
    servers = {name.lower(): ndn.net[name.lower()] for name in server_to_website.keys()}
    dns = {'dns':ndn.net['dns']} # get DNS server mininet ref
    clients = {'user1':ndn.net['user1']}

    dns['dns'].cmd(f'[ -d "dns" ] || mkdir -p "dns"')

    # advertise to the NDN network the prefixes for the chosen user, servers, and DNS
    for name, obj in {**servers, **clients, **dns}.items():
        create_logs(obj)
        advertise_prefix(host=obj, prefix=f'/{name}')
    


    for server_name, websites in server_to_website.items():
        server_name = server_name.lower()
        for website in websites:
            # put each website's index page on the NDN network
            put_file(
                host=servers[server_name], 
                address=f'/{server_name}/{website}/index.html',
                data=f'{website_dir}/{website}/index.html'
                )
            # put the address used to reach a specific website on the DNS server
            '''
            dns['dns'].cmd(f'echo "/{server_name}" > dns/{website}.dns')
            put_file(
                host=dns['dns'],
                address=f'/dns/{website}',
                data=f'dns/{website}.dns'
            )
            '''

    for server_name, images in server_to_image.items():
        server_name = server_name.lower()
        for image in images:
            # put all the images on their servers
            put_file(
                host=servers[server_name],
                address=f'{server_name}/images/{image}',
                data=f'{image_dir}/{image}'
            )
            ''' remove for speed
            dns['dns'].cmd(f'echo "/{server_name}" > dns/{image}.dns')
            put_file(
                host=dns['dns'],
                address=f'/dns/{image}',
                data=f'dns/{image}.dns'
            )
            '''
            

    chosen_client = clients['user1']

    # get all the websites in the dataset, randomize the order
    websites = np.concatenate(list(server_to_website.values()))
    np.random.shuffle(websites)
    
    for website in websites:
        chosen_client.cmd(f'mkdir {website}')
        start_packet_recording(chosen_client,f'{output_dir}/{website}.pcap')


        server_prefix = dns_request(chosen_client, website, server_to_website)

        # request the index page of the website -> save to local directory
        get_file(
            host=chosen_client,
            address=f'{server_prefix}/{website}/index.html',
            dest=f'{website}/index.html'
        )

        # now request all the images contained in that index page -> save to local directory
        for image in website_to_image[website]:

            img_server_prefix = dns_request(chosen_client, image, server_to_image)
            get_file(
                host=chosen_client,
                address=f'{img_server_prefix}/images/{image}',
                dest=f'{website}/{image}'
            )

        # end packet recording and save as pcap
        stop_packet_recording(chosen_client)

    


if __name__ == '__main__':
    experiment_dir = '/home/michaelw/Documents/experiment_data' # experiment data home directory
    topo_file = f'{experiment_dir}/topologies/test-topo-2.conf' # Mininet topology file directory
    website_dir = f'{experiment_dir}/websites' # folder containing website HTML files
    image_dir = f'{experiment_dir}/images' # folder containing image files
    config_dir = f'{website_dir}/config' # website and experimental configuration file
    
    # command-line arguments to control the number of experiments and file locations, etc.  set to defaults above
    parser = argparse.ArgumentParser(description="Process start and end arguments.")
    parser.add_argument("--start", type=int, default=0, help="Start value (integer)")
    parser.add_argument("--end", type=int, default=1, help="End value (integer)")
    parser.add_argument("--experiment_dir", type=str, default=experiment_dir, help="Directory for the experiment")
    parser.add_argument("--topo_file", type=str, default=topo_file, help="Topology file path")
    parser.add_argument("--website_dir", type=str, default=website_dir, help="Directory for the website files")
    parser.add_argument("--image_dir", type=str, default=image_dir, help="Directory for the images")
    parser.add_argument("--config_dir", type=str, default=config_dir, help="Directory for the configuration files")

    args = parser.parse_args()

    with open(args.config_dir,'rb') as f:
        config = pickle.load(f)

    for iteration in range(args.start, args.end):
        print(f'Beginning iteration: {iteration}...')
        start_time = time.time()
        output_dir = f'{args.experiment_dir}/output/{iteration}' # output folder for resulting PCAP files
        if not os.path.exists(output_dir):
            os.makedirs(output_dir) # Create the folder if it doesn't exist

        # setup mini-ndn network
        setLogLevel('info')
        Minindn.cleanUp()
        Minindn.verifyDependencies()
        ndn = Minindn(topoFile = args.topo_file)
        ndn.start()

        # Iterate through each host and write its name and IP address to the file (for processing pcap dataset)
        
        with open(f'{output_dir}/config', "w") as file:
            for name,obj in ndn.net.items():
                file.write(f"{obj.name}:{obj.IP()}\n")

        info('Starting nfd and nlsr on nodes')
        nfds = AppManager(ndn, ndn.net.hosts, Nfd)
        nlsrs = AppManager(ndn, ndn.net.hosts, Nlsr)

        #MiniNDNCLI(ndn.net)
        #sleep(1200)

        perform_single_experiment(ndn, config, args.website_dir, args.image_dir, output_dir)

        ndn.stop()

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Iteration {iteration} complete. Execution time: {execution_time} seconds")

