from utils_watchdog import *

def reformat_dict(server_dict):
    #len("server_")=7
    reformatted = {}
    for key in server_dict.keys():
        reformatted[f"s{key[7:]}"] = server_dict[key]
    return reformatted

def find_server_by_website(servers, website_name):
    for server, websites in servers.items():
        if website_name in websites:
            return server

def single_website_nodefense(ndn, config, website_dir, image_dir, output_dir, wid):
    # website_to_image: mapping of all website index files to the images they contain
    # server_to_website: mapping of all server names to the websites they host
    # server_to_image: mapping of all server names to the images they host
    website_to_image,server_to_website,server_to_image = config
    server_to_website = reformat_dict(server_to_website)
    server_to_image = reformat_dict(server_to_image)
    
    image_request_proba = 0.05 # 5% chance for each user to request resources before the primary user.
    cache_hit_proba = 0.75 #75% chance there is no need for a DNS request before the main website

    dns = {'dns': ndn.net['dns']}
    pu = {'pu': ndn.net['pu']}
    users = {f"u{i}": ndn.net[f"u{i}"] for i in range(10)}

    # determine the server we will need to setup - do not configure anything else
    sid = find_server_by_website(server_to_website, wid)
    servers = {sid: ndn.net[sid]}

    # get the resources for this one specific website - save time
    images = website_to_image[wid]

    # advertise to the NDN network the prefixes for the chosen user, servers, and DNS
    for name, obj in {**dns, **pu, **users, **servers}.items():
        create_logs(obj)
        advertise_prefix(host=obj, prefix=f'/{name}')

    # host the website
    put_file(
        host=servers[sid], 
        address=f'{sid}/{wid}/index.html',
        data=f'{website_dir}/{wid}/index.html'
        )    
    
    # host the website address in the DNS server
    put_file(
        host=dns['dns'], 
        address=f'dns/{wid}',
        data=f'{wid}:{sid}/{wid}/index.html'
        )   

    # host the resources for that website
    for image in images:
        put_file(
            host=servers[sid],
            address=f'{sid}/images/{image}',
            data=f'{image_dir}/{image}.js'
        )

    # populate network caches: each user has a 5% chance of requesting each resource
    for _, user in users.items():
        user.cmd(f'mkdir {wid}')
        for image in images:
            if random.random() < image_request_proba:
                get_file(
                    host=user,
                    address=f'{sid}/images/{image}',
                    dest=f'{wid}/{image}.js'
                )

    # get a reference to the Minindn VM for this network device to run commands on it
    chosen_client = pu['pu']

    # create a folder to store the data and begin packet recording
    chosen_client.cmd(f'mkdir {wid}')
    start_packet_recording(chosen_client,f'{output_dir}/{wid}.pcap')

    # NDN-DNS REQUEST
    dns_request(chosen_client, wid, cache_hit_proba)

    # request the index page of the website -> save to local directory
    get_file(
        host=chosen_client,
        address=f'{sid}/{wid}/index.html',
        dest=f'{wid}/index.html'
    )

    # now request all the images contained in that index page -> save to local directory
    for image in images:
        get_file(
            host=chosen_client,
            address=f'{sid}/images/{image}',
            dest=f'{wid}/{image}.js'
        )

    # end packet recording and save as pcap
    stop_packet_recording(chosen_client)

    
'''
def single_experiment_nodefense(ndn, config, website_dir, image_dir, output_dir):

    # website_to_image: mapping of all website index files to the images they contain
    # server_to_website: mapping of all server names to the websites they host
    # server_to_image: mapping of all server names to the images they host
    website_to_image,server_to_website,server_to_image = config
    server_to_website = reformat_dict(server_to_website)
    server_to_image = reformat_dict(server_to_image)
    
    dns = {'dns': ndn.net['dns']}
    pu = {'pu': ndn.net['pu']}
    relays = {f"r{i}": ndn.net[f"r{i}"] for i in range(12)}
    users = {f"u{i}": ndn.net[f"u{i}"] for i in range(10)}
    servers = {f"s{i}": ndn.net[f"s{i}"] for i in range(50)}

    # advertise to the NDN network the prefixes for the chosen user, servers, and DNS
    for name, obj in {**dns, **primary_user, **relays, **users, **servers}.items():
        create_logs(obj)
        advertise_prefix(host=obj, prefix=f'/{name}')

    # put each website's index page on the NDN network
    for server_name, websites in server_to_website.items():
        for website in websites:
            put_file(
                host=servers[server_name], 
                address=f'/{server_name}/{website}/index.html',
                data=f'{website_dir}/{website}/index.html'
                )

    # put all the images on their servers
    for server_name, images in server_to_image.items():
        for image in images:
            put_file(
                host=servers[server_name],
                address=f'{server_name}/images/{image}.js',
                data=f'{image_dir}/{image}.js'
            )
'''