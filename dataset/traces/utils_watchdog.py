from time import sleep
import numpy as np
import random
import time

global adv_slp
global put_slp
global cat_slp
global rec_slp

adv_slp = 10 # NDN prefix advertisment routing convergence wait time
put_slp = 2  #NDN putchunks wait time
cat_slp = 2  #NDN catchunks wait time
rec_slp = 10  #Wireshark recording wait time

global log_folder
log_folder = 'log'

def create_logs(host):
    log_list = " ".join([log_folder + '/' + log for log in ['advertise.log', 'putchunks.log', 'catchunks.log', 'crypto.log']])
    cmd = f'bash -c \'for file in {log_list}; do [ ! -e "$file" ] && touch "$file"; done\''
    host.cmd(cmd)

def advertise_prefix(host, prefix, verbose=1):
    cmd = f'nlsrc advertise {prefix} >> {log_folder}/advertise.log 2>&1'
    if(verbose):
        print(cmd)
    out = host.cmd(cmd)
    sleep(adv_slp)

def put_file(host, address, data, verbose=1):
    cmd = f'ndnputchunks {address} < {data} >> {log_folder}/putchunks.log 2>&1 &'
    if(verbose):
        print(cmd)
    out = host.cmd(cmd)

    found = 0
    while(found==0):
        found = int(host.cmd(f'if grep -q {address} {log_folder}/putchunks.log; then echo 1; else echo 0; fi'))
        print("NF")
        sleep(0.01)

def get_file(host, address, dest, verbose=1):
    cmd = f'ndncatchunks {address} > {dest} 2>> {log_folder}/catchunks.log'
    if(verbose):
        print(cmd)
    out = host.cmd(cmd)
    #print(out)
    #sleep(cat_slp)

def dns_request(client, data_name, cache_hit_proba = 0.75):
    # make DNS directory to save previous requests by this host
    client.cmd(f'[ -d "dns" ] || mkdir -p "dns"')

    # cache miss, send DNS packets
    if random.random() > cache_hit_proba:
        get_file(
            host=client,
            address=f'dns/{data_name}',
            dest=f'dns/{data_name}.dns'
        )

def start_packet_recording(host, filename, verbose=1):
    cmd = f'sudo tcpdump -w {filename} -i any & echo $! > tcpdump.pid'
    if(verbose):
        print(cmd)
    host.cmd(cmd)
    sleep(rec_slp)

def stop_packet_recording(host, verbose=1):
    sleep(rec_slp)
    cmd = f'sudo kill $(cat tcpdump.pid)'
    if(verbose):
        print(cmd)
    host.cmd(cmd)

def get_img_host(image, server_to_image):
    for name, images in server_to_image.items():
        if image in images:
            return name.lower()

# diffie-hellman key exchange over NDN
def share_symmetric_key(client_name, client_obj, server_name, server_obj):

    client_obj.cmd('mkdir crypto')
    server_obj.cmd('mkdir crypto')

    # generate public key parameters
    client_obj.cmd(f'openssl dhparam -out crypto/{client_name}_{server_name}_params 2048  >> {log_folder}/crypto.log 2>&1')
    
    # send parameters to server
    put_file(host=client_obj, address=f'{client_name}/crypto/{client_name}_{server_name}_params', data=f'crypto/{client_name}_{server_name}_params')
    get_file(host=server_obj, address=f'{client_name}/crypto/{client_name}_{server_name}_params', dest=f'crypto/{server_name}_{client_name}_params')

    # generate public key pairs using shared parameters at server and client
    client_obj.cmd(f'openssl genpkey -paramfile crypto/{client_name}_{server_name}_params -out crypto/{client_name}_{server_name}_priv  >> {log_folder}/crypto.log 2>&1')
    client_obj.cmd(f'openssl pkey -in crypto/{client_name}_{server_name}_priv -pubout -out crypto/{client_name}_{server_name}_pub  >> {log_folder}/crypto.log 2>&1')

    server_obj.cmd(f'openssl genpkey -paramfile crypto/{server_name}_{client_name}_params -out crypto/{server_name}_{client_name}_priv  >> {log_folder}/crypto.log 2>&1')
    server_obj.cmd(f'openssl pkey -in crypto/{server_name}_{client_name}_priv -pubout -out crypto/{server_name}_{client_name}_pub  >> {log_folder}/crypto.log 2>&1')

    # send the public key from client to server
    put_file(host=client_obj, address=f'{client_name}/crypto/{client_name}_{server_name}_pub', data=f'crypto/{client_name}_{server_name}_pub')
    get_file(host=server_obj, address=f'{client_name}/crypto/{client_name}_{server_name}_pub', dest=f'crypto/{client_name}_{server_name}_pub')

    # send the public key from server to client
    put_file(host=server_obj, address=f'{server_name}/crypto/{server_name}_{client_name}_pub', data=f'crypto/{server_name}_{client_name}_pub')
    get_file(host=client_obj, address=f'{server_name}/crypto/{server_name}_{client_name}_pub', dest=f'crypto/{server_name}_{client_name}_pub')

    # derive the shared secret
    client_obj.cmd(f'openssl pkeyutl -derive -inkey crypto/{client_name}_{server_name}_priv -peerkey crypto/{server_name}_{client_name}_pub -out crypto/{server_name}_{client_name}_sym  >> {log_folder}/crypto.log 2>&1')
    server_obj.cmd(f'openssl pkeyutl -derive -inkey crypto/{server_name}_{client_name}_priv -peerkey crypto/{client_name}_{server_name}_pub -out crypto/{client_name}_{server_name}_sym  >> {log_folder}/crypto.log 2>&1')

def get_file_andana(client_name, client_obj, ars, ar_order, interest, dest_file):

    print(f'Beginning ANDaNA Request: {interest}')

    # session ID - just use UNIX timestamp
    sid = str(int(time.time()))

    # create andana folder and interest file based on the string
    client_obj.cmd('mkdir andana')
    client_obj.cmd(f'echo "{interest}" > andana/interest_{sid}')
    
    # layer encryption in reverse outgoing order
    for ar in reversed(ar_order):
        client_obj.cmd(f'openssl enc -aes-256-cbc -salt -in andana/interest_{sid} -out andana/interest_{sid} -pass file:{client_name}_{ar}_sym >> {log_folder}/crypto.log 2>&1')

    # merge the original client with AR list for upcoming loop to send the interest along
    andana_names = [client_name] + ar_order
    andana_objects = [client_obj] + [ars[id] for id in ar_order]

    # i = "current" position
    for i in range(len(andana_names) - 1):

        # (1) host interest at current position
        put_file(
            host=andana_objects[i],
            address=f'{andana_names[i]}/andana/interest_{sid}',
            data=f'andana/interest_{sid}'
        )

        # (2) request interest from next position
        andana_objects[i+1].cmd('mkdir andana')
        get_file(
            host=andana_objects[i+1],
            address=f'{andana_names[i]}/andana/interest_{sid}',
            dest=f'andana/interest_{sid}',
        )

        # (4) decrypt the layer at next position, save file
        andana_objects[i+1].cmd(f'openssl enc -d -aes-256-cbc -in andana/interest_{sid} -out andana/interest_{sid} -pass file:{andana_names[i+1]}_{client_name}_sym >> {log_folder}/crypto.log 2>&1')

    # actually send out the original interest and get the data back.

    get_file(
        host=andana_objects[-1],
        address=interest,
        dest=f'andana/data_{sid}'
    )

    for i in range(len(andana_names) - 1, 0, -1):
        # (1) encrypt the file at the current layer
        andana_objects[i].cmd(f'openssl enc -aes-256-cbc -salt -in andana/data_{sid} -out andana/data_{sid} -pass file:{client_name}_{andana_names[i]}_sym >> {log_folder}/crypto.log 2>&1')

        # (2) host the data at the current layer
        put_file(
            host=andana_objects[i],
            address=f'{andana_names[i]}/andana/data_{sid}',
            data=f'andana/data_{sid}'
        )

        # (3) request the data at the next layer

        get_file(
            host=andana_objects[i-1],
            address=f'{andana_names[i]}/andana/data_{sid}',
            dest=f'andana/data_{sid}'
        )

    # finally, decrypt the layered encryption of the file at the destination

    for ar in ar_order:
        client_obj.cmd(f'openssl enc -d -aes-256-cbc -in andana/data_{sid} -out andana/data_{sid} -pass file:{client_name}_{ar}_sym >> {log_folder}/crypto.log 2>&1')
    
    # copy into desired output destination based on arguments
    client_obj.cmd(f'cp andana/data_{sid} {dest_file}')

    print(f'ANDaNA procedure complete, attempted save into {dest_file}')

# handles waiting times to be based on updates to log files

def wait_for_update(host, filepath, check_interval=.5):
    cmd_inner = f'FILE="{filepath}"; initial_mod_time=$(stat --format="%Y" "$FILE"); while true; do if [[ $(stat --format="%Y" "$FILE") -ne $initial_mod_time ]]; then echo "File modified, exiting." >> {log_folder}/watchdog.log; exit 0; fi; sleep {check_interval}; done'
    cmd = f'bash -c \'{cmd_inner}\''
    print(cmd)
    host.cmd(cmd)

# Other functions that are not needed/abandoned

'''

# exchange public keys
get_public_key(client_obj, server_name)
get_public_key(server_obj, client_name)

# generate symmetric key
client_obj.cmd(f'openssl rand -out crypto/{server_name}_sym.key -base64 32 >> {log_folder}/crypto.log 2>&1')

# encrypt symmetric key with server's public key
client_obj.cmd(f'openssl pkeyutl -encrypt -in crypto/{server_name}_sym.key -out crypto/{server_name}_sym.bin -pubin -inkey crypto/{server_name}.pub >> {log_folder}/crypto.log 2>&1')

# send encrypted symmetric key over network
put_file(
    host=client_obj,
    address=f'/{client_name}/crypto/{server_name}_sym',
    data=f'crypto/{server_name}_sym.bin'
)

get_file(
    host=server_obj,
    address=f'/{client_name}/crypto/{server_name}_sym',
    dest_file=f'crypto/{client_name}.bin'
)

# server decrypts symmetric key
server_obj.cmd(f'openssl pkeyutl -decrypt -in crypto/{client_name}.bin -out crypto/{client_name}_sym.key -inkey crypto/id_rsa')

# create and publish an RSA key pair under /user/crypto/id_rsa
def share_rsa_pair(users, params):

    for name,obj in users.items():

        obj.cmd('mkdir crypto')
        obj.cmd('openssl genrsa -out crypto/id_rsa 2048')
        obj.cmd('openssl rsa -in crypto/id_rsa -pubout -out crypto/id_rsa.pub')

        put_file(
            host=obj,
            address=f"/{name}/crypto/id_rsa",
            data = f"crypto/id_rsa.pub"
        )

def get_public_key(client, server_name):
        get_file(
            host=client,
            address=f'/{server_name}/crypto/id_rsa',
            dest=f'crypto/{server_name}.pub'
        )

'''