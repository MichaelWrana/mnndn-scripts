import random
import numpy as np
import bz2

from tqdm import tqdm
from itertools import product
from warnings import warn
from os import mkdir
from pickle import load, dump

# BIG BOY MODULES

try:
    from kfp_extraction import extract_features
except ModuleNotFoundError:
    warn("K-Fingerprinting module not found, related functions will crash")

from sklearn.cluster import KMeans

global folder_names
folder = {
    'results': '../results/',
    'dataset': '../datasets/',
    'data': 'data/',
    'shapelets': 'shapelets/',
    'X': 'X/',
    'y': 'y/',
    'scores': 'scores/'
}

'''make_results_folder Creates the output folder structure we typically use for our experiments (just for convenience) '''
def make_results_folder():
    mkdir(folder['results'])
    mkdir(folder['results'] + folder['shapelets'])
    mkdir(folder['results'] + folder['data'])
    mkdir(folder['results'] + folder['data'] + folder['X'])
    mkdir(folder['results'] + folder['data'] + folder['y'])

''' save_shapelets
Saves shapelets to a file based on provided arguments
Input:
    shapelets: a tuple of shapelets to be saved
    name_list: the list of names to be assigned to the shapelets
'''
def save_shapelets(shapelets, name_list):
    if isinstance(name_list, str):
        filename = folder['results'] + folder['shapelets'] + name_list
        with open(filename, 'wb') as f:
            dump(shapelets, f)
        return

    if len(shapelets) != len(name_list):
        raise IndexError("Number of filenames and shapelets do not match")

    for i, name in enumerate(name_list):
        filename = folder['results'] + folder['shapelets'] + name
        print("Saving shapelets as " + filename)
        with open(filename, 'wb') as f:
            dump(shapelets[i], f)
    
    return

''' save_xy
Saves X and y lists to a file based on provided arguments
Input:
    X: list of x values to be saved to file
    y: corresponding y values to be saved
    name: name of file
'''
def save_xy(X, y, name):
    X_name = folder['results'] + folder['data'] + folder['X'] + name
    y_name = folder['results'] + folder['data'] + folder['y'] + name

    print("Saving X as " + X_name)
    with open(X_name, 'wb') as f:
        dump(X, f)

    print("Saving y as " + y_name)
    with open(y_name, 'wb') as f:
        dump(y, f)

''' load_traces
Input:
    filename: the name of the file that contains dataset
Output: 
    new_trace: the contents of the file
'''
def load_traces(name):
    with open(folder['dataset'] + name, 'rb') as f:
        traces = load(f)
    return traces

''' load_shapelets
Loads shapelets to a file based on provided arguments
Input:
    shapelets: a tuple of shapelets to be saved
    name_list: the list of names to be assigned to the shapelets
'''
def load_shapelets(name_list):
    if isinstance(name_list, str):
        with open(folder['results'] + folder['shapelets'] + name_list, 'rb') as f:
            return load(f)

    shapelets = ()
    for name in name_list:
        filename = folder['results'] + folder['shapelets'] + name
        print("Loading shapelets from " + filename)
        with open(filename, 'rb') as f:
            shapelet = load(f)
        shapelets = shapelets + (shapelet, )

    return shapelets

def _load_single(name, x_y):
    name = folder['results'] + folder['data'] + folder[x_y] + name
    print("Loading " +  x_y +  " from " + name)
    with open(name, 'rb') as f:
        single = load(f)

    return single

'''load_xy
Load X and y files from data.  If multiple X names are provided, will only load a single y
Input:
    name_list: list of names of X and y files to be loaded (and merged)
    merge: whether to merge the X-files retrieved into a single numpy matrix
Output:
    X: data contained in X data file, (as a tuple if list is provided) (as a numpy matrix if merge=True)
    y: data contained in y data file
'''
def load_xy(name_list, merge=False):

    if isinstance(name_list, str):
        X = _load_single(name_list, 'X')
        y = _load_single(name_list, 'y')
        return X, y
    
    if len(name_list) == 1:
        X = _load_single(name_list[0], 'X')
        y = _load_single(name_list[0], 'y')
        return X, y

    X = ()
    for name in name_list:
        Xi = _load_single(name, 'X')
        X = X + (Xi, )
    if merge:
        X = np.concatenate(np.asarray(X), axis=1)
    
    y = _load_single(name_list[0], 'y')

    return X, y


'''make_name_list
Converts a dictionary of combinations of names into a single product list
'''
def make_name_list(namestring_dict):
    name_components = [] # empty list for output
    for cat in namestring_dict: # iterate over possible parameters
        values = namestring_dict[cat] # possible values for the parameter
        name_components.append([str(cat) + "=" + str(value) for value in values]) # combine and put equal sign in-between
    name_list = ['_'.join(item) for item in product(*name_components)] # create a list from the combinations
    return name_list

def merge_x(x_list):
    X = ()
    for x in x_list:
        X = X + (x, )

    X = np.concatenate(X, axis=1)
    return X

''' process_trace_positive
Input:
    trace: a list that represents a packet trace
Output: 
    new_trace: the same trace with only positive-valued packets
'''
def _process_trace_positive(trace):
    new_trace = np.zeros(len(trace), dtype=np.float16)
    last_positive = np.float16(0.0)
    for i in range(len(trace)):
        if trace[i] >= 0.0:
            last_positive = trace[i]
            new_trace[i] = trace[i]
        else:
            new_trace[i] = last_positive
            
    return new_trace

''' process_trace_negative
Input:
    trace: a list that represents a packet trace
Output: 
    new_trace: the same trace with only negative-valued packets
'''
def _process_trace_negative(trace):
    new_trace = np.zeros(len(trace), dtype=np.float16)
    last_negative = np.float16(0.0)
    for i in range(len(trace)):
        if trace[i] <= 0.0:
            last_positive = trace[i]
            new_trace[i] = trace[i]
        else:
            new_trace[i] = last_positive
            
    return new_trace

''' process_trace_zeros
Input:
    trace: a list that represents a packet trace
Output: 
    new_trace: the same trace with only nonzero-valued packets
'''
def _process_trace_zeros(trace):
    new_trace = np.asarray(trace[trace != 0.0], dtype='float32')
    return new_trace

''' process_trace_ipt
Input:
    trace: a list that represents a packet trace
Output: 
    new_trace: the same trace with matching directions, but magnitude measured in inter-packet timing
'''
def _process_trace_ipt(trace):
    new_trace = np.zeros(len(trace), dtype=np.float32)
    signs = np.sign(trace)
    magnitudes = np.abs(trace)

    for i in range(len(trace)):
        new_trace[i] = signs[i] * (magnitudes[i] - magnitudes[i-1])
    
    new_trace[0] = 0
    return new_trace
''' process_trace_kfp
Input:
    trace: a list that represents a packet trace
Output: 
    new_trace: the k-fingerprnting extracted features of a packet trace
'''
def _process_trace_kfp(trace):
    tuple_format = [(abs(packet),np.sign(packet)) for packet in trace]
    return np.asarray(extract_features(tuple_format))

''' process_trace_dfnet TODO
Input:
    trace: a list that represents a packet trace
Output: 
    new_trace: the DFNet extracted features of a packet trace
'''
def _process_trace_dfnet(trace):
    new_trace = np.zeros(len(trace), dtype=np.float32)
    return new_trace

def _process_traces(traces, process_function):
    new_traces = {} # create empty return dictionary
    
    for website_id in tqdm(traces): # loop over each key (class) in traces dictionary
        new_trace_list = list(traces[website_id]) # copy all traces in that class
        new_trace_list = [process_function(trace) for trace in new_trace_list] # process each trace in that class
        new_traces[website_id] = new_trace_list # save results into new dictionary
    
    return new_traces

''' process_traces
A convienence wrapper for processing of large sets of packet traces
Input:
    traces: A dictionary containing traces for each website class
    mode: the type of processing to be performed on each trace
            'o': do not perform any processing
            'z': return all nonzero packets
            'p': return all positive packets
            'n': return all negative packets
            'ipt': return packets measures in inter-packet timing
            'kfp': k-fingerprinting extracted features (REQUIRES k-fingerprinting module)
            'dfnet': DFNet extracted features TODO (REQUIRES DFNet module)
Output: 
    new_trace: The traces processed according to arguments provided
'''
def process_traces(traces, mode):
        if mode == 'z':
            return _process_traces(traces, _process_trace_zeros)
        elif mode == 'p':
            return _process_traces(traces, _process_trace_positive)
        elif mode == 'n':
            return _process_traces(traces, _process_trace_negative)
        elif mode == 'ipt':
            return _process_traces(traces, _process_trace_ipt)
        elif mode =='kfp':
            return _process_traces(traces, _process_trace_kfp)
        elif mode =='dfnet':
            return _process_traces(traces, _process_trace_dfnet)
        

''' generate_random_shapelets
Produce (# number) shapelets for the traces provided by randomly selecting them
Input:
    traces: dictionary of traces for each website class
    number: number of shapelets to make
Output:
    a tuple of size (# number) containing shapelets
'''
def generate_random_shapelets(traces, number):
    shapelets = () # will become tuple of length number
    for i in range(number): # generate number shapelets for each class
        website_shapelets = [None] * len(traces) # empty list for per-number shapelets
        for website_id in traces: # go through each class
            website_shapelets[website_id] = random.choice(traces[website_id]) # select a random trace
        shapelets = shapelets + (website_shapelets, ) # append to tuple

    if number == 1:
        return shapelets[0]
    else:
        return shapelets


def _reformat_cluster_shapelets(shapelets, number):
    out_format = ()
    for i in range(number):
        out_format = out_format + ([shapelets[j][i] for j in range(len(shapelets))], )
    return out_format

'''generate_cluster_shapelets
Create shapelets using clustering algorithm that clusters using feature data
Input:
    traces: dictionary of packet traces
    features: dictionary of k-fingerprinting packet features
    number: number of cluster centers to generate (must be >= 2)
Output:
    
'''
def generate_cluster_shapelets(traces, features, number):
    shapelets = [None] * len(traces)

    for website_id in tqdm(traces):
        X = features[website_id]
        X_orig = traces[website_id]

        kmeans = KMeans(n_clusters=number)
        clustering = kmeans.fit_predict(X)

        centroids = []
        for centroid in kmeans.cluster_centers_:
            centroid_idx = np.argmin(np.linalg.norm(X - centroid, axis=1))
            centroids.append(X_orig[centroid_idx])

        shapelets[website_id] = centroids

    return _reformat_cluster_shapelets(shapelets, number)

'''traces_to_xy
converts a trace dictionary into X and y for classification
Input:
    dictionary containing traces
Output: A tuple containing:
    X: list containing a packet trace at each index
    y: the correct class for the corresponding trace in X
'''
def traces_to_xy(traces):
    X, y = [], []

    for trace_id, trace_vals in traces.items():
        for trace in trace_vals:
            X.append(trace)
            y.append(trace_id)

    return X, y


'''compute_shapelet_distances
Input:
    traces: list of packet traces.
    traces.shape = (num_samples, num_traces)
    shapelets: list of shapelets for each class.
    shapelets.shape = (num_classes, )
    compare_distance: function to compute distance between single trace and single shapelet.
    should take 2 arguments: (single shapelet, single packet trace)
    should output either a single value or list of values
Output:
    distances: The computed distances between shapelets and traces.
    distances.shape = (num_samples, num_classes, len(compare_distance))
'''
def compute_shapelet_distances(traces, shapelets, compare_distance):
    # empty 2-dimensional output array
    all_distances = [[None for _ in range(len(shapelets))] for _ in range(len(traces))]

    for sample_num in tqdm(range(len(traces))): #loop over each sample
        for shapelet_num in range(len(shapelets)): # for each sample, compute the distance over every shapelet
            distance = compare_distance(shapelets[shapelet_num], traces[sample_num]) # actually compute distance
            
            all_distances[sample_num][shapelet_num] = distance # insert into output array

    return all_distances

'''mulitprocessing_distance_wrapper
Takes a list instead of multiple arguments and saves to file automatically (to work better with multiprocessing)

Input:
    [0]: filename to save output as
    [1]: X values
    [2]: y values
    [3]: shapelets
    [4]: distance function from pre-coded list: (1): 'stumpy' (2): ...
'''
def compute_shapelet_distances_mp(parameter_list):
    
    # distance function we were using before
    def stumpy_min(shapelet, trace):
        from stumpy import mass
        try:
            distance = mass(shapelet, trace)
        except ValueError:
            distance = mass(trace, shapelet)
        return distance.min()
    
    # distance function we were using before
    def stumpy_position(shapelet, trace):
        from stumpy import mass
        try:
            distance = mass(shapelet, trace)
        except ValueError:
            distance = mass(trace, shapelet)
        return np.argmin(distance)
    
    def euclid_align_pos(trace, sample):
        return euclid_align_dist(trace, sample, 'p')
    
    def euclid_align_neg(trace, sample):
        return euclid_align_dist(trace, sample, 'n')
    
    def euclid_align_dist(trace, sample, mode):
        distances = []
        
        for i in range(0, len(sample)-len(trace)):
            sample_slice = sample[i:i+len(trace)]
            moved_slice = []

            if mode == 'p':
                moved_slice = sample_slice - sample_slice[0]
            elif mode == 'n':
                moved_slice = sample_slice + abs(sample_slice[0])

            distance = np.linalg.norm(trace - moved_slice)
            distances.append(distance)
        try:
            return min(distances)
        except ValueError:
            return 0
        
    def sax_bins(packets, n_letters):

        bins = np.percentile(
            packets[packets != 0],
            np.linspace(0, 100, n_letters + 1)
        )
        bins[0] = 0
        bins[-1] = 1e1000
        return bins

    def sax_transform(packets, bins):

        indices = np.digitize(packets, bins) - 1
        alphabet = np.array([*("abcdefghijklmnopqrstuvwxyz"[:len(bins) - 1])])
        text = "".join(alphabet[indices])
        return str.encode(text)
        
    def cbd_dist(trace, sample):
        if(len(trace) >= len(sample)):
            return 0
        distances = []
        try:
            pos_trace = abs(trace)
            pos_sample = abs(sample)

            t_bins = sax_bins(pos_trace, 24)
            t_letters = sax_transform(pos_trace, t_bins)
            t_len = len(bz2.compress(t_letters))

            for i in range(0, len(sample)-len(trace), 5):
                sample_slice = pos_sample[i:i+len(trace)]
                s_bins = sax_bins(sample_slice, 24)
                s_letters = sax_transform(sample_slice, s_bins)
                s_len = len(bz2.compress(s_letters))

                len_combo = len(bz2.compress(t_letters + s_letters))

                distance = len_combo / (t_len + s_len)
                distances.append(distance)
        except IndexError:
            return 0
        
        return min(distances)
    
    name = parameter_list[0]
    X = parameter_list[1]
    y = parameter_list[2]
    shapelets = parameter_list[3]
    distance_func = parameter_list[4]
    
    if distance_func == "stumpy":
        compare_distance = stumpy_min
    elif distance_func == "stumpy_position":
        compare_distance = stumpy_position
    elif distance_func == "euclid_align_pos":
        compare_distance = euclid_align_pos
    elif distance_func == "euclid_align_neg":
        compare_distance = euclid_align_neg
    elif distance_func == "cbd":
        compare_distance = cbd_dist
    else:
        raise NameError("Invalid Distance Function Selected")

    new_X = compute_shapelet_distances(X, shapelets, compare_distance)

    save_xy(new_X, y, name)

