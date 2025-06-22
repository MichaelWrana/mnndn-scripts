# file format: "direction time size"

import math
import sys
import numpy as np

"""Feeder functions"""


def neighborhood(iterable):
    iterator = iter(iterable)
    prev = 0
    item = next(iterator)  # raises StopIteration if empty
    for next_item in iterator:
        yield (prev, item, next_item)
        prev = item
        item = next_item
    yield (prev, item, None)


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last) : int(last + avg)])
        last += avg
    return out


"""Non-feeder functions"""


def get_pkt_list(trace_data):
    first_line = trace_data[0]
    first_line = first_line.split(" ")

    first_time = float(first_line[0])
    dta = []
    for line in trace_data:
        a = line
        b = a.split(" ")

        if float(b[1]) > 0:
            # dta.append(((float(b[0])- first_time), abs(int(b[2])), 1))
            dta.append(((float(b[0]) - first_time), 1))
        else:
            # dta.append(((float(b[1]) - first_time), abs(int(b[2])), -1))
            dta.append(((float(b[0]) - first_time), -1))
    return dta


def In_Out(list_data):
    In = []
    Out = []
    for p in list_data:
        if p[1] < 0:
            In.append(p)
        if p[1] >= 0:
            Out.append(p)

    return In, Out


############### TIME FEATURES #####################


def inter_pkt_time(trace_data):
    times = [x[0] for x in trace_data]
    temp = []
    for elem, next_elem in zip(times, times[1:] + [times[0]]):
        temp.append(next_elem - elem)
    return temp[:-1]


def interarrival_times(trace_data):
    In, Out = In_Out(trace_data)
    IN = inter_pkt_time(In)
    OUT = inter_pkt_time(Out)
    TOTAL = inter_pkt_time(trace_data)
    return IN, OUT, TOTAL


def interarrival_maxminmeansd_stats(trace_data):
    global tags
    tags += "max_in_interarrival,max_out_interarrival,max_total_interarrival,avg_in_interarrival,"
    tags +=  "avg_out_interarrival,avg_total_interarrival,std_in_interarrival,std_out_interarrival,"
    tags +=  "std_total_interarrival,75th_percentile_in_interarrival,75th_percentile_out_interarrival,"
    tags +=  "75th_percentile_total_interarrival,"
    interstats = []
    In, Out, Total = interarrival_times(trace_data)
    if In and Out:
        avg_in = sum(In) / float(len(In))
        avg_out = sum(Out) / float(len(Out))
        avg_total = sum(Total) / float(len(Total))
        interstats.append(
            (
                max(In),
                max(Out),
                max(Total),
                avg_in,
                avg_out,
                avg_total,
                np.std(In),
                np.std(Out),
                np.std(Total),
                np.percentile(In, 75),
                np.percentile(Out, 75),
                np.percentile(Total, 75),
            )
        )
    elif Out and not In:
        avg_out = sum(Out) / float(len(Out))
        avg_total = sum(Total) / float(len(Total))
        interstats.append(
            (
                0,
                max(Out),
                max(Total),
                0,
                avg_out,
                avg_total,
                0,
                np.std(Out),
                np.std(Total),
                0,
                np.percentile(Out, 75),
                np.percentile(Total, 75),
            )
        )
    elif In and not Out:
        avg_in = sum(In) / float(len(In))
        avg_total = sum(Total) / float(len(Total))
        interstats.append(
            (
                max(In),
                0,
                max(Total),
                avg_in,
                0,
                avg_total,
                np.std(In),
                0,
                np.std(Total),
                np.percentile(In, 75),
                0,
                np.percentile(Total, 75),
            )
        )
    else:
        interstats.extend(([0] * 12))
    return interstats


def time_percentile_stats(trace_data):
    global tags
    tags += "25th_percentile_in_times,50th_percentile_in_times,75th_percentile_in_times,100th_percentile_in_times,"
    tags +=  "25th_percentile_out_times,50th_percentile_out_times,75th_percentile_out_times,100th_percentile_out_times,"
    tags +=  "25th_percentile_total_times,50th_percentile_total_times,75th_percentile_total_times,100th_percentile_total_times,"
    Total = trace_data
    In, Out = In_Out(Total)
    In_times = [x[0] for x in In]
    Out_times = [x[0] for x in Out]
    Total_times = [x[0] for x in Total]
    STATS = []
    if In_times:
        STATS.append(np.percentile(In_times, 25))  # return 25th percentile
        STATS.append(np.percentile(In_times, 50))
        STATS.append(np.percentile(In_times, 75))
        STATS.append(np.percentile(In_times, 100))
    if not In_times:
        STATS.extend(([0] * 4))
    if Out_times:
        STATS.append(np.percentile(Out_times, 25))  # return 25th percentile
        STATS.append(np.percentile(Out_times, 50))
        STATS.append(np.percentile(Out_times, 75))
        STATS.append(np.percentile(Out_times, 100))
    if not Out_times:
        STATS.extend(([0] * 4))
    if Total_times:
        STATS.append(np.percentile(Total_times, 25))  # return 25th percentile
        STATS.append(np.percentile(Total_times, 50))
        STATS.append(np.percentile(Total_times, 75))
        STATS.append(np.percentile(Total_times, 100))
    if not Total_times:
        STATS.extend(([0] * 4))
    return STATS


def number_pkt_stats(trace_data):
    global tags
    tags += "in_count,out_count,total_count,"
    Total = trace_data
    In, Out = In_Out(Total)
    return len(In), len(Out), len(Total)


def first_and_last_30_pkts_stats(trace_data):
    global tags
    tags += "in_count_in_first30,out_count_in_first30,in_count_in_last30,out_count_in_last30,"
    Total = trace_data
    first30 = Total[:30]
    last30 = Total[-30:]
    first30in = []
    first30out = []
    for p in first30:
        if p[1] < 0:
            first30in.append(p)
        if p[1] >= 0:
            first30out.append(p)
    last30in = []
    last30out = []
    for p in last30:
        if p[1] < 0:
            last30in.append(p)
        if p[1] >= 0:
            last30out.append(p)
    stats = []
    stats.append(len(first30in))
    stats.append(len(first30out))
    stats.append(len(last30in))
    stats.append(len(last30out))
    return stats


# concentration of outgoing packets in chunks of 20 packets
def pkt_concentration_stats(trace_data):
    global tags
    tags += "std_out_concentration,avg_out_concentration,50th_out_concentration,min_out_concentrations,max_out_concentrations,"
    Total = trace_data
    chunks = [Total[x : x + 20] for x in range(0, len(Total), 20)]
    concentrations = []
    for chunk in chunks:
        c = 0
        for packet in chunk:
            if packet[1] >= 0:
                c += 1
        concentrations.append(c)
    return (
        np.std(concentrations),
        sum(concentrations) / float(len(concentrations)),
        np.percentile(concentrations, 50),
        min(concentrations),
        max(concentrations),
        concentrations,
    )


# Average number packets sent and received per second
def number_per_sec(trace_data):
    global tags
    tags += "avg_count_per_sec,std_count_per_sec,50th_count_per_sec,min_count_per_sec,max_count_per_sec,"
    Total = trace_data
    last_time = Total[-1][0]
    last_second = math.ceil(last_time)
    temp = []
    l = []
    for i in range(1, int(last_second) + 1):
        c = 0
        for p in Total:
            if p[0] <= i:
                c += 1
        temp.append(c)
    for prev, item, next in neighborhood(temp):
        x = item - prev
        l.append(x)
    avg_number_per_sec = sum(l) / float(len(l))
    return avg_number_per_sec, np.std(l), np.percentile(l, 50), min(l), max(l), l


# Variant of packet ordering features from http://cacr.uwaterloo.ca/techreports/2014/cacr2014-05.pdf
def avg_pkt_ordering_stats(trace_data):
    global tags
    tags += "avg_order_in,avg_order_out,std_order_in,std_order_out,"
    Total = trace_data
    c1 = 0
    c2 = 0
    temp1 = []
    temp2 = []
    for p in Total:
        if p[1] >= 0:
            temp1.append(c1)
        c1 += 1
        if p[1] < 0:
            temp2.append(c2)
        c2 += 1
    avg_in = sum(temp1) / float(len(temp1))
    avg_out = sum(temp2) / float(len(temp2))

    return avg_in, avg_out, np.std(temp1), np.std(temp2)


def perc_inc_out(trace_data):
    global tags
    tags += "in_percentage,out_percentage,"
    Total = trace_data
    In, Out = In_Out(Total)
    percentage_in = len(In) / float(len(Total))
    percentage_out = len(Out) / float(len(Total))
    return percentage_in, percentage_out


############### SIZE FEATURES #####################


def total_size(trace_data):
    global tags
    tags += "total_size,"
    return sum([abs(x[1]) for x in trace_data])


def in_out_size(trace_data):
    global tags
    tags += "in_size,out_size,"
    In, Out = In_Out(trace_data)
    size_in = sum([abs(x[1]) for x in In])
    size_out = sum([abs(x[1]) for x in Out])
    return size_in, size_out


def average_total_pkt_size(trace_data):
    global tags
    tags += "avg_total_size,"
    return np.mean([abs(x[1]) for x in trace_data])


def average_in_out_pkt_size(trace_data):
    global tags
    tags += "avg_in_size,avg_out_size,"
    In, Out = In_Out(trace_data)
    average_size_in = np.mean([abs(x[1]) for x in In])
    average_size_out = np.mean([abs(x[1]) for x in Out])
    return average_size_in, average_size_out


def variance_total_pkt_size(trace_data):
    global tags
    tags += "var_total_size,"
    return np.var([abs(x[1]) for x in trace_data])


def variance_in_out_pkt_size(trace_data):
    global tags
    tags += "var_in_size,var_out_size,"
    In, Out = In_Out(trace_data)
    var_size_in = np.var([abs(x[1]) for x in In])
    var_size_out = np.var([abs(x[1]) for x in Out])
    return var_size_in, var_size_out


def std_total_pkt_size(trace_data):
    global tags
    tags += "std_total_size,"
    return np.std([abs(x[1]) for x in trace_data])


def std_in_out_pkt_size(trace_data):
    global tags
    tags += "std_in_size,std_out_size,"
    In, Out = In_Out(trace_data)
    std_size_in = np.std([abs(x[1]) for x in In])
    std_size_out = np.std([abs(x[1]) for x in Out])
    return std_size_in, std_size_out


def max_in_out_pkt_size(trace_data):
    global tags
    tags += "max_in_size,max_out_size,"
    In, Out = In_Out(trace_data)
    max_size_in = max([abs(x[1]) for x in In])
    max_size_out = max([abs(x[1]) for x in Out])
    return max_size_in, max_size_out


def unique_pkt_lengths(trace_data):
    global tags
    tags += "unique_in_lengths,unique_out_lengths,unique_total_lengths,"
    In, Out = In_Out(trace_data)
    unique_lengths_in = set(In)
    unique_lengths_out = set(Out)
    unique_lengths_total = unique_lengths_in.union(unique_lengths_out)
    return len(unique_lengths_in), len(unique_lengths_out), len(unique_lengths_total)


############### FEATURE FUNCTION #####################


# If size information available add them in to function below
def extract_features(trace_data, max_size=175, return_tags=False):
    global tags
    tags = ''
    ALL_FEATURES = []

    #print(f"PRINT--INIT--TAGS: {len(tags.split(','))}")


    # ------TIME--------
    intertimestats = [x for x in interarrival_maxminmeansd_stats(trace_data)[0]]

    #print(f"PRINT--INTERTIME--TAGS: {len(tags.split(','))}")

    timestats = time_percentile_stats(trace_data)

    #print(f"PRINT--PERCENTILE--TAGS: {len(tags.split(','))}")

    number_pkts = list(number_pkt_stats(trace_data))

    #print(f"PRINT--NUMPKTS--TAGS: {len(tags.split(','))}")

    thirtypkts = first_and_last_30_pkts_stats(trace_data)

    #print(f"PRINT--30PKTS--TAGS: {len(tags.split(','))}")

    stdconc, avgconc, medconc, minconc, maxconc, conc = pkt_concentration_stats(
        trace_data
    )

    (
        avg_per_sec,
        std_per_sec,
        med_per_sec,
        min_per_sec,
        max_per_sec,
        per_sec,
    ) = number_per_sec(trace_data)

    avg_order_in, avg_order_out, std_order_in, std_order_out = avg_pkt_ordering_stats(
        trace_data
    )
    perc_in, perc_out = perc_inc_out(trace_data)

    #conc - list of numbers of outgoing packets for each chunk, where a chunk consists of 20 packets
    #altconc - list of 70 sums that represent number of outgoing packets for each chunk (chunks - 70  (lists) ± equal)
    altconc = []
    altconc = [sum(x) for x in chunkIt(conc, 70)]
    # print(conc)
    # print(altconc)
    # print(sum(conc))
    # print(sum(altconc))
    # print("can you understand this?")
    # input()

    if len(altconc) == 70:
        altconc.append(0)

    alt_per_sec = []
    alt_per_sec = [sum(x) for x in chunkIt(per_sec, 20)]
    if len(alt_per_sec) == 20:
        alt_per_sec.append(0)

    # ------SIZE--------

    # tot_size = total_size(trace_data)
    # in_size, out_size = in_out_size(trace_data)
    # avg_total_size = average_total_pkt_size(trace_data)
    # avg_size_in, avg_size_out = average_in_out_pkt_size(trace_data)
    # var_total_size = variance_total_pkt_size(trace_data)
    # var_size_in, var_size_out = variance_in_out_pkt_size(trace_data)
    # std_total_size = std_total_pkt_size(trace_data)
    # std_size_in, std_size_out = std_in_out_pkt_size(trace_data)
    # max_size_in, max_size_out = max_in_out_pkt_size(trace_data)
    # uni_len_in, uni_len_out, uni_len_total = unique_pkt_lengths(trace_data)

    #print(f"PRINT--INIT--FEATS: {len(ALL_FEATURES)}")

    # TIME Features
    ALL_FEATURES.extend(intertimestats)
    #print(f"PRINT--INTERTIME--FEATS: {len(ALL_FEATURES)}")
    ALL_FEATURES.extend(timestats)
    #print(f"PRINT--PERCENTILE--FEATS: {len(ALL_FEATURES)}")
    ALL_FEATURES.extend(number_pkts)
    #print(f"PRINT--NUMPKTS--FEATS: {len(ALL_FEATURES)}")
    ALL_FEATURES.extend(thirtypkts)
    #print(f"PRINT--30PKTS--FEATS: {len(ALL_FEATURES)}")

    ALL_FEATURES.append(stdconc)
    ALL_FEATURES.append(avgconc)
    ALL_FEATURES.append(avg_per_sec)
    ALL_FEATURES.append(std_per_sec)
    ALL_FEATURES.append(avg_order_in)
    ALL_FEATURES.append(avg_order_out)
    ALL_FEATURES.append(std_order_in)
    ALL_FEATURES.append(std_order_out)
    ALL_FEATURES.append(medconc)
    ALL_FEATURES.append(med_per_sec)
    ALL_FEATURES.append(min_per_sec)
    ALL_FEATURES.append(max_per_sec)
    ALL_FEATURES.append(minconc)
    ALL_FEATURES.append(maxconc)
    ALL_FEATURES.append(perc_in)
    ALL_FEATURES.append(perc_out)

    #print(f"PRINT0: {len(ALL_FEATURES)}")

    tags +=  'sum_alt_concentration,sum_alt_per_sec,sum_intertimestats,sum_timestats,sum_number_pkts,'
    ALL_FEATURES.append(sum(altconc))
    ALL_FEATURES.append(sum(alt_per_sec))
    ALL_FEATURES.append(sum(intertimestats))
    ALL_FEATURES.append(sum(timestats))
    ALL_FEATURES.append(sum(number_pkts))

    #tag and append conc, altconc, per_sec and alt_per_sec 
    tags += ",".join([f"alt_per_sec_{i}" for i in range(len(alt_per_sec))]) + ","
    ALL_FEATURES.extend(alt_per_sec)

    tags += ",".join([f"altconc_{i}" for i in range(len(altconc))]) + ","
    ALL_FEATURES.extend(altconc)

    tags += ",".join([f"conc_{i}" for i in range(len(conc))]) + ","
    ALL_FEATURES.extend(conc)

    tags += ",".join([f"per_sec_{i}" for i in range(len(per_sec))]) + "," 
    ALL_FEATURES.extend(per_sec)

    #print(f"PRINT1-FEATURES: {len(ALL_FEATURES)}")
    #print(f"PRINT1-TAGS: {len(tags.split(','))}")


    # SIZE FEATURES
    # ALL_FEATURES.append(tot_size)
    # ALL_FEATURES.append(in_size)
    # ALL_FEATURES.append(out_size)
    # ALL_FEATURES.append(avg_total_size)
    # ALL_FEATURES.append(avg_size_in)
    # ALL_FEATURES.append(avg_size_out)
    # ALL_FEATURES.append(var_total_size)
    # ALL_FEATURES.append(var_size_in)
    # ALL_FEATURES.append(var_size_out)
    # ALL_FEATURES.append(std_total_size)
    # ALL_FEATURES.append(std_size_in)
    # ALL_FEATURES.append(std_size_out)
    
    # ALL_FEATURES.append(max_size_in)
    # ALL_FEATURES.append(max_size_out)
    # ALL_FEATURES.append(uni_len_in)
    # ALL_FEATURES.append(uni_len_out)
    # ALL_FEATURES.append(uni_len_total)

    #print(f"PRINT-FINAL--FEATS: {len(ALL_FEATURES)}")
    #print(f"PRINT--FINAL--TAGS: {len(tags.split(','))}")

    # FROM TIME FEATURES
    # ALL_FEATURES.extend(altconc)
    # ALL_FEATURES.extend(alt_per_sec)

    # This is optional, since all other features are of equal size this gives the first n features
    # of this particular feature subset, some may be padded with 0's if too short.
    # ALL_FEATURES.extend(conc)

    # ALL_FEATURES.extend(per_sec)


    while len(ALL_FEATURES) < max_size:
        ALL_FEATURES.append(0)
    features = ALL_FEATURES[:max_size]

    tags = tags.strip(",")

    if return_tags:
        return tags

    # features = ALL_FEATURES
    return tuple(features) #, conc, altconc


if __name__ == "__main__":
    pass

# if __name__ == "__main__":

#     import pickle

#     with open("/Users/anhelinabodak/Desktop/clean/new_dataset/website_80_processed.npz", "rb") as f:
#         sample_trace = pickle.load(f)

#     traces = []
#     for i in range(100):
#         trace = sample_trace[80][i]
#         traces.append(trace)

#     # selected_trace = traces[80]

#     formatted = [(float(abs(p)), int(np.sign(p))) for p in traces[80]]

#     features = extract_features(formatted)
#     tags = extract_features(formatted, return_tags=True).split(',')

#     # print (f'len conc {len(conc)}')
#     # print (f'len altconc {len(altconc)}')

#     print(f"\nExtracted {len(features)} features.")
#     print(f"Extracted {len(tags)} tags.")

#     for i, (name, val) in enumerate(zip(tags, features)):
#         print(f"{i:<5} | {name:<25} = {val}")