import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_trace(txt_path):
    data = []
    with open(txt_path, 'r') as f:
        for line in f:
            t, d = line.strip().split('\t')
            data.append([float(t), int(d)])
    return np.array(data)

def plot_trace(trace, title, color, start_index=0, end_index=200):

    trace = trace[start_index:end_index]
    print(len(trace))
    plt.figure(figsize=(25, 5))
    plt.scatter(trace[:, 0], trace[:, 1], color=color, alpha=0.6)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Direction")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def count_outgoing_packets(trace):
    return np.sum(trace[:, 1] == 1)

def analyse_trace(trace):
    stats = {
        "original_outgoing": np.sum(trace == 1),
        "original_incoming": np.sum(trace == -1),
        "dummy_outgoing": np.sum(trace == 2),
        "dummy_incoming": np.sum(trace == -2),
    }
    
    stats["total_outgoing"] = stats["original_outgoing"] + stats["dummy_outgoing"]
    stats["total_incoming"] = stats["original_incoming"] + stats["dummy_incoming"]
    stats["total_original"] = stats["original_outgoing"] + stats["original_incoming"]
    stats["total_padded"] = stats["total_incoming"]+ stats["total_outgoing"]
    return stats

def check_packets(site_ids, version):
    general = []
    for site_id in site_ids:
        print(f"Website {site_id}")
        for trace_id in range(100):
            print (trace_id)
            padded_file = f"/Users/anhelinabodak/Desktop/mnndn/mnndn-scripts/defenses/front/results/{version}/site_{site_id}_trace:{trace_id}.txt"
            # original_file =  f"/Users/anhelinabodak/Desktop/mnndn/mnndn-scripts/defenses/front/results/{version}/site_{site_id}_trace:{trace_id}_original.txt"
            padded_trace = load_trace(padded_file)
            # original_trace = load_trace(original_file)
            # print(f'{len(padded_trace)} ------------- {len(original_trace)}')
            stats_padded = analyse_trace(padded_trace)
            general.append(stats_padded)
            # stats_original = analyse_trace(original_trace)
            for k, v in stats_padded.items():
                print(f"  {k:20s}: {v}")
            # for k, v in stats_original.items():
            #     print(f"  {k:20s}: {v}")
            print("-" * 40)

    df = pd.DataFrame(general)
    output_file = f"/Users/anhelinabodak/Desktop/mnndn/mnndn-scripts/defenses/front/packet_count_1.csv"
    df.to_csv(output_file, index=False)

if __name__ == '__main__':
    
    site_id = 3
    trace_id = 0
    
    version = 'ranpad2_0516_0034'

    original_file = f"/Users/anhelinabodak/Desktop/mnndn/mnndn-scripts/defenses/front/results/{version}/site_{site_id}_trace:{trace_id}_original.txt"
    padded_file = f"/Users/anhelinabodak/Desktop/mnndn/mnndn-scripts/defenses/front/results/{version}/site_{site_id}_trace:{trace_id}.txt"

    site_ids = [3, 10, 62, 82]

    check_packets(site_ids, version)

    original_trace = load_trace(original_file)
    padded_trace = load_trace(padded_file)

    plot_trace(original_trace, f"Original Trace - Site {site_id}, Trace {trace_id}", color='blue')
    plot_trace(padded_trace, f"Padded Trace - Site {site_id}, Trace {trace_id}", color='red')

