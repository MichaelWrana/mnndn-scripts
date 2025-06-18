import numpy as np
import pandas as pd
import time
import pickle
import os
import re
import argparse

from tqdm import tqdm
from pipelinetools import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics

def load_data_and_process_data(src_dir):
    file_ids = 100
    traces = {}

    for i in range(file_ids):
        out_file = os.path.join(src_dir, f"website_{i}_front.npz")

        with open(out_file, 'rb') as f:
            traces_one_at_a_time = pickle.load(f)
            traces[i] = traces_one_at_a_time[i]

    return traces

def train_model(traces_kfp):
    X, y = traces_to_xy(traces_kfp)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.19, random_state=42)

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.55, random_state=42)

    model = RandomForestClassifier(n_jobs=2, n_estimators=100, oob_score=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    score = metrics.accuracy_score(y_val, y_pred)
    return score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", help="Path to the results_npz directory")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    results_csv_path = os.path.join(base_dir, "accuracy_results.csv")

    if not os.path.exists(results_csv_path):
        pd.DataFrame(columns=["client_dummy_num", "server_dummy_num", "start_padding_time", "max_wnd", "accuracy"])\
            .to_csv(results_csv_path, index=False)

    dir_list = [d for d in os.listdir(base_dir)]
    for dir_name in tqdm(dir_list, desc="Processing datasets"):
        src_dir = os.path.join(base_dir, dir_name)

        if not os.path.isdir(src_dir):
            print(src_dir)
            continue

        match = re.match(r'front_(\d+)_(\d+)_(\d+)_(\d+)', dir_name)
        if not match:
            print(f"Skipping invalid directory name format: {dir_name}")
            continue

        client_dummy, server_dummy, start_pad, max_wnd = match.groups()

        try:
            traces = load_data_and_process_data(src_dir)
            traces_kfp = process_traces(traces, 'kfp')
            accuracy = train_model(traces_kfp)

            result_row = {
                "client_dummy_num": int(client_dummy),
                "server_dummy_num": int(server_dummy),
                "start_padding_time": int(start_pad),
                "max_wnd": int(max_wnd),
                "accuracy": accuracy
            }
            print(accuracy)
            pd.DataFrame([result_row]).to_csv(results_csv_path, mode='a', header=False, index=False)
        except Exception as e:
            print(f"Failed -- {dir_name}: {e}")

    print(f"Results saved to {results_csv_path}")


if __name__ == "__main__":
    main()
