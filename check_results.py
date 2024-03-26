import json
import numpy as np
import sys
from datasets import load_dataset, load_metric
import os
from numpy import typing as npt
from typing import List, Tuple





def load_results(dataset_name: str, output_dir: str, plot=False) -> Tuple[npt.NDArray[float], List[int]]:
    all_results = os.listdir(output_dir)
    results_path = [r for r in all_results if r.startswith(f'{dataset_name}_')]
    if len(results_path) != 1:
        raise ValueError(f"Found {len(results_path)} results!")
    results_path = results_path[0]
    results = np.load(os.path.join(output_dir, results_path))
    n_shots = [int(d) for d in results_path.split('.')[-2].split('_') if d.isdigit()]
    if plot:
        plot_results_graph(results, dataset_name, n_shots)
    return results, n_shots

def extract_results(alpha=0.05):
    model = 'llama-7b'
    model = 'llama-13b'
    model = 'llama-30b'
    model = 'gpt2-xl'
    model = 'gpt2-large'
    model =sys.argv[1]
    for data_name in ['sst2', 'cr', 'subj', 'cb', 'rte', 'agnews', 'sst5', 'trec', 'dbpedia', 'nluscenario','trecfine','nlu', 'banking77', 'clinic150' ]:
        try:
            results, n_shots = load_results(data_name, output_dir='results_new_n_{}'.format(model))
        except:
            continue
        for res in res:
            print(round(np.mean(res) * 100, 1), round(np.std(res) * 100, 1))


if __name__ == '__main__':

    extract_results()
