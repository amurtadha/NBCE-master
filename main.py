import argparse
import logging
from typing import List, Optional
import torch
import pandas as pd
from transformers import PreTrainedTokenizerBase
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from datasets_loader import DATASET_NAMES2LOADERS
from experiment_manager import ExperimentManager
from model_loaders import load_nbce_wrapper
from utils import get_max_n_shots, filter_extremely_long_samples, save_results

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


def get_dataset(dataset: str, tokenizer: PreTrainedTokenizerBase) -> (pd.DataFrame, pd.DataFrame, List):
    da = DATASET_NAMES2LOADERS[dataset]()
    # Filter extremely long samples from both train and test samples:
    _logger.info("filtering test set:")
    test_df = filter_extremely_long_samples(da.test_df, tokenizer, test=True)
    _logger.info("filtering train set:")
    train_df = filter_extremely_long_samples(da.train_df, tokenizer)

    return test_df, train_df, da.labels


def run_nbce_experiment(dataset: str, model: str, cache_dir: str, subsample_test_set: int, output_dir: str,
                       n_windows: List[int], n_shots_per_window: Optional[int], n_runs: int,
                       random_seed: int, right_indentation: bool, beta:float) -> None:
    n_windows=[n+1 if n>1 else n for n in n_windows]
    nbce_model = load_nbce_wrapper(model,dataset, cache_dir, right_indentation, max(n_windows), beta)


    test_df, train_df, labels = get_dataset(dataset, nbce_model.tokenizer)

    if n_shots_per_window is None:
        n_shots_per_window = get_max_n_shots(train_df, test_df, nbce_model.tokenizer, nbce_model.context_window_size)
        _logger.info(f"Found max n shot per window = {n_shots_per_window}")

    n_shots = [i * n_shots_per_window for i in n_windows]

    em = ExperimentManager(dataset, test_df, train_df, nbce_model, labels, random_seed=random_seed,
                           n_shots_per_window=n_shots_per_window, subsample_test_set=subsample_test_set)

    accuracies = em.run_experiment_across_shots(n_shots, n_runs)
    save_results(dataset, n_shots, accuracies, output_dir, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', action='store', required=True,
                        help=f'Name of dataset (for example sst2).'
                             f' The supported datasets are: {DATASET_NAMES2LOADERS.keys()}')
    parser.add_argument('--model', dest='model', action='store', default='gpt2',
                        help='HF model name to use, either gpt2 or LLaMa family models')
    parser.add_argument('--subsample-test-set', dest='subsample_test_set', action='store', required=False, type=int,
                        help='Size of test set to use to speed up eval. None means using all test set.')
    parser.add_argument('--output-dir', dest='output_dir', required=False, help="Directory for saving the results",
                        default='./temp', action='store', type=str)
    parser.add_argument('--cache-dir', help="Hugging face cache dir", type=str, default='/workspace/plm/', dest='cache_dir')
    parser.add_argument('--random-seed', dest='random_seed', required=False, default=42, action='store', type=int)
    parser.add_argument('--beta', dest='beta', required=False, default=0.25, action='store', type=float)
    parser.add_argument('--n-runs', dest='n_runs',
                        help="Number of times experiments are repeated for every number of windows", action='store',
                        type=int, default=1)
    parser.add_argument('-n', '--n-windows', dest='n_windows', help="Number of parallel context windows",
                        action='append', type=int)
    parser.add_argument('--n-shots-per-window', dest='n_shots_per_window',
                        help="number of examples to fit in each window", type=int, default=None)
    parser.add_argument('--right-indentation', dest='right_indentation', help="ident all windows to the right",
                        action='store_true', default=False)
    args = parser.parse_args()
    run_nbce_experiment(**vars(args))
