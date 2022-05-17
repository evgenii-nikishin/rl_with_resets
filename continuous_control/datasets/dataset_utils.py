from typing import Tuple

import gym

#from continuous_control.datasets.awac_dataset import AWACDataset
#from continuous_control.datasets.d4rl_dataset import D4RLDataset
from continuous_control.datasets.dataset import Dataset
from continuous_control.utils import make_env


def make_env_and_dataset(env_name: str, seed: int, dataset_name: str,
                         video_save_folder: str) -> Tuple[gym.Env, Dataset]:
    env = make_env(env_name, seed, video_save_folder)

    if 'd4rl' in dataset_name:
        dataset = D4RLDataset(env)
    elif 'awac' in dataset_name:
        dataset = AWACDataset(env_name)
    else:
        raise NotImplementedError(f'{dataset_name} is not available!')

    return env, dataset
