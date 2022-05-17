from typing import Optional

import os
import gym
import pickle
import numpy as np

from continuous_control.datasets.dataset import Dataset


class ReplayBuffer(Dataset):
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
                 capacity: int):

        observations = np.empty((capacity, *observation_space.shape),
                                dtype=observation_space.dtype)
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity, ), dtype=np.float32)
        masks = np.empty((capacity, ), dtype=np.float32)
        dones_float = np.empty((capacity, ), dtype=np.float32)
        next_observations = np.empty((capacity, *observation_space.shape),
                                     dtype=observation_space.dtype)
        super().__init__(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         masks=masks,
                         dones_float=dones_float,
                         next_observations=next_observations,
                         size=0)

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity
        
        # for saving the buffer
        self.n_parts = 4
        assert self.capacity % self.n_parts == 0

    def initialize_with_dataset(self, dataset: Dataset,
                                num_samples: Optional[int]):
        assert self.insert_index == 0, 'Can insert a batch online in an empty replay buffer.'

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert self.capacity >= num_samples, 'Dataset cannot be larger than the replay buffer capacity.'

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[
            indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: np.ndarray):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def save(self, data_path: str):
        # because of memory limits, we will dump the buffer into multiple files
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        chunk_size = self.capacity // self.n_parts
        
        for i in range(self.n_parts):
            data_chunk = [
                self.observations[i*chunk_size : (i+1)*chunk_size],
                self.actions[i*chunk_size : (i+1)*chunk_size],
                self.rewards[i*chunk_size : (i+1)*chunk_size],
                self.masks[i*chunk_size : (i+1)*chunk_size],
                self.dones_float[i*chunk_size : (i+1)*chunk_size],
                self.next_observations[i*chunk_size : (i+1)*chunk_size]
            ]
            
            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            pickle.dump(data_chunk, open(data_path_chunk, 'wb'))

    def load(self, data_path: str):
        chunk_size = self.capacity // self.n_parts
        total_size = 0
        
        for i in range(self.n_parts):            
            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            data_chunk = pickle.load(open(data_path_chunk, "rb"))
            total_size += len(data_chunk[0])

            self.observations[i*chunk_size : (i+1)*chunk_size], \
            self.actions[i*chunk_size : (i+1)*chunk_size], \
            self.rewards[i*chunk_size : (i+1)*chunk_size], \
            self.masks[i*chunk_size : (i+1)*chunk_size], \
            self.dones_float[i*chunk_size : (i+1)*chunk_size], \
            self.next_observations[i*chunk_size : (i+1)*chunk_size] = data_chunk
            
        if self.capacity != total_size:
            print('WARNING: buffer capacity does not match size of loaded data!')
        self.insert_index = 0
        self.size = total_size
