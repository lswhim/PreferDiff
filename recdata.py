import os


class AbstractRecData:
    def __init__(self, config: dict):
        self.config = config
        self.accelerator = self.config['accelerator']

        self.item_seqs = {}
        self.all_item_seqs = []

        self.mappings = {
            'user2id': {'【PAD】': 0},
            'item2id': {'【PAD】': 0},
            'id2user': ['【PAD】'],
            'id2item': ['【PAD】']
        }

    def __str__(self) -> str:
        return f'【RecData】 {self.__class__.__name__}\n' \
               f'\tNumber of users: {self.n_users}\n' \
               f'\tNumber of items: {self.n_items}\n' \
               f'\tNumber of interactions: {self.n_interactions}\n' \
               f'\tAverage item sequence length: {self.avg_item_seq_len}'

    @property
    def n_users(self):
        """
        Returns the number of users in the dataset.

        Returns:
            int: The number of users in the dataset.
        """
        return len(self.user2id)

    @property
    def n_items(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.item2id)


    def n_interactions(self):
        """
        Returns the total number of interactions in the dataset.

        Returns:
            int: The total number of interactions.
        """
        n_inters = 0
        for user in self.item_seqs:
            n_inters += len(self.item_seqs[user])
        return n_inters

    @property
    def avg_item_seq_len(self):
        """
        Returns the average length of item sequences in the dataset.

        Returns:
            float: The average length of item sequences.
        """
        return self.n_interactions / self.n_users

    @property
    def user2id(self):
        """
        Returns the user-to-id mapping.

        Returns:
            dict: The user-to-id mapping.
        """
        return self.mappings['user2id']

    @property
    def item2id(self):
        """
        Returns the item-to-id mapping.

        Returns:
            dict: The item-to-id mapping.
        """
        return self.mappings['item2id']

import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Sampler
class SequenceDataset(Dataset):
    def __init__(self, config, sequences, indexes):
        self.sequences = sequences
        self.indexes = indexes
        self.config = config
        self.index_to_samples = defaultdict(list)
        for i, index in enumerate(indexes):
            self.index_to_samples[index].append(i)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        index = self.indexes[idx]
        item_seq = seq[:-1]
        labels = seq[-1]
        seq_length = len(item_seq)
        padding_length = self.config['max_seq_length'] - len(item_seq)
        if padding_length > 0:
            item_seq = item_seq + [0] * padding_length  # 在后面填充0
        return {
            'item_seqs': torch.tensor(item_seq, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'seq_lengths': seq_length,
            'idx': index
        }


class NormalRecData:
    def __init__(self, config: dict):
        self.config = config

    def load_data(self):
        from pathlib import Path

        source_dict = {
            'A': 'Automotive',
            'B': 'Beauty',
            'C': 'CDs_and_Vinyl',
            'M': 'Movies_and_TV',
            'O': 'Sports_and_Outdoors',
            'T': 'Toys_and_Games',
            'S': 'steam'
        }
        self.config['source_dict'] = source_dict
        def read_data_from_file(domain, mode=''):
            # 基础路径和域名映射字典
            base_path = Path('data/')
            # 获取目标文件路径
            file_path = base_path / source_dict[domain] / '{}data.txt'.format(mode)
            with file_path.open('r') as file:
                item_seqs = [list(map(int, line.split())) for line in file]
            if mode == '':
                flat_list = [item for sublist in item_seqs for item in sublist]
                import numpy as np
                item_num = np.max(flat_list)
                return item_seqs, item_num
            else:
                return item_seqs

        total_item_num = 0
        all_data = []
        train_data = []
        valid_data = []
        test_data = []
        train_index = []
        valid_index = []
        test_index = []
        select_pool = None
        for index, key in enumerate(source_dict.keys()):
            if self.config.get('ab', None) == 'single':
                if key != self.config['sd']:
                    continue
            tmp_item_seqs, cur_item_num = read_data_from_file(key)
            all_data.extend([[item + total_item_num for item in sublist] for sublist in tmp_item_seqs])
            tmp_train_item_seqs, tmp_valid_item_seqs, tmp_test_item_seqs = (read_data_from_file(key, mode='train_'),
                                                                            read_data_from_file(key, mode='valid_'),
                                                                            read_data_from_file(key, mode='test_'))
            tmp_train_item_seqs = [[item + total_item_num for item in sublist] for sublist in tmp_train_item_seqs]
            tmp_valid_item_seqs = [[item + total_item_num for item in sublist] for sublist in tmp_valid_item_seqs]
            tmp_test_item_seqs = [[item + total_item_num for item in sublist] for sublist in tmp_test_item_seqs]

            tmp_train_idx = [index] * len(tmp_train_item_seqs)
            tmp_valid_idx = [index] * len(tmp_valid_item_seqs)
            tmp_test_idx = [index] * len(tmp_test_item_seqs)
            if key == self.config['td']:
                valid_data.extend(tmp_valid_item_seqs)
                valid_index.extend(tmp_valid_idx)
                test_data.extend(tmp_test_item_seqs)
                test_index.extend(tmp_test_idx)
                select_pool = [total_item_num + 1, total_item_num + cur_item_num + 1]
            if key in self.config['sd']:
                train_data.extend(tmp_train_item_seqs)
                train_index.extend(tmp_train_idx)
            total_item_num += cur_item_num
        return (SequenceDataset(self.config, train_data, train_index), SequenceDataset(self.config, valid_data, valid_index),
                SequenceDataset(self.config, test_data, test_index), select_pool, total_item_num)








# train_data[111]
# test_data[130]