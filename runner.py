import torch
from typing import Union

from accelerate import Accelerator
from torch.utils.data import DataLoader, Sampler

from recdata import AbstractRecData, NormalRecData
from base import AbstractModel

from utils import get_config, init_device, init_seed, get_model, get_mapper
from trainer import BaseTrainer


class Runner:
    def __init__(
            self,
            model_name: Union[str, AbstractModel],
            config_dict: dict = None,
            config_file: str = None,
    ):
        self.config = get_config(
            model_name=model_name,
            config_file=config_file,
            config_dict=config_dict
        )

        # Automatically set devices and ddp
        self.config['device'], self.config['use_ddp'] = init_device()
        self.accelerator = Accelerator(log_with='wandb')

        self.config['accelerator'] = self.accelerator

        init_seed(self.config['rand_seed'], self.config['reproducibility'])
        _ = NormalRecData(self.config).load_data()

        self.recdata = {
            'train': _[0],
            'valid': _[1],
            'test': _[2]
        }
        self.config['select_pool'] = _[3]
        self.config['item_num'] = _[4]
        self.config['eos_token'] = _[4] + 1


        with self.accelerator.main_process_first():
            self.model = get_model(model_name)(self.config)

        print(self.model)
        print(self.model.n_parameters)
        self.trainer = BaseTrainer(self.config, self.model)

    def run(self):
        import random
        class IndexSampler(Sampler):
            def __init__(self, dataset, batch_size):
                self.dataset = dataset
                self.batch_size = batch_size
                self.unique_indexes = list(set(self.dataset.index_to_samples.keys()))  # 使用 set 去重

            def __iter__(self):
                while True:
                    chosen_index = random.choice(self.unique_indexes)
                    sample_indices = self.dataset.index_to_samples[chosen_index]
                    if len(sample_indices) < self.batch_size:
                        sample_indices = random.choices(sample_indices, k=self.batch_size)
                    else:
                        sample_indices = random.sample(sample_indices, self.batch_size)
                    yield sample_indices

            def __len__(self):
                return len(self.dataset) // self.batch_size

        # sampler = IndexSampler(self.recdata['train'], self.config['train_batch_size'])
        # if len(self.config['sd']) > 1:
        #     train_dataloader = DataLoader(
        #         self.recdata['train'],
        #         sampler=sampler
        #     )
        # else:
        train_dataloader = DataLoader(
                self.recdata['train'],
                batch_size=self.config['train_batch_size'],
                shuffle=True,
            )
        val_dataloader = DataLoader(
            self.recdata['valid'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
        )
        test_dataloader = DataLoader(
            self.recdata['test'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
        )
        self.trainer.train(train_dataloader, val_dataloader)

        self.accelerator.wait_for_everyone()
        self.model = self.accelerator.unwrap_model(self.model)

        if self.config.get('steps', None) != 0:
            self.model.load_state_dict(torch.load(self.trainer.saved_model_ckpt))
        else:
            pass

        self.model, test_dataloader = self.accelerator.prepare(
            self.model, test_dataloader
        )
        if self.accelerator.is_main_process:
            print(f'Loaded best model checkpoint from {self.trainer.saved_model_ckpt}')

        if self.config.get('step', None) != 0:
            test_results = self.trainer.evaluate(test_dataloader)
            if self.accelerator.is_main_process:
                for key in test_results:
                    self.accelerator.log({f'Test_Metric/{key}': test_results[key]})

        import numpy as np
        if self.config['exp_type'] == 'check':
            np.save('{}_{}_vis_embeddings.npy'.format(self.config['model'], self.config['sd']),
                    np.array(self.model.samples))
            np.save('{}_{}_pred_embeddings.npy'.format(self.config['model'], self.config['sd']),
                    np.array(self.model.predict_embeddings.detach().cpu().numpy()))
            np.save('{}_{}_target_embeddings.npy'.format(self.config['model'], self.config['sd']),
                    np.array(self.model.target_embedding.detach().cpu().numpy()))
        if self.accelerator.is_main_process:
            if self.config['save'] is False:
                import os
                if os.path.exists(self.trainer.saved_model_ckpt):
                    os.remove(self.trainer.saved_model_ckpt)
                    print(f"{self.trainer.saved_model_ckpt} has been deleted.")
                else:
                    print(f"{self.trainer.saved_model_ckpt} not found.")
        self.trainer.end()

