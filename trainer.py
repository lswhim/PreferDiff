import os
import torch
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from base import AbstractModel
from transformers.optimization import get_scheduler
from collections import defaultdict, OrderedDict
from utils import get_file_name, get_total_steps
from evaluator import Evaluator


class BaseTrainer(object):
    def __init__(self, config: dict, model: AbstractModel):
        self.config = config
        self.model = model
        self.accelerator = config['accelerator']
        self.evaluator = Evaluator(config)
        self.saved_model_ckpt = os.path.join(
            self.config['ckpt_dir'],
            get_file_name(self.config, suffix='.pth')
        )
        os.makedirs(os.path.dirname(self.saved_model_ckpt), exist_ok=True)
        self.best_metric = 0
        self.best_epoch = 0
        self.count = 0

        self.checkpoints_deque = []

    def train(self, train_dataloader, val_dataloader):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay'],
        )
        total_n_steps = get_total_steps(self.config, train_dataloader)

        self.model, optimizer, train_dataloader, val_dataloader = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, val_dataloader)
        self.config.pop('accelerator')
        self.accelerator.init_trackers(
            project_name="PreferDiff",
            config=self.config
        )
        n_epochs = np.ceil(total_n_steps / (len(train_dataloader) * self.accelerator.num_processes)).astype(int)
        best_epoch = 0
        best_val_score = -1
        for epoch in range(n_epochs):
            # Training
            self.model.train()
            total_loss = 0.0
            train_progress_bar = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"Training - [Epoch {epoch + 1}]",
            )
            for batch in train_progress_bar:
                optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs['loss']
                self.accelerator.backward(loss)
                optimizer.step()
                # scheduler.step()
                total_loss = total_loss + loss.item()

            self.accelerator.log({"Loss/train_loss": total_loss / len(train_dataloader)}, step=epoch + 1)

            # Evaluation
            if (epoch + 1) % self.config['eval_interval'] == 0:
                all_results = self.evaluate(val_dataloader, split='val')
                if self.accelerator.is_main_process:
                    for key in all_results:
                        self.accelerator.log({f"Val_Metric/{key}": all_results[key]}, step=epoch + 1)
                    print(all_results)

                val_score = all_results[self.config['val_metric']]
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_epoch = epoch + 1
                    if self.accelerator.is_main_process:
                        if self.config['use_ddp']:  # unwrap model for saving
                            unwrapped_model = self.accelerator.unwrap_model(self.model)
                            torch.save(unwrapped_model.state_dict(), self.saved_model_ckpt)
                        else:
                            torch.save(self.model.state_dict(), self.saved_model_ckpt)
                        print(f'[Epoch {epoch + 1}] Saved model checkpoint to {self.saved_model_ckpt}')
                else:
                    print('Patience for {} Times'.format(epoch + 1 - best_epoch))

                if self.config['patience'] is not None and epoch + 1 - best_epoch >= self.config['patience']:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break
        print(f'Best epoch: {best_epoch}, Best val score: {best_val_score}')

    def evaluate(self, dataloader, split='test'):

        self.model.eval()

        all_results = defaultdict(list)
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval - {split}",
        )
        for batch in val_progress_bar:
            with torch.no_grad():
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                if self.config['use_ddp']:  # ddp, gather data from all devices for evaluation
                    preds = self.model.module.predict(batch, n_return_sequences=self.evaluator.maxk)
                    all_preds, all_labels = self.accelerator.gather_for_metrics((preds, batch['labels']))
                    results = self.evaluator.calculate_metrics(all_preds, all_labels)
                else:
                    preds = self.model.predict(batch, n_return_sequences=self.evaluator.maxk)
                    results = self.evaluator.calculate_metrics(preds, batch['labels'])

                for key, value in results.items():
                    all_results[key].append(value)


        output_results = OrderedDict()
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                key = f"{metric}@{k}"
                output_results[key] = torch.cat(all_results[key]).mean().item()
        return output_results

    def end(self):
        """
        Ends the training process and releases any used resources
        """
        self.accelerator.end_training()
