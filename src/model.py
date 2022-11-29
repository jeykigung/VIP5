import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modeling_vip5 import VIP5

class VIP5Tuning(VIP5):
    def __init__(self, config): 
        super().__init__(config)

        self.losses = self.config.losses.split(',')

    def train_step(self, batch):
        device = next(self.parameters()).device
        task = batch["task"][0] # if every batch belongs to the same
            
        input_ids = batch['input_ids'].to(device)
        whole_word_ids = batch['whole_word_ids'].to(device)
        category_ids = batch['category_ids'].to(device)
        vis_feats = batch['vis_feats'].to(device)

        lm_labels = batch["target_ids"].to(device)

        loss_weights = batch["loss_weights"].to(device)

        output = self(
            input_ids=input_ids,
            whole_word_ids=whole_word_ids,
            category_ids=category_ids,
            vis_feats=vis_feats,
            labels=lm_labels,
            return_dict=True,
            task=task
        )
        assert 'loss' in output

        lm_mask = lm_labels != -100
        lm_mask = lm_mask.float()
        B, L = lm_labels.size()

        loss = output['loss']

        loss = loss.view(B, L) * lm_mask

        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)

        task_counts = {task: 0 for task in self.losses}
        task_loss = {task: 0 for task in self.losses}

        results = {}

        results['loss'] = (loss * loss_weights).mean()
        results['total_loss'] = loss.detach().sum()
        results['total_loss_count'] = len(loss)

        task_counts = {task: 0 for task in self.losses}
        task_loss = {task: 0 for task in self.losses}

        for _loss, task in zip(loss.detach(), batch['task']):
            task_loss[task] += _loss
            task_counts[task] += 1

        for task in self.losses:
            if task_counts[task] > 0:
                results[f'{task}_loss'] = task_loss[task]
                results[f'{task}_loss_count'] = task_counts[task]

        return results

    @torch.no_grad()
    def valid_step(self, batch):
        self.eval()
        device = next(self.parameters()).device
        task = batch["task"][0] # if every batch belongs to the same
        
        input_ids = batch['input_ids'].to(device)
        whole_word_ids = batch['whole_word_ids'].to(device)
        category_ids = batch['category_ids'].to(device)
        vis_feats = batch['vis_feats'].to(device)

        lm_labels = batch["target_ids"].to(device)

        loss_weights = batch["loss_weights"].to(device)

        output = self(
            input_ids=input_ids,
            whole_word_ids=whole_word_ids,
            category_ids=category_ids,
            vis_feats=vis_feats,
            labels=lm_labels,
            return_dict=True,
            task=task
        )
        assert 'loss' in output

        lm_mask = lm_labels != -100
        lm_mask = lm_mask.float()
        B, L = lm_labels.size()

        loss = output['loss']

        loss = loss.view(B, L) * lm_mask

        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)

        results = {}

        results['loss'] = (loss * loss_weights).mean()
        results['total_loss'] = loss.detach().sum()
        results['total_loss_count'] = len(loss)

        task_counts = {task: 0 for task in self.losses}
        task_loss = {task: 0 for task in self.losses}

        for _loss, task in zip(loss.detach(), batch['task']):
            task_loss[task] += _loss
            task_counts[task] += 1

        for task in self.losses:
            if task_counts[task] > 0:
                results[f'{task}_loss'] = task_loss[task]
                results[f'{task}_loss_count'] = task_counts[task]

        return results

    @torch.no_grad()
    def generate_step(self, batch):
        self.eval()
        device = next(self.parameters()).device
        task = batch["task"][0] # if every batch belongs to the same
            
        input_ids = batch['input_ids'].to(device)
        whole_word_ids = batch['whole_word_ids'].to(device)
        category_ids = batch['category_ids'].to(device)
        vis_feats = batch['vis_feats'].to(device)

        output = self.generate(
            input_ids=input_ids,
            whole_word_ids=whole_word_ids,
            category_ids=category_ids,
            vis_feats=vis_feats,
            task=task
        )

        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        return generated_sents
