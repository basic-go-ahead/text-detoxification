from typing import Any, Dict, Iterable, Union

import torch
from tqdm import tqdm


def forward(model, batch):
    source_ids = batch['source_ids'].to(model.device, dtype=torch.long)
    source_mask = batch['source_mask'].to(model.device, dtype=torch.long)
    
    target_ids = batch['target_ids']
    target_ids[target_ids[:, :] == 0] = -100
    target_ids = target_ids.to(model.device, dtype=torch.long)

    return model(input_ids=source_ids, attention_mask=source_mask, labels=target_ids).loss


def train(current_epoch: int, model, loader, optimizer, params: Dict[str, Any]):
    model.train()
    total_loss = 0.
    
    with tqdm(loader) as progress_bar:
        for batch_index, batch in enumerate(progress_bar, 1):
            loss = forward(model, batch)

            optimizer.zero_grad()
            loss.backward()

            if 'MAX_GRAD_NORM' in params:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params['MAX_GRAD_NORM'])

            optimizer.step()

            total_loss += loss.item()

            if batch_index % 10 == 0:
                progress_bar.set_description(f'Epoch {current_epoch:03d}: mean_train_loss = {total_loss / batch_index:.05f}')


def validate(current_epoch, model, loader):
    model.eval()
    total_loss = 0.
    
    with torch.no_grad(), tqdm(loader) as progress_bar:
        for batch_index, batch in enumerate(progress_bar, 1):
            loss = forward(model, batch)
            total_loss += loss.item()
            
            if batch_index % 10 == 0:
                progress_bar.set_description(f'Epoch {current_epoch:02d}: mean_valid_loss = {total_loss / batch_index:.05f}')
        
        valid_loss = total_loss / batch_index
        progress_bar.set_description(f'Epoch {current_epoch:02d}: mean_valid_loss = {valid_loss:.05f}')
        
    return valid_loss

    
def run_finetuning(model,
    model_params: Dict[str, Any],
    optimizer,
    train_loader,
    n_epochs: int,
    save_model_when: Union[Iterable[int], None]=None):
    for current_epoch in range(1, n_epochs + 1):
        train(current_epoch, model, train_loader, optimizer, model_params)
        if current_epoch in save_model_when:        
            torch.save(model, f'./model-{current_epoch:02d}.dump')
