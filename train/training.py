import os

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from tqdm import tqdm


def batch_pearsonr_gpu(preds, labels):
    preds_mean = preds.mean(dim=0, keepdim=True)
    labels_mean = labels.mean(dim=0, keepdim=True)

    preds_centered = preds - preds_mean
    labels_centered = labels - labels_mean

    numerator = (preds_centered * labels_centered).sum(dim=0)
    denominator = torch.sqrt((preds_centered ** 2).sum(dim=0) * (labels_centered ** 2).sum(dim=0))

    corr = numerator / (denominator + 1e-8)
    corr = torch.nan_to_num(corr, nan=0.0)

    return corr.mean().item()


def train(model, dataloaders, optimizer, num_epochs=200, run=None, split=0,
          save_on='loss', stop_on='loss', delta=0.5, save_dir='', phases=['train', 'val'],
          use_amp=True, loss_cfg: dict | None = None, patience=20):
    device = model.device
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None
    _ = loss_cfg

    best_loss = float('inf')
    best_corr = -float('inf')
    patience_counter = 0
    patience_counter_loss = 0
    corr_at_patience_start = -float('inf')
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 40)

        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            seen = 0
            all_preds = []
            all_labels = []

            dataloader = dataloaders[phase]

            for x, y, _, _ in tqdm(dataloader, desc=f'{phase} phase'):
                x = x.float().to(device)
                y = y.float().to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    if scaler is not None and phase == 'train':
                        with torch.amp.autocast('cuda'):
                            outputs = model(x)
                            loss = loss_fn(outputs, y)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(x)
                        loss = loss_fn(outputs, y)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                running_loss += loss.item() * x.size(0)
                seen += int(x.size(0))
                all_preds.append(outputs.detach())
                all_labels.append(y.detach())

            denom = float(max(seen, 1))
            epoch_loss = running_loss / denom

            preds = torch.cat(all_preds, dim=0)
            labels = torch.cat(all_labels, dim=0)
            avg_corr = batch_pearsonr_gpu(preds, labels)

            print(f'{phase} Loss(MSE): {epoch_loss:.4f} | Correlation: {avg_corr:.4f}')

            if run is not None:
                run.log({
                    f'{phase}_loss_fold_{split}': epoch_loss,
                    f'{phase}_corr_fold_{split}': avg_corr,
                    'epoch': epoch,
                })

            if phase != 'val':
                continue

            is_better = False
            loss_improved = False

            if save_on == 'loss':
                if epoch_loss < best_loss:
                    is_better = True
                    loss_improved = True
                    best_loss = epoch_loss
            elif save_on == 'corr':
                if avg_corr > best_corr:
                    is_better = True
                    best_corr = avg_corr
            elif save_on == 'loss+corr':
                if epoch_loss < best_loss:
                    is_better = True
                    loss_improved = True
                    best_loss = epoch_loss
                    best_corr = avg_corr
                elif avg_corr > best_corr:
                    is_better = True
                    best_corr = avg_corr

            if loss_improved:
                patience_counter_loss = 0
                corr_at_patience_start = avg_corr
            else:
                patience_counter_loss += 1

            if is_better:
                patience_counter = 0
                if save_dir:
                    suff = f'_{split}' if split > 0 else ''
                    save_path = os.path.join(save_dir, f'model_best{suff}.pt')
                    torch.save(model.state_dict(), save_path)
                    print(f'Saved best model to {save_path}')
            else:
                patience_counter += 1

            if stop_on == 'loss':
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break
            elif stop_on == 'corr':
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break
            elif stop_on == 'loss+corr':
                if patience_counter_loss >= patience:
                    corr_improved = avg_corr > corr_at_patience_start
                    loss_reasonable = epoch_loss < (best_loss + delta)

                    if corr_improved and loss_reasonable:
                        print(f'Epoch {epoch + 1}: MSE plateau but correlation improved, continuing...')
                        patience_counter_loss = 0
                        corr_at_patience_start = avg_corr
                    else:
                        print(f'Early stopping at epoch {epoch + 1}')
                        print(f'  Corr improved: {corr_improved}, Loss reasonable: {loss_reasonable}')
                        break

        if patience_counter >= patience:
            break

    if save_dir and os.path.exists(save_dir):
        suff = f'_{split}' if split > 0 else ''
        best_model_path = os.path.join(save_dir, f'model_best{suff}.pt')
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            print(f'Loaded best model from {best_model_path}')

    return model


def evaluate(model, dataloader, run=None, suff=''):
    device = model.device
    model.eval()

    all_preds = []
    all_labels = []
    all_wsi_names = []
    all_projects = []

    with torch.no_grad():
        for x, y, wsi_names, projects in tqdm(dataloader, desc='Evaluating'):
            x = x.float().to(device)

            outputs = model(x)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(y.numpy())
            all_wsi_names.extend(wsi_names)
            all_projects.extend(projects)

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    wsi_names = np.array(all_wsi_names)
    projects = np.array(all_projects)

    loss_fn = nn.MSELoss()
    test_loss = loss_fn(torch.from_numpy(preds), torch.from_numpy(labels)).item()

    correlations = []
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) > 1 and len(np.unique(preds[:, i])) > 1:
            corr, _ = pearsonr(labels[:, i], preds[:, i])
            if not np.isnan(corr):
                correlations.append(corr)
    avg_corr = np.mean(correlations) if correlations else 0.0

    print(f'Test Loss: {test_loss:.4f} | Test Correlation: {avg_corr:.4f}')

    if run is not None:
        run.log({
            f'test_loss{suff}': test_loss,
            f'test_corr{suff}': avg_corr,
        })

    return preds, labels, wsi_names, projects
