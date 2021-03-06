import warnings
from pathlib import Path
import yaml
from tqdm import tqdm
from datetime import datetime
import click
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from utils import utils
from utils.datasets import SpamDataset, collate_fn
from models.bilstm import BiLSTM
from models.simple_transformer import TransformerClassification

warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)

def load_model(config, ds:SpamDataset):
    target_model = config['config']['model']
    if target_model == 'bilstm':
        embedding_dim = config['model']['bilstm']['embedding_dim']
        hidden_size = config['model']['bilstm']['hidden_size']
        grad_clip_th = config['model']['bilstm']['grad_clip_th']
        model = BiLSTM(ds.word_vocab.vocab_size, embedding_dim, hidden_size, grad_clip_th)
        model.load_fasttext_embedding(config['model']['bilstm']['fasttext_weights'], ds.word_vocab)
        
    elif target_model == 'simple_transformer':
        embedding_dim = config['model']['simple_transformer']['embedding_dim']
        d_model = config['model']['simple_transformer']['d_model']
        model = TransformerClassification(ds.word_vocab.vocab_size, embedding_dim, d_model, ds.max_length, output_dim=1)
        model.load_fasttext_embedding(config['model']['bilstm']['fasttext_weights'], ds.word_vocab)
        
    else:
        raise NotImplementedError("No such a model hasn't been implemented: {}".format(target_model))
    
    return model

@click.group()
def cli():
    pass

@cli.command()
@click.option('--config', type=click.Path(exists=True), help='path to train_config.yml')
def run_train(config):
    # read config
    config = utils.read_config(config)
    utils.show_train_settings(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # set seed
    utils.set_seed(config['config']['seed'])
    
    # load dataset
    data_path = config['config']['data_path']
    max_length = config['config']['max_sent_len']
    valid_size = config['config']['valid_size']
    phase = config['config']['phase']
    dataset = SpamDataset(data_path, max_length=max_length, phase=phase)
    train_ds_size = int(len(dataset) * (1 - valid_size))
    valid_ds_size = len(dataset) - train_ds_size
    
    # load model
    model = load_model(config, dataset)
    model = model.to(device)
    model = model.train()
    
    # train settings
    epochs = config['config']['epochs']
    batch_size = config['config']['batch_size']
    learning_rate = config['config']['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss(reduction='mean')
    history = utils.History(config['config']['log_dir'])
    
    # weights dir
    save_dir = Path(config['config']['save_dir']) / datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
            
    # ====== epochs ===============================
    with tqdm(total=epochs, leave=True) as epoch_pbar:
        for epoch in range(epochs):
            train_ds, valid_ds = random_split(dataset, [train_ds_size, valid_ds_size])
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

            # ====== train ============================
            outputs, ys, losses = np.array([]), np.array([]), []
            with tqdm(total=len(train_dl), leave=None) as train_pbar:
                for idx, x, y in train_dl:
                    idx = idx.to(device)
                    x = x.to(device)
                    y = y.to(device)

                    optimizer.zero_grad()
                    
                    if config['config']['model'] == 'simple_transformer':
                        input_pad = dataset.word_vocab.token2idx('<pad>')
                        input_mask = (x != input_pad)
                        out, norm_weights_1, norm_weights_2 = model(x, input_mask)
                    else:
                        out = model(x)
                        
                    out = out.squeeze()
                    out = out.clamp(min=0.0, max=1.0)
                    
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()

                    batch_output = out.round().detach().cpu().numpy()
                    batch_y = y.cpu().numpy()
                    batch_loss = float(loss.detach().cpu().numpy())
                        
                    # add logs into history
                    outputs = np.hstack([outputs, batch_output])
                    ys = np.hstack([ys, batch_y])
                    losses.append(batch_loss)

                    cm = confusion_matrix(batch_y, batch_output).flatten()
                    if len(cm) == 1:
                        tn, fp, fn, tp = int(len(cm) / 2.0), 0, 0, int(len(cm) / 2.0)
                    else:
                        tn, fp, fn, tp = cm
                        
                    # update progress bar
                    train_pbar.update(1)
                    train_pbar.set_description('<Train> Epoch:{} Loss:{:.3f} Acc:{:.3f} F1:{:.3f} P:{:.3f} R:{:.3f} (TP,FP,TN,FN):({},{},{},{})'.format(
                            epoch + 1, batch_loss,
                            accuracy_score(batch_y, batch_output), f1_score(batch_y, batch_output),
                            precision_score(batch_y, batch_output), recall_score(batch_y, batch_output),
                            tp, fp, tn, fn))

                history.add_train_value(epoch+1, outputs, ys, np.mean(losses))
                
 
            # ====== valid ================================
            outputs, ys = np.array([]), np.array([])
            with tqdm(total=len(valid_dl), leave=None) as valid_pbar:
                for idx, x, y, in valid_dl:
                    idx = idx.to(device)
                    x = x.to(device)
                    y = y.to(device)

                    if config['config']['model'] == 'simple_transformer':
                        input_pad = dataset.word_vocab.token2idx('<pad>')
                        input_mask = (x != input_pad)
                        out, norm_weights_1, norm_weights_2 = model(x, input_mask)
                    else:
                        out = model(x)
                        
                    out = out.squeeze()

                    batch_output = out.round().detach().cpu().numpy()
                    batch_y = y.cpu().numpy()

                    cm = confusion_matrix(batch_y, batch_output).flatten()
                    if len(cm) == 1:
                        tn, fp, fn, tp = int(len(cm) / 2.0), 0, 0, int(len(cm) / 2.0)
                    else:
                        tn, fp, fn, tp = cm
                        
                    valid_pbar.update(1)
                    valid_pbar.set_description('<Valid> Epoch:{} Loss:{:.3f} Acc:{:.3f} F1:{:.3f} P:{:.3f} R:{:.3f} (TP,FP,TN,FN):({},{},{},{})'.format(
                        epoch + 1, batch_loss,
                        accuracy_score(batch_y, batch_output), f1_score(batch_y, batch_output),
                        precision_score(batch_y, batch_output), recall_score(batch_y, batch_output),
                        tp, fp, tn, fn))

                    outputs = np.hstack([outputs, batch_output])
                    ys = np.hstack([ys, batch_y])

                history.add_test_value(epoch + 1, outputs, ys)

            # ====== save weights ========================
            epoch_pbar.update(1)
            epoch_pbar.set_description(history.description())

            save_dir.mkdir(parents=True, exist_ok=True)
            
            if history.is_best:
                best_path = save_dir / 'best.pth'
                torch.save(model.state_dict(), str(best_path))

                checkpoint = save_dir / 'checkpoint_best.pth'
                torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': history.loss,
                  }, checkpoint)

            if (epoch + 1) % 20 == 0:
                last_path = save_dir / 'last_at_{}.pth'.format(epoch + 1)
                torch.save(model.state_dict(), last_path)

            # save checkpoint
            checkpoint = save_dir / 'checkpoint.pth'
            torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': history.loss,
              }, checkpoint)
            
if __name__ == '__main__':
    cli()
    