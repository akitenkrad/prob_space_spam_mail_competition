import warnings
from pathlib import Path
import yaml
from configparser import ConfigParser
from tqdm import tqdm
from datetime import datetime
import click
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from utils import utils
from utils.datasets import SpamDataset, collate_fn
from models.bilstm import BiLSTM

warnings.filterwarnings('ignore')

def __read_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def __show_settings(config):
    fmt = '{:25s}: {}'
    print()
    print('<<< settings >>>')
    print('='*50)
    print(fmt.format('Epochs', config['config']['epochs']))
    print(fmt.format('Batch Size', config['config']['batch_size']))
    print(fmt.format('Model', config['config']['model']))
    print(fmt.format('Data Path', config['config']['data_path']))
    print(fmt.format('Max Sentence Length', config['config']['max_sent_len']))
    print(fmt.format('Validation Dataset Size', config['config']['valid_size']))
    print(fmt.format('Learning Rate', config['config']['learning_rate']))
    print(fmt.format('Save Directory', config['config']['save_dir']))
    print('='*50)
    print()
 
def __load_model(config, ds:SpamDataset):
    target_model = config['config']['model']
    if target_model == 'bilstm':
        embedding_dim = config['model']['bilstm']['embedding_dim']
        hidden_size = config['model']['bilstm']['hidden_size']
        model = BiLSTM(ds.word_vocab.vocab_size, embedding_dim, hidden_size)
        model.load_fasttext_embedding(config['model']['bilstm']['fasttext_weights'], ds.word_vocab)
        
    else:
        raise NotImplementedError("No such a model hasn't been implemented: {}".format(target_model))
    
    return model

@click.group()
def cli():
    pass

@cli.command()
@click.option('--config', type=click.Path(exists=True), help='path to config.ini')
def run_train(config):
    # read config
    config = __read_config(config)
    __show_settings(config)
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
    model = __load_model(config, dataset)
    model = model.to(device)
    
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

                    out = model(x)
                    out = out.squeeze()
                    
                    loss = criterion(out.type(torch.float32), y.type(torch.float32))
                    loss.backward()
                    optimizer.step()

                    batch_output = out.round().detach().cpu().numpy()
                    batch_y = y.cpu().numpy()
                    batch_loss = float(loss.detach().cpu().numpy())

                    # add logs into history
                    outputs = np.hstack([outputs, batch_output])
                    ys = np.hstack([ys, batch_y])
                    losses.append(batch_loss)

                    # update progress bar
                    train_pbar.update(1)
                    train_pbar.set_description('<Train> Epoch:{} Loss:{:.6f} Acc:{:.6f}'.format(
                            epoch + 1, batch_loss, accuracy_score(batch_output, batch_y)))

                history.add_train_value(epoch+1, outputs, ys, np.mean(losses))
                
 
            # ====== valid ================================
            outputs, ys = np.array([]), np.array([])
            with tqdm(total=len(valid_dl), leave=None) as valid_pbar:
                for idx, x, y, in valid_dl:
                    idx = idx.to(device)
                    x = x.to(device)
                    y = y.to(device)

                    out = model(x)
                    out = out.squeeze()

                    batch_output = out.round().detach().cpu().numpy()
                    batch_y = y.cpu().numpy()

                    valid_pbar.update(1)
                    valid_pbar.set_description('<Valid> Epoch:{} Loss:{:.6f} Acc:{:.6f}'.format(
                        epoch + 1, batch_loss, accuracy_score(batch_output, batch_y)))

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
    