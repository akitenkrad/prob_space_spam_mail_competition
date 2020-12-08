import warnings
from pathlib import Path
import yaml
from tqdm import tqdm
from datetime import datetime
import click
import torch
import csv
import numpy as np
from torch.utils.data import DataLoader
from utils import utils
from utils.datasets import SpamDataset, collate_fn
from models.bilstm import BiLSTM
from models.simple_transformer import TransformerClassification

warnings.filterwarnings('ignore')

def load_model(config, ds:SpamDataset):
    target_model = config['config']['model']
    if target_model == 'bilstm':
        embedding_dim = config['model']['bilstm']['embedding_dim']
        hidden_size = config['model']['bilstm']['hidden_size']
        grad_clip_th = config['model']['bilstm']['grad_clip_th']
        model = BiLSTM(ds.word_vocab.vocab_size, embedding_dim, hidden_size, grad_clip_th)
        
    elif target_model == 'simple_transformer':
        embedding_dim = config['model']['simple_transformer']['embedding_dim']
        d_model = config['model']['simple_transformer']['d_model']
        model = TransformerClassification(ds.word_vocab.vocab_size, embedding_dim, d_model, ds.max_length, output_dim=1)
        
    else:
        raise NotImplementedError("No such a model hasn't been implemented: {}".format(target_model))
    
    return model

@click.group()
def cli():
    pass

@cli.command()
@click.option('--config', type=click.Path(exists=True), help='path to test_config.yml')
def run_predict(config):
    
    # read config
    config = utils.read_config(config)
    utils.show_test_settings(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = config['config']['batch_size']
    weights_path = Path(config['config']['weights_path'])
    out_dir = Path(config['config']['out_dir'])
    
    # set seed
    utils.set_seed(config['config']['seed'])
    
    # load dataset
    data_path = config['config']['data_path']
    max_length = config['config']['max_sent_len']
    phase = config['config']['phase']
    dataset = SpamDataset(data_path, max_length=max_length, phase=phase)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # load model
    model = load_model(config, dataset)
    model = model.to(device)
    model = model.eval()
    model.load_state_dict(torch.load(str(weights_path), map_location=torch.device('cpu')))
    
    # ====== predict ===============================
    outputs = []
    for batch_idx, (idx, x, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
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
        out = out.clamp(min=0.0, max=1.0)
        out = out.round().detach().cpu().numpy()
        
        for i, pred in zip(idx, out):
            outputs.append({'id': int(i), 'y': int(pred)})

    outputs = sorted(outputs, key=lambda x: x['id'])
    out_file = out_dir / 'predicts.csv'
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['id', 'y'])
        for row in outputs:
            writer.writerow([row['id'], row['y']])
            
    print('Done!!! ->', str(out_file))
        
if __name__ == '__main__':
    cli()
    