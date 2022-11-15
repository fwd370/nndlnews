import pandas as pd
import numpy as np
import re
import contractions
import json
import nltk
nltk.download('punkt')
import random
import time
import datetime
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
from rouge import Rouge
import argparse
import json

USE_WANDB = True
BOS_token = 0
EOS_token = 1
UNK_token = 2
PAD_token = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Vocabulary:
    def __init__(self):
        self.word2index = {"BOS":0,"EOS":1,"UNK":2,"PAD":3}
        self.word2count = {}
        self.index2word = {0:"BOS",1:"EOS",2:"UNK",3:"PAD"}
        self.n_words = 4 #Start count with "SOS", "EOS", UNK, PAD

    def addToVocab(self, sentence: str):
        for word in sentence.split(" "):
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words+= 1
                self.word2count[word] = 1
            else:
                self.word2count[word] += 1

    def convertSentenceToIndex(self, sentence: str):
        idxs = [self.word2index[word] for word in sentence.split(" ")]
        idxs.append(EOS_token)
        return idxs
        #return torch.tensor(idxs, dtype=torch.long, device=device).view(-1,1)

def processString(text: str): 
    text = re.sub("\\\\n","",text)                  # Remove Line breaks in the article 
    text = re.sub("SINGAPORE","",text)              # Unique processing for Today/CNA articles 
    text = contractions.fix(text)                   # Replaces apostrophes with full word to reduce dimensionality 
    text = re.sub(r"([.!?])",r" \1", text) 
    text = re.sub(r"[^a-zA-Z0-9.!?]+", r" ", text) 
    text = text.lower().strip() 
    text = re.sub(r"[0-9]+","N",text) 
    return text 

def read_data(n_truncate: int, file='rawData.csv'):
    summarys = []
    contents = []
    MAX_LENGTH = 300 # We limit the amount of words being fed into model
    vocab = Vocabulary()
    df = pd.read_csv('../../data/'+file)
    print(f"Total number of data rows (articles): {len(df.index)}")

    for index,row in df.iterrows():
        text = row["Content"]
        title = processString(row["Title"])

        sent_list = nltk.tokenize.sent_tokenize(text)
        num_of_sentences = len(sent_list)

        end_boundary = n_truncate if n_truncate < num_of_sentences else num_of_sentences
        first_sentences = sent_list[0:end_boundary]
        text = processString(''.join(first_sentences))
        if (len(text.split(' ')) < MAX_LENGTH and len(title.split(' ')) < MAX_LENGTH):
            vocab.addToVocab(text)
            vocab.addToVocab(title)
        
        contents.append(vocab.convertSentenceToIndex(text))
        summarys.append(vocab.convertSentenceToIndex(title))

    return contents, summarys, vocab.word2index, vocab.index2word



class AINLPEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim).to(device)

        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout).to(device)

        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, src):

        embedded = self.dropout(self.embedding(src))

        outputs,  hidden = self.rnn(embedded)

        return outputs, hidden

class AINLPDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim).to(device)

        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout).to(device)

        self.fc_out = nn.Linear(hid_dim, output_dim).to(device)

        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, input, hidden):

        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        output, hidden = self.rnn(embedded, hidden)

        prediction = self.fc_out(output.squeeze(0))

        return prediction, hidden

class AINLPSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size,
                              trg_vocab_size).to(self.device)

        output, hidden = self.encoder(src)

        input = trg[0, :]

        for t in range(1, trg_len):

            # Get the output from decoder
            output, hidden = self.decoder(input, hidden)

            # Repalce the output at time t
            outputs[t] = output

            # Teacher Forcing
            teacher_force = random.random() < teacher_forcing_ratio

            # Get the output with maximal probability
            top1 = output.argmax(1)

            # The new one as the input of next step
            input = trg[t] if teacher_force else top1

        return outputs

class CuDataset(Dataset):
    def __init__(self, sources, targets):
        self.sources = sources
        self.targets = targets

    def __getitem__(self, idx):
        idx = int(idx)
        item = {}
        item['src'] = torch.tensor(self.sources[idx])
        item['tgt'] = torch.tensor(self.targets[idx])
        return item

    def __len__(self):
        return len(self.sources)

def init_model(word2index,device,HID_DIM,N_LAYERS,ENC_EMB_DIM,DEC_EMB_DIM,ENC_DROPOUT,DEC_DROPOUT):
    INPUT_DIM = len(word2index)
    OUTPUT_DIM = len(word2index)

    enc = AINLPEncoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = AINLPDecoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = AINLPSeq2Seq(enc, dec, device).to(device)
    return model

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def pad_data(contents,summarys,word2index):
    max_src = min(max(map(len, contents)), 512)
    max_tgt = min(max(map(len, summarys)), 128)
    sources = []
    targets = []
    for i in contents:
        if len(i) < max_src:
            i += [word2index['PAD']] * (max_src - len(i))
        sources.append(i[:max_src])

    max_tgt += 2
    for i in summarys:
        if len(i) < max_tgt:
            i = [word2index['BOS']] + i + [word2index['EOS']]
            i += [word2index['PAD']] * (max_src - len(i))
        targets.append(i[:max_tgt])
    return sources, targets

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model, iterator, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):

        src = batch['src'].transpose(0, 1).to(device)
        trg = batch['tgt'].transpose(0, 1).to(device)

        optimizer.zero_grad()

        input_tgt = trg[:-1].to(device)
        label_tgt = trg[1:].to(device)

        output = model(src, input_tgt)

        output_dim = output.shape[-1]

        output = output.view(-1, output_dim)
        trg = label_tgt.contiguous().view(-1)

        loss = criterion(output, trg)
        
        if i%50==0:
            print(f'[Step] Train Loss: {loss:.3f} | Train PPL: {math.exp(loss):7.3f}')

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

@torch.no_grad()
def evaluate(model, dataloader, criterion):

    model.eval()

    epoch_loss = 0
    dataloader
    with torch.no_grad():

        for i, batch in enumerate(dataloader):

            src = batch['src'].transpose(0, 1).to(device)
            trg = batch['tgt'].transpose(0, 1).to(device)

            input_tgt = trg[:-1]
            label_tgt = trg[1:]
            output = model(src, input_tgt, 0)  #turn off teacher forcing

            output_dim = output.shape[-1]

            output = output.view(-1, output_dim)
            label_tgt = label_tgt.contiguous().view(-1)

            loss = criterion(output, label_tgt)

            print(
                f'[Step] Train Loss: {loss:.3f} | Train PPL: {math.exp(loss):7.3f}'
            )

            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

@torch.no_grad()
def generate(model, dataloader, max_len=50):

    model.eval()

    all_output = []

    for batch in dataloader:

        src = batch['src'].transpose(0, 1)
        trg = batch['tgt'].transpose(0, 1)

        batch_size = trg.shape[1]

        output, hidden = model.encoder(src)

        trg_indexes = [word2index['BOS']] * batch_size

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        outputs = torch.zeros(max_len, batch_size, dtype=torch.int)

        input = trg_tensor[0, :]

        for i in range(max_len):
            output, hidden = model.decoder(input, hidden)

            pred_token = output.argmax(1)
            outputs[i] = pred_token
            input = torch.LongTensor(pred_token).to(device)

        all_output.append(outputs.transpose(0, 1).numpy())

    return np.concatenate(all_output, axis=0)

def compute_rouge(decoded_preds, decoded_labels):
    rouge = Rouge()
    scores = []
    for i, j in zip(decoded_preds, decoded_labels):
        score = rouge.get_scores(' '.join(i), ' '.join(j))
        scores.append([
            score[0]['rouge-1']['f'], score[0]['rouge-2']['f'],
            score[0]['rouge-l']['f']
        ])
    rouge1 = np.mean([i[0] for i in scores])
    rouge2 = np.mean([i[1] for i in scores])
    rougel = np.mean([i[2] for i in scores])
    result = {
        "rouge1": round(rouge1, 4),
        "rouge2": round(rouge2, 4),
        "rouge-l": round(rougel, 4)
    }
    return result


def main(LR,N_EPOCHS, runID,HID_DIM,N_LAYERS,ENC_EMB_DIM, DEC_EMB_DIM,ENC_DROPOUT,DEC_DROPOUT,BATCH_SIZE,CLIP,SEED,updateLR ):
    with open('curr_valid_loss.json','r') as jfile:
        curr_data=json.load(jfile)
    jfile.close()
    best_valid_loss = float(curr_data['valid_loss'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    
    trainFlag = True
    generateFlag = False
    
    contents, summarys, word2index, index2word = read_data(3)
    sources, targets = pad_data(contents, summarys, word2index)
    train_sources, test_sources, train_targets, test_targets = train_test_split(sources, targets, test_size=0.1, random_state=43)
    train_dataset = CuDataset(train_sources, train_targets)
    val_dataset = CuDataset(test_sources, test_targets)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = init_model(word2index, device,HID_DIM,N_LAYERS,ENC_EMB_DIM,DEC_EMB_DIM,ENC_DROPOUT,DEC_DROPOUT)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(),lr=LR)

    criterion = nn.CrossEntropyLoss(ignore_index=word2index['PAD'])
    if trainFlag:
        wallStart = datetime.datetime.now().replace(microsecond=0)
        for epoch in range(N_EPOCHS):
            if updateLR:
                updatedLR = LR - epoch*(LR/N_EPOCHS)
                for grp in optim.param_groups:
                    grp['lr'] = updatedLR
            start_time = time.time()
            train_loss = train(model, train_loader, optimizer, criterion, CLIP)
            valid_loss = evaluate(model, eval_loader, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if USE_WANDB:
                wandb.log({'epoch':epoch,'learning-rate':LR, 'train-loss':train_loss, 'train-ppl':math.exp(train_loss),'valid-loss':valid_loss, 'valid-ppl':math.exp(valid_loss)})
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(
                f'[Epoch] Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}'
            )
            print(
                f'[Epoch] Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}'
            )
        curr_data['runtime']= str(datetime.datetime.now().replace(microsecond=0)-wallStart)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'preprocessed_gru_model.pt')
            curr_data['valid_loss'] = f'{best_valid_loss:.3f}'
            curr_data['valid_ppl'] = f'{math.exp(best_valid_loss):7.3f}'
            curr_data['runID'] = runID
            with open('curr_valid_loss.json','w') as jfile:
                json.dump(curr_data,jfile,indent=6)
            jfile.close()
        
        with open('training_results.json','r+') as outfile:
            filedata=json.load(outfile)
            filedata['run_details'].append(curr_data)
            outfile.seek(0)
            json.dump(filedata, outfile, indent=4)
        outfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tuning GRU")
    parser.add_argument('--lr',type=float,default=0.001, required=False)
    parser.add_argument('--epoch',type=int,default=10, required=False)
    parser.add_argument('--run',type=int, required=True)
    parser.add_argument('--h-dim',type=int,default=512, required=False)
    parser.add_argument('--n-layers',type=int,default=2, required=False)
    parser.add_argument('--enc-dim',type=int,default=256, required=False)
    parser.add_argument('--dec-dim',type=int,default=256, required=False)
    parser.add_argument('--enc-dropout',type=float,default=0.5, required=False)
    parser.add_argument('--dec-dropout',type=float,default=0.5, required=False)
    parser.add_argument('--bs',type=int,default=2,required=False)
    parser.add_argument('--clip',type=float,default=1.0,required=False)
    parser.add_argument('--seed',type=int,default=1234,required=False)
    parser.add_argument('--update-lr',type=bool, default=False, required=False)

    args = parser.parse_args()
    kwargs ={'LR': args.lr,
            'N_EPOCHS': args.epoch,
            'runID': args.run,
            'HID_DIM': args.h_dim,
            'N_LAYERS': args.n_layers,
            'ENC_EMB_DIM': args.enc_dim,
            'DEC_EMB_DIM': args.dec_dim,
            'ENC_DROPOUT': args.enc_dropout,
            'DEC_DROPOUT': args.dec_dropout,
            'BATCH_SIZE': args.bs,
            'CLIP': args.clip,
            'SEED': args.seed,
            'updateLR':args.update_lr
            }
    if USE_WANDB:
        import wandb
        wandb.init(project='nndl-news', config={'algo':'GRU_PP','direction':'Uni',**kwargs})
    main(**kwargs)
