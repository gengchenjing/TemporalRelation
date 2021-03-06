import glob
import re
import os
import unicodedata
import string
import sys
import codecs
import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from io import open
from sklearn.metrics import f1_score
from sklearn import datasets, linear_model
from gensim.models import KeyedVectors
import matplotlib

matplotlib.use('PS')
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from gensim.models import KeyedVectors
from torch.autograd import Variable
from sklearn.model_selection import train_test_split, GroupKFold, GroupShuffleSplit, KFold
from sklearn.metrics import accuracy_score
import itertools
from transformers import BertTokenizer, BertModel
from bert_process_functions import batched_index_select

print(torch.nn.utils.clip_grad_norm_)
print("cudnn version", torch.backends.cudnn.version())

"""
def get_pretrained_embed_vector():
    # model_dir = 'entity_vector/entity_vector.model.bin'
    print("pre_vector start...")
    model_dir = 'nwjc_word_skip_300_8_25_0_1e4_6_1_0_15.txt.vec'
    pre_embed = KeyedVectors.load_word2vec_format(model_dir)
    pre_vector = pre_embed.vectors
    print("pre_vector done...")

    word2ix = {'UNK': 0}
    for word, value in pre_embed.vocab.items():
        word2ix[word] = value.index + 1

    pre_vector = np.insert(pre_vector, 0, np.zeros(pre_vector.shape[-1]), 0)
    return word2ix, pre_vector


word2ix, pre_vector = get_pretrained_embed_vector()
"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

bert_model = BertModel.from_pretrained('./pytorch_model/', output_hidden_states=True)
bert_model = bert_model.to(device)
tokenizer = BertTokenizer.from_pretrained('./pytorch_model/')

for name, param in bert_model.named_parameters():
    param.requires_grad = False


def unicodeToUtf8(s):
    return s.encode('utf-8')


def readlines(filename):
    f = codecs.open(filename, 'r', 'utf-8')
    lines = f.read().strip().split('\n')
    f.close()
    return [unicodeToUtf8(line) for line in lines if 'reltype' in line]


def read_multi_files(files_namelist):
    files_list = []
    for filename in files_namelist:
        each_file_line_list = readlines(filename)
        files_list += each_file_line_list
    # return [unicodeToUtf8(line) for line in lines if 'reltype' in line]
    return files_list


def doc_kfold(data_dir, cv=5):
    file_list, file_splits = [], []
    for file in sorted(os.listdir(data_dir)):
        if file.endswith(".txt"):
            dir_file = os.path.join(data_dir, file)
            file_list.append(dir_file)
    #    logger.info("[Number] %i files in '%s'" % (len(file_list), data_dir))
    gss = KFold(n_splits=cv, shuffle=True, random_state=1029)
    for train_split, test_split in gss.split(file_list):
        file_splits.append((
            [file_list[fid] for fid in train_split],
            [file_list[fid] for fid in test_split]
        ))
    return file_splits


def padding_2d(seq_2d, max_len, pad_tok=0, direct='right'):
    for seq_1d in seq_2d:
        for i in range(0, max_len - len(seq_1d)):
            if direct in ['right']:
                seq_1d.append(pad_tok)
            else:
                seq_1d.insert(0, pad_tok)
    return seq_2d


def getSentAndCategory(datalist):
    global sent_list1
    global sent_list2
    global category_list
    sent_list1 = []
    sent_list2 = []

    category_list = []
    for data in datalist:
        #        print(data[2])
        # sent_list.append((data[3],data[5]))
        sent_list1.append((data[5], data[6]))
        sent_list2.append((data[12], data[13]))
        category_list.append(data[1])
    n_categories = len(set(category_list))

    return (sent_list1,sent_list2, category_list, n_categories)


def process_data(train_path_list, test_path_list):
    datalist_train = [re.split('\t|\|', line.decode('utf-8')) for line in read_multi_files(train_path_list)]
    datalist_test = [re.split('\t|\|', line.decode('utf-8')) for line in read_multi_files(test_path_list)]

    sent_list1_train, sent_list2_train, category_list_train, n_train_categories = getSentAndCategory(datalist_train)
    sent_list1_test, sent_list2_test, category_list_test, n_test_categories = getSentAndCategory(datalist_test)
    sent_list1_all, sent_list2_all, category_list, _ = getSentAndCategory(datalist_train + datalist_test)
    sent2category = {}
    for i in range(len(sent_list1)):
        sent2category[sent_list1[i]] = category_list[i]

    x1_train = np.array(sent_list1_train)
    x2_train = np.array(sent_list2_train)
    y_train = np.array(category_list_train)
    x1_test = np.array(sent_list1_test)
    x2_test = np.array(sent_list2_test)
    y_test = np.array(category_list_test)
    #  count_text_length( x_train, x_test )

    # cat2ix, word2ix, pre_vector = get_pre_embed_vector(  )
    cat2ix = get_cat2ix()
    # doc_tensor_dict = get_doc_2_tensor(cat2ix, x_train, y_train, x_test, y_test)
    doc_tensor_dict = get_sequence_2_TensorDataset(cat2ix, x1_train, x2_train,y_train, x1_test,x2_test, y_test)
    # doc_tensor_dict = get_doc_3_tensor(cat2ix, x_train, y_train, x_test, y_test)

    return cat2ix, doc_tensor_dict


def get_cat2ix():
    cat2ix = {}
    for cat in category_list:
        if cat not in cat2ix:
            cat2ix[cat] = len(cat2ix)
    print(cat2ix)
    return cat2ix


def seg_sent(x_data):
    seg_list = []
    for line in x_data:
        toks = mt.parse(line.strip()).splitlines()
        seg_list.append([tok.split('\t')[0] for tok in toks][:-1])
    return seg_list


def get_doc_2_tensor(cat2ix, x1_train,x2_train, y_train, x1_test,x2_test, y_test):
    global max_len1
    global max_len2
    tok_num1 = sum([len(line_tok) for line_tok in seg_sent(sent_list1)])

    max_len1 = max([len(line_tok) for line_tok in seg_sent(sent_list1)])

    unk_count1 = 0
    for line_tok in seg_sent(sent_list1):
        for tok in line_tok:
            if tok not in word2ix:
                unk_count1 += 1
    tok_num2 = sum([len(line_tok) for line_tok in seg_sent(sent_list2)])

    max_len2 = max([len(line_tok) for line_tok in seg_sent(sent_list2)])

    unk_count2 = 0
    for line_tok in seg_sent(sent_list2):
        for tok in line_tok:
            if tok not in word2ix:
                unk_count2 += 1
    print(unk_count1, tok_num1, unk_count1 / tok_num1, max_len1)
    print(unk_count2, tok_num2, unk_count2 / tok_num2, max_len2)
    x1_train_t = [torch.tensor([(word2ix[tok] if tok in word2ix else 0) for tok in seg_line]) for seg_line in
                  seg_sent(x1_train)]
    x2_train_t = [torch.tensor([(word2ix[tok] if tok in word2ix else 0) for tok in seg_line]) for seg_line in
                  seg_sent(x2_train)]
    y_train_t = [torch.tensor([cat2ix[y]]) for y in y_train]
    x1_test_t = [torch.tensor([(word2ix[tok] if tok in word2ix else 0) for tok in seg_line]) for seg_line in
                 seg_sent(x1_test)]
    x2_test_t = [torch.tensor([(word2ix[tok] if tok in word2ix else 0) for tok in seg_line]) for seg_line in
                 seg_sent(x2_test)]
    y_test_t = [torch.tensor([cat2ix[y]]) for y in y_test]
    doc_tensor_dict = {}

    doc_tensor_dict["x1_train_t"] = x1_train_t
    doc_tensor_dict["x2_train_t"] = x2_train_t
    doc_tensor_dict["y_train_t"] = y_train_t
    doc_tensor_dict["x1_test_t"] = x1_test_t
    doc_tensor_dict["x2_test_t"] = x2_test_t
    doc_tensor_dict["y_test_t"] = y_test_t


    return doc_tensor_dict

""""
def convert_tokens_2_ids(x):
    input_bert = "[CLS] " + x[1] + " [SEP] " + " ".join(eval(x[0])) + " [SEP]"
    
    # input_bert = "[CLS] "+ " ".join(eval(x[0])) + " [SEP]"
    input_bert = input_bert.split()
    input_bert_ids = torch.tensor(tokenizer.convert_tokens_to_ids(input_bert)).unsqueeze(0)
    token_type_ids = [0 if i <= input_bert.index("[SEP]") else 1 for i in range(len(input_bert))]
    token_type_ids = torch.tensor([token_type_ids])

    return input_bert_ids, token_type_ids
"""


def count_text_length(x_train, x_test):
    length_counter = []
    x_all = x_train + x_test
    for x in x_all:
        # input_bert = "[CLS] " + x[1] + " [SEP] " + " ".join(eval(x[0])) + " [SEP]"
        input_bert = "[CLS] " + x[1] + " [SEP] "  # + " ".join(eval(x[0])) + " [SEP]"
        input_bert = input_bert.split()
        length_counter.append(len(input_bert))
    ll = length_counter

    print("max len ", np.max(ll))

    print("percentile95", np.percentile(ll, 95))
    print("percentile90", np.percentile(ll, 90))
    print("percentile90 ", np.percentile(ll, 80))


def get_doc_3_tensor(cat2ix, x1_train, x2_train, y_train, x1_test, x2_test, y_test):

    max_seq_length = 106
    from bert_process_functions import convert_tokens_to_features, convert_full_sequence_sdp_to_features
    train_features1 = convert_full_sequence_sdp_to_features(x1_train, max_seq_length, tokenizer)
    test_features1 = convert_full_sequence_sdp_to_features(x1_test, max_seq_length, tokenizer)
    train_features2 = convert_full_sequence_sdp_to_features(x2_train, max_seq_length, tokenizer)
    test_features2 = convert_full_sequence_sdp_to_features(x2_test, max_seq_length, tokenizer)
    # features_list  contains a list of feature dict

    tr_input_ids1 = torch.tensor([d["input_ids"] for d in train_features1])
    tr_input_mask1 = torch.tensor([d["input_mask"] for d in train_features1])
    tr_position_mask1 = torch.tensor([d["position_mask"] for d in train_features1])
    tr_type_ids1 = torch.tensor([d["type_ids"] for d in train_features1])

    te_input_ids1 = torch.tensor([d["input_ids"] for d in test_features1])
    te_input_mask1 = torch.tensor([d["input_mask"] for d in test_features1])
    te_position_mask1 = torch.tensor([d["position_mask"] for d in test_features1])
    te_type_ids1 = torch.tensor([d["type_ids"] for d in test_features1])

    tr_input_ids2 = torch.tensor([d["input_ids"] for d in train_features2])
    tr_input_mask2 = torch.tensor([d["input_mask"] for d in train_features2])
    tr_position_mask2 = torch.tensor([d["position_mask"] for d in train_features2])
    tr_type_ids2 = torch.tensor([d["type_ids"] for d in train_features2])

    te_input_ids2 = torch.tensor([d["input_ids"] for d in test_features2])
    te_input_mask2 = torch.tensor([d["input_mask"] for d in test_features2])
    te_position_mask2 = torch.tensor([d["position_mask"] for d in test_features2])
    te_type_ids2 = torch.tensor([d["type_ids"] for d in test_features2])

    y_train_t = torch.tensor([cat2ix[y] for y in y_train])
    y_test_t = torch.tensor([cat2ix[y] for y in y_test])

    train_data = TensorDataset(tr_input_ids1, tr_input_mask1, tr_position_mask1, tr_type_ids1,
                               tr_input_ids2, tr_input_mask2, tr_position_mask2, tr_type_ids2,y_train_t)
    test_data = TensorDataset(te_input_ids1, te_input_mask1, te_position_mask1, te_type_ids1,
                              te_input_ids2, te_input_mask2, te_position_mask2, te_type_ids2, y_test_t)

    doc_tensor_dict = {
        "train_dataset": train_data,
        "test_dataset": test_data
    }


    return doc_tensor_dict


def get_sequence_2_TensorDataset(cat2ix, x1_train, x2_train, y_train, x1_test, x2_test, y_test):

    max_seq_length = 113
    from bert_process_functions import convert_tokens_to_features, convert_full_sequence_sdp_to_features
    train_features1 = convert_full_sequence_sdp_to_features(x1_train, max_seq_length, tokenizer)
    test_features1 = convert_full_sequence_sdp_to_features(x1_test, max_seq_length, tokenizer)
    train_features2 = convert_full_sequence_sdp_to_features(x2_train, max_seq_length, tokenizer)
    test_features2 = convert_full_sequence_sdp_to_features(x2_test, max_seq_length, tokenizer)
    # features_list  contains a list of feature dict

    tr_input_ids1 = torch.tensor([d["input_ids"] for d in train_features1])
    tr_input_mask1 = torch.tensor([d["input_mask"] for d in train_features1])
    tr_position_mask1 = torch.tensor([d["position_mask"] for d in train_features1])
    tr_type_ids1 = torch.tensor([d["type_ids"] for d in train_features1])

    te_input_ids1 = torch.tensor([d["input_ids"] for d in test_features1])
    te_input_mask1 = torch.tensor([d["input_mask"] for d in test_features1])
    te_position_mask1 = torch.tensor([d["position_mask"] for d in test_features1])
    te_type_ids1 = torch.tensor([d["type_ids"] for d in test_features1])

    tr_input_ids2 = torch.tensor([d["input_ids"] for d in train_features2])
    tr_input_mask2 = torch.tensor([d["input_mask"] for d in train_features2])
    tr_position_mask2 = torch.tensor([d["position_mask"] for d in train_features2])
    tr_type_ids2 = torch.tensor([d["type_ids"] for d in train_features2])

    te_input_ids2 = torch.tensor([d["input_ids"] for d in test_features2])
    te_input_mask2 = torch.tensor([d["input_mask"] for d in test_features2])
    te_position_mask2 = torch.tensor([d["position_mask"] for d in test_features2])
    te_type_ids2 = torch.tensor([d["type_ids"] for d in test_features2])

    y_train_t = torch.tensor([cat2ix[y] for y in y_train])
    y_test_t = torch.tensor([cat2ix[y] for y in y_test])

    train_data = TensorDataset(tr_input_ids1, tr_input_mask1, tr_position_mask1, tr_type_ids1,
                               tr_input_ids2, tr_input_mask2, tr_position_mask2, tr_type_ids2,y_train_t)
    test_data = TensorDataset(te_input_ids1, te_input_mask1, te_position_mask1, te_type_ids1,
                              te_input_ids2, te_input_mask2, te_position_mask2, te_type_ids2, y_test_t)

    doc_tensor_dict = {
        "train_dataset": train_data,
        "test_dataset": test_data
    }

    return doc_tensor_dict


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, concat_all=True, with_lstm= True):
        super(BiLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        print("input_size : ", input_size)
        self.concat_all = concat_all
        self.with_lstm = with_lstm
        if self.concat_all:
            self.input_size = 4 * input_size
        else:
            self.input_size = 1 * input_size
        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True,
                            dropout=0.3)
        #self.hidden = nn.Linear(hidden_size * 4, hidden_size * 2)
        #self.out = nn.Linear(hidden_size * 2, output_size)
        if self.with_lstm:
            self.hidden = nn.Linear(hidden_size * 4, hidden_size * 2)
            self.out = nn.Linear(hidden_size * 2, output_size)

        else:
            if self.concat_all:
                self.hidden = nn.Linear(768 * 8, 400)
                self.out = nn.Linear(hidden_size * 2, output_size)

            else:
                self.hidden = nn.Linear(768*2, 400)
                self.out = nn.Linear(hidden_size * 2, output_size)
        
        self.out = nn.Linear(hidden_size * 2, output_size)
        self.bert_model = BertModel.from_pretrained('./pytorch_model/', output_hidden_states=True)
        for name, param in self.bert_model.named_parameters():
            param.requires_grad = True

    def forward(self, input_ids1, input_ids2,attention_mask1=None, position_mask1=None, token_type_ids1=None,
                  attention_mask2=None, position_mask2=None, token_type_ids2=None,hidden_state=None):
#        print(input_ids1.shape,token_type_ids1.shape,attention_mask1.shape)
#        print(input_ids2.shape,token_type_ids2.shape,attention_mask2.shape)
        with torch.no_grad():
            output_bert1 = bert_model(input_ids1, token_type_ids=token_type_ids1, attention_mask=attention_mask1)
            output_bert2 = bert_model(input_ids2, token_type_ids=token_type_ids2, attention_mask=attention_mask2)

        # print ( type (output_bert) , len( output_bert ))
        # print ( [ type(e ) for e  in  output_bert])
        if self.concat_all:
            bert_emebddings1 = torch.cat(tuple([output_bert1[2][i] for i in [-1, -2, -3, -4]]), dim=-1)
            bert_emebddings2 = torch.cat(tuple([output_bert2[2][i] for i in [-1, -2, -3, -4]]), dim=-1)

        else:
            bert_emebddings1 = output_bert1[2][-1]  # second to last layer embeddings
            bert_emebddings2 = output_bert2[2][-1]

        sdp_bert_embeddings1 = batched_index_select(bert_emebddings1, 1, position_mask1)
        sdp_bert_embeddings2 = batched_index_select(bert_emebddings2, 1, position_mask2)
        # sdp_bert_emebddings =  bert_emebddings
        if self.with_lstm:
            output1, hidden_state = self.lstm(sdp_bert_embeddings1.to(device), hidden_state)
            output1, _ = torch.max(F.relu(output1), 1)
            
            output2, hidden_state = self.lstm(sdp_bert_embeddings2.to(device), hidden_state)
            output2, _ = torch.max(F.relu(output2), 1)
        else:
            output1, _ = torch.max(F.relu(sdp_bert_embeddings1), 1)
            output2, _ = torch.max(F.relu(sdp_bert_embeddings2), 1)


        output = torch.cat((output1, output2), -1)
        output = self.hidden(output)
        # print ( "hidden  out ", output.shape )

        output = self.out(F.relu(output))
        # print ( "relu  out ", output.shape )

        output = F.log_softmax(output, dim=-1)
        # print ( "log_softmax.shape ",  output.shape   )

        return output

    def initHidden(self, batch_size):
        h = torch.zeros(2, batch_size, self.hidden_size).to(device)
        c = torch.zeros(2, batch_size, self.hidden_size).to(device)
        return (h, c)


def define_model(len_cat2ix, concat_all=True,with_lstm = True):
    n_hidden = 200
    n_epoch = 20
    model = BiLSTM(768, n_hidden, len_cat2ix, concat_all=concat_all, with_lstm= with_lstm)
    model = model.to(device)

    return model


def train_model(n_epoch=10, train_loader=None, valid_loader= None, test_loader= None,
                model=None, optimizer=None, criterion=None,flag = None):
    loss_list = []
    acc_list = []
    for epoch in range(1, n_epoch + 1):

        epoch_loss = []
        model.train()
        y_pred = []
        y_true = []
        best_loss_1 = 100000
        best_loss_2 = 100000
        best_loss_3 = 100000
        best_loss_4 = 100000

        for batch_data in tqdm(train_loader):
            # input_bert_ids, token_type_ids = batch_data
            batch_data = (e.to(device) for e in batch_data)
            input_bert_ids1, input_mask1, position_mask1, token_type_ids1,\
            input_bert_ids2, input_mask2, position_mask2, token_type_ids2,y = batch_data

            model.train()
            model.zero_grad()
            batch_size_ = input_bert_ids1.shape[0]
            hidden_state = model.initHidden(batch_size_)

            # output = model(input_bert_ids.to(device), token_type_ids.to(device), hidden_state)
            output = model(input_bert_ids1, input_bert_ids2,input_mask1, position_mask1, token_type_ids1,\
             input_mask2, position_mask2, token_type_ids2, hidden_state)
            y = y.to(device)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            pred = torch.argmax(output, dim=-1)

            y_true.append(y.detach().cpu().numpy())
            y_pred.append(pred.detach().cpu().numpy())
            epoch_loss.append(loss.item())

        y_true = np.array(list(itertools.chain(*y_true)))
        y_pred = np.array(list(itertools.chain(*y_pred)))
        accuracy = accuracy_score(y_true, y_pred)
        print('epoch %i, loss=%.4f, acc=%.2f%%' % (
            epoch, sum(epoch_loss) / len(epoch_loss), accuracy * 100))

        model.eval()
        with torch.no_grad():

            epoch_loss_val = []
            y_pred_val = []
            y_true_val = []
            for batch_data in valid_loader:
                batch_data = (e.to(device) for e in batch_data)
                input_bert_ids1, input_mask1, position_mask1, token_type_ids1, \
                input_bert_ids2, input_mask2, position_mask2, token_type_ids2, y = batch_data

                batch_size_ = input_bert_ids1.shape[0]
                hidden_state = model.initHidden(batch_size_)

                # output = model(input_bert_ids.to(device), token_type_ids.to(device), hidden_state)
                output = model(input_bert_ids1, input_bert_ids2,input_mask1, position_mask1, token_type_ids1,\
                input_mask2, position_mask2, token_type_ids2, hidden_state)
                y = y.to(device)
                loss_val = criterion(output, y)
                pred = torch.argmax(output, dim=-1)
                y_true_val.append(y.detach().cpu().numpy())
                y_pred_val.append(pred.detach().cpu().numpy())
                epoch_loss_val.append(loss.item())

            y_true_val = np.array(list(itertools.chain(*y_true_val)))
            y_pred_val = np.array(list(itertools.chain(*y_pred_val)))
            acc_val = accuracy_score(y_true_val, y_pred_val)
            print('epoch %i, loss=%.4f, acc=%.2f%%' % (
                epoch, sum(epoch_loss_val) / len(epoch_loss_val), acc_val * 100))
            loss_list.append(sum(epoch_loss_val) / len(epoch_loss_val))
            acc_list.append(acc_val)

            #########
            # global best_loss_1, best_loss_2

            if loss_val < best_loss_1 and flag == 1:
                torch.save(model.state_dict(), './model/model1')
                best_loss_1 = loss_val

            if loss_val < best_loss_2 and flag == 2:
                torch.save(model.state_dict(), './model/model2')
                best_loss_2 = loss_val
            if loss_val < best_loss_3 and flag == 3:
                torch.save(model.state_dict(), './model/model3')
                best_loss_3 = loss_val

            if loss_val < best_loss_4 and flag == 4:
                torch.save(model.state_dict(), './model/model4')
                best_loss_4 = loss_val

        model.eval()
        with torch.no_grad():
            y_pred = []
            y_true = []

            for batch_data in test_loader:
                batch_data = (e.to(device) for e in batch_data)
                input_bert_ids1, input_mask1, position_mask1, token_type_ids1, \
                input_bert_ids2, input_mask2, position_mask2, token_type_ids2, y = batch_data
                batch_size_ = input_bert_ids1.shape[0]
                hidden_state = model.initHidden(batch_size_)

                output = model(input_bert_ids1, input_bert_ids2,input_mask1, position_mask1, token_type_ids1,\
                    input_mask2, position_mask2, token_type_ids2, hidden_state)
                y = y.to(device)
                pred = torch.argmax(output, dim=-1)
                y_true.append(y.detach().cpu().numpy())
                y_pred.append(pred.detach().cpu().numpy())

            y_true = np.array(list(itertools.chain(*y_true)))
            y_pred = np.array(list(itertools.chain(*y_pred)))
            acc = accuracy_score(y_true, y_pred)
            from sklearn.metrics import f1_score
            f1_macro = f1_score(y_true, y_pred, average='macro')
            f1_micro = f1_score(y_true, y_pred, average='micro')
            f1_weighted = f1_score(y_true, y_pred, average='weighted')
            results = {
                "acc": acc,
                "f1_macro": f1_macro,
                "f1_micro": f1_micro,
                "f1_weighted": f1_weighted,

            }
            print("------------------")
            print("test result", acc * 100)
            print("------------------")



def evaluate_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        test_correct = 0
        y_pred = []
        y_true = []

        for batch_data in test_loader:
            # input_bert_ids, token_type_ids = batch_data
            batch_data = (e.to(device) for e in batch_data)
            input_bert_ids1, input_mask1, position_mask1, token_type_ids1, \
            input_bert_ids2, input_mask2, position_mask2, token_type_ids2, y = batch_data

            #model.train()
            #model.zero_grad()
            batch_size_ = input_bert_ids1.shape[0]
            hidden_state = model.initHidden(batch_size_)

            # output = model(input_bert_ids.to(device), token_type_ids.to(device), hidden_state)

            output = model(input_bert_ids1, input_bert_ids2,input_mask1, position_mask1, token_type_ids1,\
             input_mask2, position_mask2, token_type_ids2, hidden_state)
            y = y.to(device)
            pred = torch.argmax(output, dim=-1)
            y_true.append(y.detach().cpu().numpy())
            y_pred.append(pred.detach().cpu().numpy())

        y_true = np.array(list(itertools.chain(*y_true)))
        y_pred = np.array(list(itertools.chain(*y_pred)))
        acc = accuracy_score(y_true, y_pred)
        from sklearn.metrics import f1_score
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        results = {
            "acc": acc,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,

        }
        return results,acc





def create_data_loader_new(doc_tensor_dict, batch_size=16):
    train_dataset = doc_tensor_dict["train_dataset"]
    test_dataset = doc_tensor_dict["test_dataset"]
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    return train_loader, test_loader
def create_data_loader(doc_tensor_dict, batch_size=16):
    train_dataset = doc_tensor_dict["train_dataset"]
    test_dataset = doc_tensor_dict["test_dataset"]
    #train_dataset, valid_dataset = split_hanshu(  train_dataset  )
    train_split =0.85
    train_size = int(train_split * len(train_dataset))
    val_size = len( train_dataset  ) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset , shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    return train_loader, valid_loader, test_loader

def InitialModel(len_cat2ix, concat_all = True, with_lstm = True):


    model = define_model(len_cat2ix, concat_all= concat_all, with_lstm= with_lstm)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    return model, optimizer, criterion
    
def main_cv(kfold_num=5):
    print("**** run main main_cv **** ")
    results_cv_list = list()
    # kfold_num = 5
    data_dir = '/LanguageTime-master/bert_data/mat_sdp6'
    data_splits = doc_kfold(data_dir, cv=kfold_num)
    acc_all1 = []
    acc_all2 = []
    acc_all3 = []
    acc_all4 = []
    for cv_index in range(kfold_num):
        print("run the ")
        train_path = data_splits[cv_index][0]
        test_path = data_splits[cv_index][1]

        train_path_list = data_splits[cv_index][0]
        test_path_list = data_splits[cv_index][1]
        print(len(train_path_list))
        print(len(test_path_list))

        cat2ix, doc_tensor_dict = process_data(train_path_list, test_path_list)

        # get_dataloader
        # train_loader, test_loader = create_data_loader(doc_tensor_dict,  batch_size=16)
        train_loader, valid_loader, test_loader = create_data_loader(doc_tensor_dict,  batch_size=16)
        # define model and optimizer
        len_cat2ix = len(cat2ix)
        n_epoch = 20
        # ======== Train model    =====================#
        # train_model(n_epoch, train_loader=train_loader, valid_loader= valid_loader, test_loader=test_loader,
        #            model=model, optimizer=optimizer, criterion=criterion)
        model1, optimizer1, criterion1 = InitialModel(len(cat2ix), concat_all=False, with_lstm=True)
        model2, optimizer2, criterion2 = InitialModel(len(cat2ix), concat_all=False, with_lstm=False)
        model3, optimizer3, criterion3 = InitialModel(len(cat2ix), concat_all=True, with_lstm=True)
        model4, optimizer4, criterion4 = InitialModel(len(cat2ix), concat_all=True, with_lstm=False)
        n_epoch = 20

        train_model(n_epoch, train_loader=train_loader, valid_loader=valid_loader,
                                                test_loader=test_loader,
                                                model=model1,
                                                optimizer=optimizer1,
                                                criterion=criterion1,flag=1)
        train_model(n_epoch, train_loader=train_loader, valid_loader=valid_loader,
                            test_loader=test_loader,
                            model=model2,
                            optimizer=optimizer2,
                            criterion=criterion2,flag=2)
        train_model(n_epoch, train_loader=train_loader, valid_loader=valid_loader,
                        test_loader=test_loader,
                        model=model3,
                        optimizer=optimizer3,
                        criterion=criterion3, flag=3)
        train_model(n_epoch, train_loader=train_loader, valid_loader=valid_loader,
                        test_loader=test_loader,
                        model=model4,
                        optimizer=optimizer4,
                        criterion=criterion4,flag=4)
        # ======== evaluate  model    =====================#
        #results1 = evaluate_model(model1, test_loader)
        #results2 = evaluate_model(model2, test_loader)
        #results_cv_list.append(results)
        model1.load_state_dict(torch.load('./model/model1'))
        model2.load_state_dict(torch.load('./model/model2'))
        model3.load_state_dict(torch.load('./model/model3'))
        model4.load_state_dict(torch.load('./model/model4'))
        results1, acc1 = evaluate_model(model1, test_loader)
        results2, acc2 = evaluate_model(model2, test_loader)
        results3, acc3 = evaluate_model(model3, test_loader)
        results4, acc4 = evaluate_model(model4, test_loader)
        acc_all1.append(acc1)
        acc_all2.append(acc2)
        acc_all3.append(acc3)
        acc_all4.append(acc4)
            #results_cv_list.append(results)

        #df = pd.DataFrame(results_cv_list)
        #print(" results cv is : ")
        #print(df)
        #df.to_csv('result_cv.tsv', float_format='%.2f', sep='\t')
    print("acc_last_yes:", sum(acc_all1) / kfold_num)
    print("acc_last_no:", sum(acc_all2) / kfold_num)
    print("acc_all_yes:", sum(acc_all3) / kfold_num)
    print("acc_all_no:", sum(acc_all4) / kfold_num)
        

def main():
    main_cv(kfold_num=5)


if __name__ == "__main__":
    main()





