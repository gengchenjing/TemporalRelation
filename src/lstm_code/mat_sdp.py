import glob
import re
import os
import unicodedata
import string
import sys
import codecs
import random
import torch
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import numpy as np
from io import open
from sklearn.metrics import f1_score
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('PS')
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Turning Words into Tensors
from gensim.models import KeyedVectors
import MeCab
# create network
from torch.autograd import Variable
from sklearn.model_selection import train_test_split, GroupKFold, GroupShuffleSplit, KFold
from sklearn.metrics import accuracy_score 
import itertools

mt = MeCab.Tagger()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)


def get_pretrained_embed_vector( ):
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

    #pre_vector = None 
    #print(pre_vector.shape, len(word2ix))
    return   word2ix, pre_vector 



word2ix, pre_vector   =  get_pretrained_embed_vector (  )
#word2ix  = {'UNK': 0} #, pre_vector   =  get_pretrained_embed_vector (  )
#pre_vector = None 
def unicodeToUtf8(s):
    return s.encode('utf-8')

def readlines(filename):
    f = codecs.open(filename, 'r', 'utf-8')
    lines = f.read().strip().split('\n')
    f.close()
    return [unicodeToUtf8(line) for line in lines if 'reltype' in line]
    
def read_multi_files(files_namelist  ):
    files_list = [ ]
    for filename  in  files_namelist:
        each_file_line_list  =  readlines (  filename)
        files_list+= each_file_line_list
    #return [unicodeToUtf8(line) for line in lines if 'reltype' in line]
    return files_list

def doc_kfold(data_dir, cv=5):
    file_list, file_splits = [], []
    for file in sorted(os.listdir(data_dir)):
        if file.endswith(".txt"):
            dir_file = os.path.join(data_dir, file)
            file_list.append(dir_file)

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

def padding_pos(seq_2d, max_len, pad=[0, 0], direct='right'):

    for seq_1d in seq_2d:
        for i in range(0, max_len - len(seq_1d)):
            if direct in ['right']:
                seq_1d.append(pad)
            else:
                seq_1d.insert(0, pad)
    return seq_2d


def pos2idx(doc_dic, link_type):
    tok_idx = {'zeropadding': 0}
    for doc in doc_dic.values():
        for link in doc.get_links_by_type(link_type):
            for tok_l, tok_r in link.interpos:
                tok_idx.setdefault(tok_l, len(tok_idx))
                tok_idx.setdefault(tok_r, len(tok_idx))
    return tok_idx

def getSentAndCategory(datalist):
    global sent_list1
    global sent_list2
    global category_list
    sent_list1 = []
    sent_list2 = []

    category_list = []
    for data in datalist:

        sent_list1.append(data[1])
        sent_list2.append(data[4])
        category_list.append(data[2])
    n_categories = len(set(category_list))
    
    return (sent_list1,sent_list2, category_list, n_categories)

def process_data ( train_data_list, test_data_list ):
    sent_list1_train, sent_list2_train, category_list_train, n_train_categories = getSentAndCategory(train_data_list)
    sent_list1_test, sent_list2_test, category_list_test, n_test_categories = getSentAndCategory(test_data_list)
    sent_list1_all, sent_list2_all,category_list, _ = getSentAndCategory(train_data_list+test_data_list)
    sent2category = {}
    for i in range(len(sent_list1)):
        sent2category[sent_list1[i]] = category_list[i]
    

    x1_train = np.array(sent_list1_train )
    x2_train = np.array(sent_list2_train)
    y_train = np.array(category_list_train )
    x1_test = np.array(sent_list1_test)
    x2_test = np.array(sent_list2_test)
    y_test = np.array(category_list_test)

    cat2ix  = get_cat2ix(   )

    doc_tensor_dict  = get_doc_2_tensor ( cat2ix, x1_train, x2_train,y_train, x1_test,x2_test, y_test  )
    pos_embedding_size, pos_embedding,doc_topic_tensor_dict  =   get_doc_2_topic_tensor( x1_train, x2_train,x1_test,x2_test)
    return cat2ix , pos_embedding_size,pos_embedding , doc_tensor_dict , doc_topic_tensor_dict
 
def get_cat2ix( ):
 
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
 
def get_doc_2_tensor(   cat2ix, x1_train,x2_train, y_train, x1_test,x2_test, y_test ):
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
    doc_tensor_dict = { }


    doc_tensor_dict["x1_train_t" ] = x1_train_t
    doc_tensor_dict["x2_train_t"] = x2_train_t
    doc_tensor_dict["y_train_t" ] = y_train_t
    doc_tensor_dict["x1_test_t" ] = x1_test_t
    doc_tensor_dict["x2_test_t"] = x2_test_t
    doc_tensor_dict["y_test_t" ] = y_test_t

    return doc_tensor_dict

def get_doc_2_topic_tensor( x1_train, x2_train, x1_test, x2_test):
    from  pos_functions_collections  import seg_sent_pos,  build_pos_dict , build_pos_tensor, build_pos_tensor


    _, pos_list1 = seg_sent_pos(sent_list1)
    _, pos_list2 = seg_sent_pos(sent_list2)
    _, x1_train_pos = seg_sent_pos(x1_train)
    _, x2_train_pos = seg_sent_pos(x2_train)
    _, x1_test_pos = seg_sent_pos(x1_test)
    _, x2_test_pos = seg_sent_pos(x2_test)
    pos2dict = build_pos_dict(pos_list1+pos_list2)

    x1_train_pos_t = build_pos_tensor(x1_train_pos, pos2dict)
    x2_train_pos_t = build_pos_tensor(x2_train_pos, pos2dict)
    x1_test_pos_t = build_pos_tensor(x1_test_pos, pos2dict)
    x2_test_pos_t = build_pos_tensor(x2_test_pos, pos2dict)
 

    doc_topic_tensor_dict = { }
    doc_topic_tensor_dict["x1_train_pos_t" ] = x1_train_pos_t
    doc_topic_tensor_dict["x2_train_pos_t" ] = x2_train_pos_t
    doc_topic_tensor_dict["x1_test_pos_t" ] = x1_test_pos_t
    doc_topic_tensor_dict["x2_test_pos_t" ] = x2_test_pos_t
 
    pos_embedding_size  = 25  
    pos_embedding =  nn.Embedding(len(pos2dict ), pos_embedding_size )
    return pos_embedding_size, pos_embedding ,doc_topic_tensor_dict




class BiLSTM_pos(nn.Module):
    def __init__(self, input_size, hidden_size,  output_size, pre_embed, pos_embedding_size,  pos_embedding ):
        super(BiLSTM_pos, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size+pos_embedding_size, hidden_size, num_layers=1,batch_first=True, bidirectional=True)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pre_embed), freeze=True)
        self.pos_embedding = pos_embedding 



        self.hidden = nn.Linear(hidden_size * 4, hidden_size * 2)

        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_embeded1,input_embeded2, hidden_state = None):
        output1, hidden_state = self.lstm(input_embeded1, hidden_state)
        output1, _ = torch.max(F.relu(output1), 1)
        output2, hidden_state = self.lstm(input_embeded2, hidden_state)
        output2, _ = torch.max(F.relu(output2), 1)
        output = torch.cat((output1, output2), -1)
        output = self.hidden(output)
        output = self.out(F.relu(output))
        output = F.log_softmax(output, dim=-1)

        return output

    def initHidden(self):
        h = Variable(torch.zeros(2, 1, self.hidden_size))
        c = Variable(torch.zeros(2, 1, self.hidden_size))
        return (h, c)



def define_model ( len_cat2ix,pre_embed, pos_embedding_size,pos_embedding ):
    n_hidden = 200
    n_epoch = 20
    #model = BiLSTM(300, n_hidden, len(cat2ix), pre_vector,  pos_embedding_size,  pos_embedding )
    model = BiLSTM_pos(300, n_hidden,len_cat2ix, pre_embed ,  pos_embedding_size,  pos_embedding )
    
    return model
def train_model( n_epoch = 10, train_loader= None , valid_loader = None, model=None ,  optimizer = None, criterion=None  ):
    
    loss_list = []
    acc_list = []
    
    for epoch in range(1, n_epoch + 1):

        epoch_loss = []
        model.train() 
        y_pred =[ ]
        y_true =[ ]
        best_loss = 10000

        for batch_data in tqdm(train_loader):
 
            x1, x1_pos, x2, x2_pos, y   =  batch_data
            
            model.zero_grad()
            embed_x1 = model.embedding(x1).unsqueeze(1)
            embed_pos1 = model.pos_embedding(x1_pos ).unsqueeze(1)

            embed_x1_pos = torch.cat(( embed_x1 , embed_pos1 ), -1 )
            embed_x1_pos = embed_x1_pos.squeeze(1)
            embed_x2 = model.embedding(x2).unsqueeze(1)
            embed_pos2 = model.pos_embedding(x2_pos).unsqueeze(1)
            embed_x2_pos = torch.cat((embed_x2, embed_pos2), -1)
            embed_x2_pos = embed_x2_pos.squeeze(1)
            output = model(embed_x1_pos,embed_x2_pos)

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            pred = torch.argmax(output, dim=-1)
            y_true.append ( y.detach().cpu().numpy() )
            y_pred.append( pred.detach().cpu().numpy() )

            epoch_loss.append(loss.item())


        y_true  = np.array (list (itertools.chain  (  *y_true)))
        y_pred  = np.array (list (itertools.chain  (  *y_pred)))
        accuracy  =  accuracy_score(y_true, y_pred)
        print('epoch %i, loss=%.4f, acc=%.2f%%' % (
        epoch, sum(epoch_loss) / len(epoch_loss),  accuracy *100 ))
        
        model.eval()
        with torch.no_grad():
            
            epoch_loss_val = []
            y_pred_val =[ ]
            y_true_val =[ ]
            for batch_data in valid_loader :
                model.zero_grad()
                embed_x1 = model.embedding(x1).unsqueeze(1)
                embed_pos1 = model.pos_embedding(x1_pos ).unsqueeze(1)

                embed_x1_pos = torch.cat(( embed_x1 , embed_pos1 ), -1 )
                embed_x1_pos = embed_x1_pos.squeeze(1)
                embed_x2 = model.embedding(x2).unsqueeze(1)
                embed_pos2 = model.pos_embedding(x2_pos).unsqueeze(1)
                embed_x2_pos = torch.cat((embed_x2, embed_pos2), -1)
                embed_x2_pos = embed_x2_pos.squeeze(1)
                
                val_out = model(embed_x1_pos,embed_x2_pos) 
                loss_val = criterion(val_out, y)
                pred = torch.argmax(val_out, dim=-1)
                y_true_val.append ( y.detach().cpu().numpy() )
                y_pred_val.append( pred.detach().cpu().numpy())
                epoch_loss_val.append(loss.item())
                

            y_true_val  = np.array (list (itertools.chain  (  *y_true_val)))
            y_pred_val  = np.array (list (itertools.chain  (  *y_pred_val)))
            acc_val = accuracy_score(y_true, y_pred)
            print('epoch %i, loss=%.4f, acc=%.2f%%' % (
                epoch, sum(epoch_loss_val) / len(epoch_loss_val), acc_val * 100))
            loss_list.append(sum(epoch_loss_val) / len(epoch_loss_val))
            acc_list.append(acc_val)
            
            if loss_val < best_loss:
                torch.save(model.state_dict(), './model')
                best_loss = loss_val
        

def evaluate_model (model , test_loader  ):
    model.eval()
    with torch.no_grad():
        test_correct = 0
        y_pred =[ ]
        y_true =[ ]

        for batch_data in test_loader:
            
            x1,x1_pos, x2, x2_pos, y = batch_data
  
            embed_x1 = model.embedding(x1).unsqueeze(1)
            embed_pos1 = model.pos_embedding(x1_pos ).unsqueeze(1)
            embed_x1_pos = torch.cat(( embed_x1 , embed_pos1 ), -1 )
            #print ("embed_x_pos shape : ", embed_x_pos.shape  )
            embed_x1_pos = embed_x1_pos.squeeze(1)
            embed_x2 = model.embedding(x2).unsqueeze(1)
            embed_pos2 = model.pos_embedding(x2_pos).unsqueeze(1)
            embed_x2_pos = torch.cat((embed_x2, embed_pos2), -1)
            # print ("embed_x_pos shape : ", embed_x_pos.shape  )
            embed_x2_pos = embed_x2_pos.squeeze(1)
            test_out = model(embed_x1_pos, embed_x2_pos )
            pred = torch.argmax(test_out, dim=-1)
            y_true.append ( y.detach().cpu().numpy() )
            y_pred.append( pred.detach().cpu().numpy() )

        y_true  = np.array (list (itertools.chain  (  *y_true)))
        y_pred  = np.array (list (itertools.chain  (  *y_pred)))
        acc   =  accuracy_score(y_true, y_pred)
    
        from sklearn.metrics import f1_score
        f1_macro= f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')

        results = {
           "acc":  acc,
           "f1_macro":  f1_macro, 
           "f1_micro":  f1_micro,
           "f1_weighted":  f1_weighted,

        }
        return results, acc
        
def create_data_loader(   doc_tensor_dict,  doc_topic_tensor_dict,batch_size =16  ):
    x1_train_pos_t = doc_topic_tensor_dict["x1_train_pos_t" ]
    x1_test_pos_t  = doc_topic_tensor_dict["x1_test_pos_t" ]
    x1_train_t = doc_tensor_dict["x1_train_t" ]
    y_train_t = doc_tensor_dict["y_train_t" ]  
    x1_test_t = doc_tensor_dict["x1_test_t" ]
    y_test_t = doc_tensor_dict["y_test_t" ]
    x2_train_pos_t = doc_topic_tensor_dict["x2_train_pos_t"]
    x2_test_pos_t = doc_topic_tensor_dict["x2_test_pos_t"]
    x2_train_t = doc_tensor_dict["x2_train_t"]
    x2_test_t = doc_tensor_dict["x2_test_t"]

    y_test_t  = torch.cat(  y_test_t, 0 )
    y_train_t  = torch.cat(  y_train_t, 0 )

    x1_train_t = torch.stack([torch.cat([i, i.new_zeros(max_len1  - i.size(0))], 0) for i in  x1_train_t ],0)
    x1_test_t = torch.stack([torch.cat([i, i.new_zeros(max_len1 - i.size(0))], 0) for i in  x1_test_t ],0)
    x1_train_pos_t = torch.stack([torch.cat([i, i.new_zeros(max_len1 - i.size(0))], 0) for i in  x1_train_pos_t ],0)
    x1_test_pos_t = torch.stack([torch.cat([i, i.new_zeros(max_len1 - i.size(0))], 0) for i in  x1_test_pos_t ],0)
    x2_train_t = torch.stack([torch.cat([i, i.new_zeros(max_len2  - i.size(0))], 0) for i in  x2_train_t ],0)
    x2_test_t = torch.stack([torch.cat([i, i.new_zeros(max_len2 - i.size(0))], 0) for i in  x2_test_t ],0)
    x2_train_pos_t = torch.stack([torch.cat([i, i.new_zeros(max_len2 - i.size(0))], 0) for i in  x2_train_pos_t ],0)
    x2_test_pos_t = torch.stack([torch.cat([i, i.new_zeros(max_len2 - i.size(0))], 0) for i in  x2_test_pos_t ],0)

    train_dataset = TensorDataset(  x1_train_t, x1_train_pos_t,x2_train_t, x2_train_pos_t,y_train_t  )
    test_dataset = TensorDataset(   x1_test_t,  x1_test_pos_t,x2_test_t,  x2_test_pos_t, y_test_t  )


    train_split =0.85
    train_size = int(train_split * len(train_dataset))
    val_size = len( train_dataset  ) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset , shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    #train_loader = DataLoader(train_dataset ,shuffle=True,  batch_size= batch_size)
    #test_loader = DataLoader(test_dataset,shuffle=False,  batch_size= batch_size)
    return train_loader ,valid_loader, test_loader



def main_cv(kfold_num  = 5):
    print ( "**** run main main_cv **** ")
    results_cv_list = list ( )
     
    data_dir = '/Users/gengchenjing/LanguageTime/data_words/mat6'
    data_splits = doc_kfold(data_dir, cv=kfold_num )
    acc_all = []
    
    for cv_index in range( kfold_num ): 
        print ("run the ")
 
        train_path_list  =  data_splits[cv_index][0]           
        test_path_list  =  data_splits[cv_index][1] 
        print(len(train_path_list)) 
        print(len(test_path_list))
        
        datalist_train = [re.split('\t|\|', line.decode('utf-8')) for line in read_multi_files(train_path_list )]
        datalist_test= [re.split('\t|\|', line.decode('utf-8')) for line in read_multi_files(test_path_list )]
        cat2ix, pos_embedding_size, pos_embedding,  doc_tensor_dict , doc_topic_tensor_dict  \
        = process_data ( datalist_train, datalist_test) 
          

        # get_dataloader 
        train_loader, valid_loader, test_loader = create_data_loader( doc_tensor_dict,  doc_topic_tensor_dict, batch_size = 16  )
        
        #define model and optimizer  
        model = define_model(  len(cat2ix ),pre_vector, pos_embedding_size, pos_embedding   )
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
        criterion = nn.NLLLoss()
        n_epoch = 20
        #======== Train model    =====================#
        train_model ( n_epoch, train_loader= train_loader, valid_loader=valid_loader, model=model,  optimizer = optimizer, 
        criterion=criterion )
        #======== evaluate  model    =====================#
        
        model.load_state_dict(torch.load('./model'))
        results, acc = evaluate_model (model, test_loader)
        acc_all.append(acc)

    print("acc_aver:", sum(acc_all) / kfold_num)



def main():

    main_cv ( kfold_num=5 )

if __name__ == "__main__":    
    main( )




