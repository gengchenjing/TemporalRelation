# coding: utf-8

import torch
import pandas as pd 
import numpy as np 
from torch import nn 
import spacy
import MeCab
mt = MeCab.Tagger()

 
def build_pos_tensor (  pos_list  , pos2dict  ):

    
    data_pos_tensor = [torch.tensor([(pos2dict[pos] if pos in pos2dict else 0) for pos in pos_line]) for pos_line  in
             pos_list ] 

    return data_pos_tensor 

def build_pos_dict (   pos_list  ):
    all_pos_types  = [item  for sent  in pos_list for item in sent ]
    unique_pos  = set( all_pos_types )
    pos2dict = {'UNK': 0}
    for i , pos  in enumerate(unique_pos):
        pos2dict[pos ] = i + 1

    return  pos2dict

def seg_sent_pos(x_data):
    seg_list = []
    seg_pos_list = [ ]
    for line in x_data:
        #print ( line )
        toks = mt.parse(line.strip()).splitlines()[:-1] 
         
        toks_list =    [  tok.split('\t')     for tok  in toks ]
        #print ( "toks_list" ,toks_list   )
          
        """ 
        for tok_and_parse in toks: 
            tok_and_parse = tok_and_parse.split('\t') 
            tok =  tok_and_parse[0]
            pos  =  tok_and_parse[1].split( ",")[0 ]
        """ 
        seg_list.append([ tok_and_parse[0]     for tok_and_parse in toks_list] )
        seg_pos_list.append([   tok_and_parse[1].split( ",")[0 ]     for tok_and_parse in toks_list] )
        
        
    return seg_list ,seg_pos_list  


def test (  ):
    sent_list = ["今日もしないとね","明日天気"]
    token_list ,pos_list   = seg_sent_pos(sent_list) 

    print ("token_list: ",token_list )
    print ("pos_list: ",pos_list )
    pos2dict = build_pos_dict (   pos_list  ) 
    print ("pos2dict: ",pos2dict )

    # for example , x_train_pos is 
    x_train =  ["明日もしないとね","今日天気"] 

    x_train_token_list , x_train_pos_list   = seg_sent_pos(x_train ) 


    x_train_pos = build_pos_tensor (  x_train_pos_list,pos2ix  )

    print ( "x_train_pos " )
    print ( x_train_pos )



    pos_embedding_size  = 3  
    pos_embedding =  nn.Embedding(len(pos2ix ), pos_embedding_size )
    for x in x_train_pos:
        pos_out = pos_embedding ( x )

        print ( pos_out )

if __name__ == "__main__":
    test()  
