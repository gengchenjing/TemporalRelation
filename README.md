# TemporalRelation

Required Packages

glob
re
os
unicodedata
codecs
random
torch
pandas
numpy
tqdm
io
sklearn
gensim
matplotlib
itertools

Usage:

Fasttext model

*******************************************

Reprocessing 
1. Generate files for dct_sdp: make_test_dct.py
2. Generate string used for sdp: sdp.py/ sdp_mat.py
3. Generate SDP: dependency_dct.py

Test

1. Train and test : lstm code/ fasttext.py/ mat_sdp.py
2. Generating embedding for POS(part-of-speech) : pos_functions_collections.py

*******************************************

BERT Model

***********************************************

Reprocessing

1. Extracting features from Conll files: extract_features_connl.py
2. Convert SDP to BERT features: convert_sdp_to_bert_feats.py
3. Add position information for SDP words: add_positon.py

Test
1. BERT without finetune: temp_bert.py
2. BERT with finetune: bert_finetuning.py (layers and lstm)
3. Processing with mask on SDP words: bert_process_functions.py
