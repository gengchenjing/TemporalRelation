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

mecab

必要なファイル:　
BERT - pytorch_model
FastText - nwjc_word_skip_300_8_25_0_1e4_6_1_0_15.txt.vec

Process:

1. SDPを生成するため

src/merge/BCCWJ-TIMEXファイルから、3人アノテーション一致の識別対象の単語（イベントと時間表現）、インデックス（eiXX）と時間関係(reltype)をとる

DCT: sent_id  event   event_id                      時間関係

T2E: sent_id1 time    time_id     event   event_id  時間関係

E2E: sent_id1 event1  event_id1   event2  event_id2 時間関係

MAT: sent_id1 event1  event_id1   sent_id2  event2  event_id2 時間関係

生成したファイルは/merged/DCT.txt(T2E.txt/ E2E.txt/ MAT.txt)と同じように

DCTタスク、root情報の追加が必要ある：src/preprocessing/make_test_dct.pyを使って、結果はTemporalRelation/src/merge/merged/test/で



２.必要なStringを生成する

./merge/merged/conllを使って必要だけのconll　stringを生成する

sdp.py/ sdp_mat.pyを使って、結果はTemporalRelation/src/merge/tmp/dct_finalのところ


3.SDPを生成する

src/preprocessing/dependency_dct.pyを使って、

結果はTemporalRelation/src/merge/merged/result/


4. ラベルをまとめって、54ファイルに分ける

python3 count.py
python3 document.py



Usage:

Fasttext model

*******************************************

Reprocessing 
1. Generate files for dct_sdp: python3 make_test_dct.py
2. Generate string used for sdp: python3 sdp.py(DCT, E2E, T2E)/ python3 sdp_mat.py(MAT)
3. Generate SDP: python3 dependency_dct.py

Test

1. Train and test : lstm code/ python3 fasttext.py(DCT, E2E, T2E)/ python3 mat_sdp.py(MAT)
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
