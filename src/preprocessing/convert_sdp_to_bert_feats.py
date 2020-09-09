
f_connl_feats = open("/Users/gengchenjing/LanguageTime-master/data/connl_all_feats.txt")
lines = f_connl_feats.readlines()
feats_dict = {}

for line in lines:
    feats = line.strip().split('\t')
    feats_dict[feats[0]] = feats[1:]

def get_lemma (feats, sdp):

    lemmas_sdp = []

    conj_form = feats[1]
    base_form = feats[2]
    print(len(conj_form.split()), len(base_form.split()))

    for i in range(0, len(conj_form.split())):
        print(conj_form.split()[i], base_form.split()[i])
    assert(len(conj_form.split()) == len(base_form.split()))

    map_conj_base = {}
    for i in range(0, len(conj_form.split())):
        map_conj_base[conj_form.split()[i]] = base_form.split()[i]

    for tok in sdp:
        lemmas_sdp.append(map_conj_base[tok])

    return lemmas_sdp

def clean_marker(text):
    text = text.replace("||| ", "")
    text = text.replace(" ||| ", "")
    text = text.replace(" |||", "")
    text = text.replace("\n", "")
    return text

#f_in = open("/Users/lisk/PycharmProjects/LanguageTime-master/data_agree2/dct_result4/train.txt")
#f_in = open("/Users/lisk/PycharmProjects/LanguageTime-master/data_agree2/dct_result4/test.txt")
f_in = open("/Users/gengchenjing/LanguageTime/input data/merge/merged/conll_result/mat_result2_conll.txt")
lines = f_in.readlines()

#f_out = open("/Users/lisk/PycharmProjects/LanguageTime-master/data_agree2/dct_result4/train_bert.txt", "w")
#f_out = open("/Users/lisk/PycharmProjects/LanguageTime-master/data_agree2/dct_result4/test_bert.txt", "w")
f_out = open("/Users/gengchenjing/LanguageTime/bert_data/mat_result2.txt", "w")

for line in lines:
    if line.strip() != "":
        data = line.strip().split('\t')
        id = data[0]
        feats = feats_dict[id]
#        id1 = data[0]
#        feats1 = feats_dict[id1]
#        id2 = data[3]
#        feats2 = feats_dict[id2]
        #print("->", feats)
        sdp = eval(data[1].strip())

        sdp_lemma = get_lemma(feats, sdp)
#        sdp1 = eval(data[1].strip())

#        sdp_lemma1 = get_lemma(feats, sdp1)
#        sdp2 = eval(data[4].strip())

#        sdp_lemma2 = get_lemma(feats, sdp2)
        f_out.write(data[0]+"\t"+data[2]+"\t"+data[1]+"\t"+str(sdp_lemma)+"\t"+clean_marker(feats[2])+"\t"+clean_marker(feats[2])+"\t"+data[4])
        f_out.write("\n")
f_out.flush()
f_out.close()
