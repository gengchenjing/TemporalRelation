import os

datadir = "/Users/gengchenjing/LanguageTime-master/data/addid"
files = os.listdir(datadir)

f_out = open("../../data/connl_all_feats.txt", "w")

for file in files:
    f = open(os.path.join(datadir, file))
    lines = f.readlines()

    for i in range(0, len(lines)):

        lines[i] = lines[i].strip()

        if lines[i].startswith("# sent_id ="):
            sent_id = lines[i].replace("# sent_id =", "").strip()
            i = i+1
            original_sent = lines[i].replace("# text = ","").strip()
            i = i +1

            conj_form = ""
            base_form = ""

            while(lines[i].strip()!= "" ):
                word_feats = lines[i].split('\t')
                #print(word_feats)
                if word_feats[1].strip()!="":
                    conj_form += word_feats[1]+" "
                else:
                    conj_form += "|||" + " "
                if word_feats[2].strip() != "":
                    base_form += word_feats[2]+" "
                else:
                    base_form += "|||" + " "

                i = i +1

            f_out.write(sent_id+'\t'+original_sent+'\t'+conj_form.strip()+'\t'+base_form.strip())
            f_out.write('\n')

f_out.flush()
f_out.close()


