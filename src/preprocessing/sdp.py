j = 1
def conll_final(path1,path2,path3,path4):
    with open(txt3_path,'r+')as f1,open(conll1_path,'r+')as f2,open("111.txt",'w+')as f3, open(conll2_path,'w+')as f4:
    
        list1 = []
        ars = []
        conll_str = ''
        arr = {}
           
        for l in f2.readlines():
            
            if not len(l.strip()):
                print(conll_str,file = f3)
                for lins in conll_str.split('\n'):
                    if ('# sent_id') in lins:
                        sen_id = lins[12:]
                        arr[sen_id] = conll_str
                        #print(arr)
                        #print(sen_id)
                conll_str = ''

            else:
                conll_str += l

            
        lines = f1.readlines()
        for line in lines:
            cols = line.split("\t")
            list1.append(cols[0])
            ars.append((cols[1],cols[4]))
        for ar in ars:
        
            print(ar)

                
    
        for col in list1:
            if arr.__contains__(col) == True:
                print(arr.get(col),file = f4)
                
while j < 55:
    conll1_path = "./{:0>5d}-ud.conll".format(j)
    conll2_path = "./dct_final/{:0>3d}.conll".format(j)
    txt3_path = "./test/test{:0>2d}.txt".format(j)
    j += 1 
    conll_final(txt3_path,conll1_path,"111.txt",conll2_path)
