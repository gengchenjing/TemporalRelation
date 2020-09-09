import re

arr=[]
i = 0
while i < 54:
    arr.append({})
    i += 1
with open("./merge/merged/DCT.txt",'r+') as f1:
    lst = {}
    for line in f1.readlines():
        line = str(line)
        cols = line.split("\t")
        page = int(cols[0])-1
        #print((page , '-' , cols[2]),file= f2)
        arr[page][cols[2]] = cols[3]


def conll_select(path1,path2): 
    with open(conll1_path,'r+')as f, open(conll2_path,'w+')as f2:
        conll_str = ''
        lines = f.readlines()
        true_event = False
        for l in lines:
        
            if l.startswith('#'):
                conll_str += l
            
                continue
        
            else:
                ar = re.findall(r"eid=\"(.+?)\"",l)
    
                if not ar == None:
                    if len(ar) > 0:
                        
                        event=ar[len(ar) - 1]
                        #print (event)
                        #print ("Value : %s" %  arr[0].__contains__(event))
                        if arr[i].__contains__(event) == True:
                            true_event = True
    
        
            if not len(l.strip()):
                if true_event == True:
                    print(conll_str,file = f2)
                conll_str = ''
                true_event = False
            else:
                conll_str += l

def event_select(path1,path2,path3):

    with open(conll2_path,'r+')as f2,open(txt1_path,'w+')as f3, open(txt2_path,'w+')as f4:
        con_str = ''
        con_str2 = ''
        for lin in f2.readlines():
            if lin.startswith('# sent_id'):
                con_str += lin[12:-2]+'\t'
                con_str2 = lin[12:-2]
                #print(lines[12:-2])
            elif not lin.startswith('#'):
                lin = str(lin)
                if len(lin.strip()) > 8:
                    colm = lin.split("\t")
                    if colm[7] == 'root':
                        con_str += colm[0]+'\t'+colm[1]+'\t'+'root'+'\n'
                    ar = re.findall(r"eid=\"(.+?)\"",lin)
                    if not ar == None:
                        if len(ar) > 0:
                            colm[10]=ar[len(ar) - 1]
                            if arr[i].__contains__(colm[10]) == True:
                                print((con_str2+ '\t'+colm[0]+'\t'+colm[1]+'\t'+colm[10] ),file = f4)
                            
        print(con_str,file = f3)
def event_combine(path1,path2,path3):
    with open(txt1_path,'r+')as f3, open(txt2_path,'r+')as f4,open(txt3_path,'w+')as f5,open("DCT.txt",'r+')as f6:
        arr2={}
        events=f4.readlines
        for line in f3.readlines():
            list1 =[]
            if len(line):
                line = str(line.strip('\n'))
                cols = line.split("\t")
                if len(cols)==4:
                    list1 = cols[1]+'\t'+cols[2]+'\t'+cols[3]
                    arr2[cols[0]] = list1

    
        
        for lines in f4.readlines():
            lines = str(lines.strip("\n"))
            col = lines.split("\t")
            for lin in f6.readlines():
                lin = str(lin)
                colss = lin.split("\t")
                if arr2.__contains__(col[0]) == True:
                    print((col[0]+'\t'+col[1]+'\t'+col[2]+'\t'+col[3]+'\t'+arr2.get(col[0])+'\t'+colss[3]),file = f5)
i = 0
while i < 54:
    conll1_path = "./merge/merged/conll/{:0>5d}-ud.conll".format(i+1)
    conll2_path = "./merge/merged/dct_conll/{:0>2d}.conll".format(i+1)
    txt1_path = "./merge/merged/dct_root/{:0>2d}.txt".format(i+1)
    txt2_path = "./merge/merged/dct_event/{:0>2d}_event.txt".format(i+1)
    txt3_path = "./merge/merged/dct_root/test/test{:0>2d}.txt".format(i+1) 
    conll_select(conll1_path,conll2_path)
    event_select(conll2_path,txt1_path,txt2_path)
    event_combine(txt1_path,txt2_path,txt3_path)
    i += 1
        