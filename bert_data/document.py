import random
arr=[]
i = 0
while i < 54:
    arr.append('')
    i += 1

        
with open("mat_sdp4.txt",'r+')as f1:
        #line_str = ''
    for line in f1.readlines():
        line = str(line)
        ids = line[1:4]
        #print(ids)
        page = int(ids)-1
        #print(page)
        arr[page]+=line
            
j = 1
while j < 55:
    txt_path = "./mat_sdp4/{:0>3d}.txt".format(j)
    with open(txt_path,'w+')as f2:
        print(arr[j-1],file =f2)
    j+= 1   
             
   
        




