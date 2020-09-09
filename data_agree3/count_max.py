import numpy as np
with open("dct_result.txt","r+")as f1:
    length_counter=[]
    for lines in  f1.readlines() :

        #length_counter.append(i)
        
        length_counter.append(lines.count(","))
        ll = length_counter
    print(max(ll)) 
     
    #print ( "max len ", np.max( ll ))
