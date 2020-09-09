import xmltodict
import json
import os 
import sys
import io
import xml.etree.ElementTree as ET
import numpy as np
import re

conllPath = './merge/merged/conll/'
for curDir, dirs, filesConll in os.walk(conllPath):
    pass
filesConll.sort()
idx = 0
conll_lines = []
for f_conll in filesConll:
    fConll = open(conllPath + f_conll)
    for idx, line in enumerate(fConll.readlines()):
        conll_lines.append(line)

resultPath = './merge/merged/mat/'
for curDir, dirs, filesResult in os.walk(resultPath):
    pass

for fileName in filesResult:
    saveFile = io.open('./merge/merged/conll_result/' + fileName[:-4] + '_conll.txt', 'w')
    curFile = io.open(resultPath+fileName, 'r')
    lines = curFile.readlines()
    for item in lines:
        origin_item = item
        item = item.split('\t')
        
        if fileName[:-4] == 'mat_word1':
            words = item[1]
            types = item[2]
        
        elif fileName[:-4] == 'mat_word2':
            words = item[1]
            types = item[2]
        elif fileName[:-4] == 'mat_result1':
            words = item[1]
            types = item[2]
        
        elif fileName[:-4] == 'mat_result2':
            words = item[1]
            types = item[2]
            """"
        elif fileName[:-4] == 't2e_result':
            words = item[2]
            types = item[1]
        elif fileName[:-4] == 'dct_word':
            words = item[2]
            types = item[1]
            
        elif fileName[:-4] == 'e2e_word':
            words = item[2]
            types = item[1]
    
        elif fileName[:-4] == 't2e_word':
            words = item[2]
            types = item[1]
            """    
        else:
            break
        pattern = re.compile("'(\w+)'")          
        words = pattern.findall(words)
        # iter conll file
        for idx, conll in enumerate(conll_lines):
            if conll.startswith('# sent_id') and conll[12:-2] == item[0]:
                posList = []
                times = len(words)
                for idx_word in range(idx+2,10000000):
                    conll_word = conll_lines[idx_word].split('\t')
                    if conll_word[1] in words:
                        posList.append(conll_word[0])
                        times = times - 1
                    if times == 0:
                        break
                break
        saveLine = origin_item.strip() + '\t' + conll_lines[idx+1][9:].strip() + \
                        '\t' + str(posList) + '\n'
        saveFile.write(saveLine)        
    
