import xmltodict
import json
import os 
import sys
import xml.etree.ElementTree as ET
import numpy as np

f= open("txt/dctt.txt","r")
dctt = []
dctStr = []
for item in f.readlines():
    dctt.append(item.split('\t'))
    dctStr.append(item)
f.close()

f= open("txt/e2et.txt","r")
e2e = []
e2eStr = []
for item in f.readlines():
    e2e.append(item.split('\t'))
    e2eStr.append(item)
f.close()

f= open("txt/t2et.txt","r")
t2e = []
t2eStr = []
for item in f.readlines():
    t2e.append(item.split('\t'))
    t2eStr.append(item)
f.close()

f= open("txt/matt.txt","r")
mat = []
matStr = []
for item in f.readlines():
    mat.append(item.split('\t'))
    matStr.append(item)
f.close()


curPath = 'xml/'
for curDir, dirs, files in os.walk(curPath):
    pass
files.sort()
DCT_path ='txt/' + 'DCT.txt'
DCT = open(DCT_path, 'w+')

E2E_path ='txt/' + 'E2E.txt'
E2E = open(E2E_path, 'w+')

T2E_path ='txt/' + 'T2E.txt'
T2E = open(T2E_path, 'w+')

MAT_path ='txt/' + 'MAT.txt'
MAT = open(MAT_path, 'w+')

for xml in files:
    if xml.endswith('.xml'):
        

        tree = ET.parse('xml/'+xml)
        root = tree.getroot()
        es1 = root.findall('.//EVENT')
        tLink = root.findall('.//TLINK')
        
        for e2 in tLink:
            ev = str(e2.get('task'))
            reltypeA = str(e2.get('relTypeA'))
            reltypeB = str(e2.get('relTypeB'))
            reltypeC = str(e2.get('relTypeC'))
            time = str(e2.get('timeID'))
            event = str(e2.get('relatedToEventInstance')).replace('ei', 'e')
            if ev == 'DCT':
                if reltypeA == reltypeB == reltypeC:
                    line = event + '\t' + reltypeA
                    text = line + '\n'
                    for idx, item in enumerate(dctt):
                        if item[0] == xml[:5]:
                            if item[2] == event:
                                DCT.write(dctStr[idx])
            elif ev == 'E2E':
                if reltypeA == reltypeB == reltypeC:
                    f_event = str(e2.get('eventInstanceID')).replace('ei','e')
                    s_event = str(e2.get('relatedToEventInstance')).replace('ei','e')
                    line = f_event + '\t' + s_event + '\t' + reltypeA
                    text = line + '\n'
                    for idx, item in enumerate(e2e):
                        if item[0][2:4] == xml[3:5]:
                            if item[3] == f_event and item[7] == s_event:
                                E2E.write(e2eStr[idx])
            elif ev == 'T2E':
                if reltypeA == reltypeB == reltypeC:
                    time = str(e2.get('timeID'))
                    line = time + '\t' + event + '\t' + reltypeA
                    text = line + '\n'
                    for idx, item in enumerate(t2e):
                        if item[0][2:4] == xml[3:5]:
                            if item[3] == time and item[7] == event:
                                T2E.write(t2eStr[idx])
            elif ev == 'MAT':
                if reltypeA == reltypeB and reltypeB == reltypeC:
                    f_event = str(e2.get('eventInstanceID')).replace('ei','e')
                    s_event = str(e2.get('relatedToEventInstance')).replace('ei','e')
                    line = f_event + '\t' + s_event + '\t' + reltypeA
                    text = line + '\n'
                    for idx, item in enumerate(mat):
                        if item[0][2:4] == xml[3:5]:
                            if item[3] == f_event and item[10] == s_event:
                                MAT.write(matStr[idx])
            else:
                continue



