import codecs
import os
root = "./e2e4"
test1_path = "./e2e4/test.txt"
train1_path = "./e2e4/train.txt"
dct_path = "./e2e_result4.txt"
codecs.open(os.path.join(root, "test.txt"), "r+", "utf-8")
str1 = []
str2 = []
str_dump = []
fa = open(test1_path, 'r', encoding='utf-8')
fb = open(dct_path, 'r', encoding='utf-8')
fc = open(train1_path, 'w+', encoding='utf-8')

for line in fa.readlines():
    str1.append(line.replace("\n", ''))
for line in fb.readlines():
    str2.append(line.replace("\n", ''))

for i in str1:
    if i in str2:
        str_dump.append(i)

str_all = set(str1 + str2)

for i in str_dump:
    if i in str_all:
        str_all.remove(i)
for i in list(str_all):
    fc.write(i + '\n')

fa.close()
fb.close()
fc.close()