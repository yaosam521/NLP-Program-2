import cs5322s2 as cs

f = open('./test_data/tissue_test_2.txt',mode='r')
corpus = list()
for line in f:
    corpus.append(line)
results = cs.WSD_test_tissue(corpus)
print(results)
f.close()

f = open('./testp2.txt',mode='r')
corpus = list()
for line in f:
    corpus.append(line)
results = cs.WSD_test_rubbish(corpus)
print(results)
f.close()