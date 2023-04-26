import cs5322s2 as cs

f = open('./test_data/tissue_test_2.txt',mode='r')
corpus = list()
for line in f:
    corpus.append(line)
cs.WSD_test_tissue(corpus)
f.close()

f = open('./testp2.txt',mode='r')
corpus = list()
for line in f:
    corpus.append(line)
cs.WSD_test_rubbish(corpus)
f.close()

f = open('./yarn1.txt',mode='r')
corpus = list()
for line in f:
    corpus.append(line)
cs.WSD_test_yarn(corpus)
f.close()