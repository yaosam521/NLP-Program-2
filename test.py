import cs5322s2 as cs

f = open('./rubbish1.txt',mode='r')
corpus = list()
for line in f:
    corpus.append(line)
cs.WSD_test_rubbish(corpus)
f.close()

f = open('./yarn2.txt',mode='r')
corpus = list()
for line in f:
    corpus.append(line)
cs.WSD_test_yarn(corpus)
f.close()

f = open('./tissue1.txt',mode='r')
corpus = list()
for line in f:
    corpus.append(line)
cs.WSD_test_tissue(corpus)
f.close()