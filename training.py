#use this file to train the ML Model
from sklearn import svm
from nltk.corpus import wordnet as wn

tag_set_yarn = wn.synsets('yarn', pos=wn.NOUN)
tag_set_tissue = wn.synsets('tissue', pos=wn.NOUN)
tag_set_rubbish = wn.synsets('rubbish', pos=wn.NOUN)

svm = svm()
