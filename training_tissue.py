#use this file to train the ML Model
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import numpy as np
import pickle

corpus = list()
target_column = list()

f = open('./tissue1.txt',mode='r')
for line in f:
    target_column.append(1)
    corpus.append(line)
f.close()

f = open('./tissue2.txt',mode='r')
for line in f:
    target_column.append(2)
    corpus.append(line)
f.close()

vectorizor = TfidfVectorizer(stop_words='english')
bruh = vectorizor.fit_transform(corpus)
df = pd.DataFrame(bruh.todense(), columns=vectorizor.get_feature_names_out())
df['target_sense'] = target_column
X = df.loc[:, df.columns != 'target_sense']
y = df['target_sense']

#PCA reduction to 7 components
pca = PCA(n_components=7)
pca.fit(X)

X_pca = pca.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_pca,y, test_size=0.2, stratify=y,shuffle=True)
clf_tissue = svm.LinearSVC(penalty="l1",dual=False)
clf_tissue.fit(X_train, y_train)
y_test_hat = clf_tissue.predict(X_test)

kf = KFold(n_splits=4, shuffle=True)
acc_train=[]
acc_test=[]
for fold, (train_i, test_i) in enumerate(kf.split(X_pca)):
    X_train, X_test = X_pca[train_i], X_pca[test_i]
    y_train, y_test = y[train_i], y[test_i]
    clf_tissue.fit(X_train, y_train)
    y_hat_train = clf_tissue.predict(X_train)
    train_acc = accuracy_score(y_train, y_hat_train)
    acc_train.append(train_acc)
    
    y_hat_test = clf_tissue.predict(X_test)
    test_acc = accuracy_score(y_test, y_hat_test)
    acc_test.append(test_acc)

avg_tr=0 
avg_te=0
for tr_rec, te_rec in zip(acc_train, acc_test):
    avg_tr += np.mean(tr_rec)
    avg_te += np.mean(te_rec)
avg_tr =avg_tr/len(acc_train)
avg_te = avg_te/len(acc_test)
print('Average Accuracy Train:{:0.2f} Accuracy Test:{:0.2f}'.format(avg_tr,avg_te))

filename = 'model_tissue.sav'
pickle.dump(clf_tissue, open(filename, 'wb'))

# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
