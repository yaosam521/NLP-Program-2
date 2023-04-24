#use this file to train the ML Model
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y)
#svm = svm()

lr = LogisticRegression()

lr.fit(X_train, y_train)

y_test_hat = lr.predict(X_test)

print(accuracy_score(y_test,y_test_hat))