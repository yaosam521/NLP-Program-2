from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import pandas as pd
import pickle

'''
Find information on WSD from Lexical Semantics (slides) and Chapter 23 (textbook)
'''
# Comment 1
def WSD_test_rubbish(text):
    #Load the model
    model_tissue = pickle.load(open('model_rubbish.sav', 'rb'))

    #Put text into a document format
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(text)
    df = pd.DataFrame(matrix.todense(), columns=vectorizer.get_feature_names_out())

    #Do PCA reduction and reduce to 7 components
    pca = PCA(n_components= 7)
    X_pca = pca.fit_transform(df)

    results = list()
    #predict
    for document in X_pca:
        prediction = model_tissue.predict([document])
        results.append(prediction)

    #print results of prediction to text file
    return results

def WSD_test_tissue(text):
    #Load the model
    model_tissue = pickle.load(open('model_tissue.sav', 'rb'))

    #Put text into a document format
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(text)
    df = pd.DataFrame(matrix.todense(), columns=vectorizer.get_feature_names_out())

    #Do PCA reduction and reduce to 7 components
    pca = PCA(n_components= 7)
    X_pca = pca.fit_transform(df)

    results = list()
    #predict
    for document in X_pca:
        prediction = model_tissue.predict([document])
        results.append(prediction)

    #print results of prediction to text file
    return results

def WSD_test_yarn(text):
    #Load the model
    model_tissue = pickle.load(open('model_yarn.sav', 'rb'))

    #Put text into a document format
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(text)
    df = pd.DataFrame(matrix.todense(), columns=vectorizer.get_feature_names_out())

    #Do PCA reduction and reduce to 7 components
    pca = PCA(n_components= 7)
    X_pca = pca.fit_transform(df)

    results = list()
    #predict
    for document in X_pca:
        prediction = model_tissue.predict([document])
        results.append(prediction)

    #print results of prediction to text file
    return results
