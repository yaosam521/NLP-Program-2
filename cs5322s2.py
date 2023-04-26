from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import pandas as pd
import pickle

'''
Find information on WSD from Lexical Semantics (slides) and Chapter 23 (textbook)
'''
# Comment 1
def WSD_test_rubbish(text):
    output_file = open("./result_rubbish_yao.txt", mode='w')
    #Load the model
    model_tissue = pickle.load(open('model_rubbish.sav', 'rb'))

    #Put text into a document format
    vectorizer = pickle.load(open('rubbish_vectorizor.sav','rb'))
    matrix = vectorizer.fit_transform(text)
    df = pd.DataFrame(matrix.todense(), columns=vectorizer.get_feature_names_out())

    #Do PCA reduction and reduce to 7 components
    pca = PCA(n_components= 7)
    X_pca = pca.fit_transform(df)

    #predict
    for document in X_pca:
        prediction = model_tissue.predict([document])
        output_file.write(str(prediction[0]))
        output_file.write('\n')

    #print results of prediction to text file
    output_file.close()

def WSD_test_tissue(text):
    output_file = open("./result_tissue_yao.txt", mode='w')
    #Load the model
    model_tissue = pickle.load(open('model_tissue.sav', 'rb'))

    #Put text into a document format
    vectorizer = pickle.load(open('tissue_vectorizor.sav','rb'))
    matrix = vectorizer.fit_transform(text)
    df = pd.DataFrame(matrix.todense(), columns=vectorizer.get_feature_names_out())

    #Do PCA reduction and reduce to 7 components
    pca = PCA(n_components= 7)
    X_pca = pca.fit_transform(df)

    #predict
    for document in X_pca:
        prediction = model_tissue.predict([document])
        output_file.write(str(prediction[0]))
        output_file.write('\n')

    output_file.close()

def WSD_test_yarn(text):
    output_file = open("./result_yarn_yao.txt", mode='w')
    #Load the model
    model_tissue = pickle.load(open('model_yarn.sav', 'rb'))

    #Put text into a document format
    vectorizer = pickle.load(open('yarn_vectorizor.sav','rb'))
    matrix = vectorizer.fit_transform(text)
    df = pd.DataFrame(matrix.todense(), columns=vectorizer.get_feature_names_out())

    #Do PCA reduction and reduce to 7 components
    pca = PCA(n_components= 7)
    X_pca = pca.fit_transform(df)

    #predict
    for document in X_pca:
        prediction = model_tissue.predict([document])
        output_file.write(str(prediction[0]))
        output_file.write('\n')
