from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

'''
Find information on WSD from Lexical Semantics (slides) and Chapter 23 (textbook)
'''
# Comment 1
def WSD_test_rubbish(text):
    vectorizer = TfidfVectorizer(stop_words='english')
    rubbish_matrix = vectorizer.fit_transform(text)
    df_rubbish = pd.DataFrame(rubbish_matrix.todense(), columns=vectorizer.get_feature_names_out())
    results = list()
    #preprocess the text
    return results

def WSD_test_tissue(text,model):
    results = list()
    for sentence in text:
        #classify word in sentence (load model in here)
        prediction = model.predict(sentence)
        results.append(prediction)
    return results

def WSD_test_yarn(text,model):
    results = list()
    for sentence in text:
        #classify word in sentence (load model in here)
        prediction = model.predict(sentence)
        results.append(prediction)
    return results
