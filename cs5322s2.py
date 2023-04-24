'''
Find information on WSD from Lexical Semantics (slides) and Chapter 23 (textbook)
'''
# Comment 1
def WSD_test_rubbish(text,model):
    results = list()
    #preprocess the text
    for sentence in text:
        #classify word in sentence (load model in here)
        prediction = model.predict(sentence)
        results.append(prediction)
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
