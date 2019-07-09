#%%
#https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import nltk as nltk
import sklearn as sklearn
import sklearn.naive_bayes as nb
import sklearn.svm as svm

#pd.read_json("dataset/reviews_Musical_Instruments_5.json.gz", lines=True)
#%%
# nltk.download('punkt')
# nltk.download('stopwords')

#%%
# 1. Preprocessing data
    # a. load text
    # b. tokenize by word
    # c. remove stop words
    # d. remove non alphabet text
    # e. word stemming or lemmatization
    # f. Vectorize the data using TFIDF method
Corpus = pd.read_csv("dataset/en/amazon_review_2_label.csv")

# drop empty rows
Corpus["text"].dropna(inplace=True)

# convert to lowercase
Corpus["text"] = [entry.lower() for entry in Corpus["text"]]


for index, entry in enumerate(Corpus["text"]):
    final_words = []
    words = nltk.word_tokenize(entry)
    for word in words:
        if word not in nltk.corpus.stopwords.words("english") and word.isalpha():
            # stemming is not done
            final_words.append(word)
    Corpus.loc[index, "text_final"] = str(final_words)

Train_X, Test_X, Train_Y, Test_Y = sklearn.model_selection.train_test_split(Corpus["text_final"], Corpus["label"], test_size=0.3)

Encoder = sklearn.preprocessing.LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

Tfidf_vect = sklearn.feature_extraction.text.TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus["text_final"])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#%%
# 1. Naive Bayes Classification
Naive = nb.MultinomialNB()
Naive.fit(Train_X_Tfidf, Train_Y)
predicted_nb = Naive.predict(Test_X_Tfidf)

sklearn.metrics.accuracy_score(Test_Y, predicted_nb) * 100

#%%
# 2. SVM Classification
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf, Train_Y)
predicted_svm = SVM.predict(Test_X_Tfidf)

sklearn.metrics.accuracy_score(Test_Y, predicted_svm) * 100

#%%
