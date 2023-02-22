import pandas as pd
import numpy as np
from nltk.util import pr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword = set(stopwords.words("english"))

df = pd.read_csv("hate_speech.csv", encoding="windows_1258")
df.head()

df= df[["Tweet", "HS"]]

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|ww\.\S+', '', text)
    text = re.sub('<.?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text
df["Tweet"] = df["Tweet"].apply(clean)

x = np.array(df["Tweet"])
y = np.array(df["HS"])
cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

clf.score(X_test, y_test)

def hate_speech_detection():
    import streamlit as st
    st.title("Hate Speech etection")
    user = st.text_area("Enter any tweet: ")
    if len(user) < 1:
        st.write(" ")
    else:
        sample = user
        df= cv.transform([sample]).toarray()
        a = clf.predict(df)
        st.title(a)

hate_speech_detection()