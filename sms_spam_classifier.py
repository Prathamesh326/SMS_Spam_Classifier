import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from collections import Counter
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
import pickle

df = pd.read_csv('E:\Spam Mini Project\SMS_SPAM_CLASSIFIER\spam.csv', encoding = 'latin-1')
df.sample(5)

df.shape

"""Data Cleaning"""

df.info()

df.drop(columns = ['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace = True)

df.sample(5)

df.rename(columns= {'v1':'target','v2':'text'}, inplace = True)

df.sample(5)


encoder = LabelEncoder()

df['target'] = encoder.fit_transform(df['target'])

df.head()

df.isnull().sum()

df.duplicated().sum()

df = df.drop_duplicates(keep = 'first')
df.duplicated().sum()

df.shape

"""EDA"""

df['target'].value_counts()


plt.pie(df['target'].value_counts(), labels=['ham','spam'], autopct = '%0.2f')
plt.show()



nltk.download('punkt')

#number of character
df['num_char'] = df['text'].apply(len)

df.head()

df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))

df.head(10)

df['num_sent'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

df.head(5)

df[['num_char','num_words','num_sent']].describe()

#HAM
df[df['target'] == 0][['num_char','num_words','num_sent']].describe()

#SPAM
df[df['target'] == 1][['num_char','num_words','num_sent']].describe()



sns.histplot(df[df['target'] == 0]['num_char'])
sns.histplot(df[df['target'] == 1]['num_char'], color ='red')

sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'], color ='red')

sns.pairplot(df, hue='target')

sns.heatmap(df.corr(), annot = True)

"""TEXT PREPROCESSING"""

def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  y = []
  for i in text :
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)

nltk.download('stopwords')


stopwords.words('english')


string.punctuation

ps = PorterStemmer()
ps.stem('loving')

transform_text('Hi How Are You 20@ ???.My name is Pratzz')

df['transformed_text'] = df['text'].apply(transform_text)

df.head(10)

wc = WordCloud(width = 500, height = 500, min_font_size = 10, background_color='white')

spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep = " "))

plt.figure(figsize=(15,8))
plt.imshow(spam_wc)

ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep = " "))
plt.figure(figsize=(15,8))
plt.imshow(ham_wc)

df.head(10)

spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist() :
  for word in msg.split():
    spam_corpus.append(word)

len(spam_corpus)

sns.barplot(x=pd.DataFrame(Counter(spam_corpus).most_common(30))[0],y=pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()

ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist() :
  for word in msg.split():
    ham_corpus.append(word)

sns.barplot(x=pd.DataFrame(Counter(ham_corpus).most_common(30))[0],y=pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()

"""Model Building"""

#cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features = 3000)

#X = cv.fit_transform(df['transformed_text']).toarray()
x = tfidf.fit_transform(df['transformed_text']).toarray()

Y = df['target'].values


X_train, X_test, y_train, y_test = train_test_split(x,Y,test_size=0.2,random_state=2)


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))

mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))

bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))

"""TFIDF with MNB"""


svc = SVC(kernel = 'sigmoid', gamma=1.0)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

clfs = {
    'SVC' : svc,
    'NB' : mnb,
    'ETC' : etc
}

def train_classifier(clf, X_train,y_train, X_test,y_test) :
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  precision_val = precision_score(y_test, y_pred)

  return accuracy, precision_val

accuracy_scores = []
precision_scores = []

for name, clf in clfs.items():

  current_accuracy,current_precision = train_classifier(clf, X_train, y_train , X_test, y_test)

  print('For', name)
  print('Accuracy - ', current_accuracy)
  print('Precision - ', current_precision)

  accuracy_scores.append(current_accuracy)
  precision_scores.append(current_precision)

performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)

performance_df

pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))