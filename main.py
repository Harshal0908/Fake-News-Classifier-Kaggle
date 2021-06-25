import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#importing the dataset
df = pd.read_csv(r'E:\Natural Language Processing Projects\Fake news Classifier\train.csv')

#droping NA values
df = df.dropna()

X = df.drop('label',axis=1)
Y = df['label']

voc_size = 5000

messages = X.copy()
messages.reset_index(inplace=True)
ps = PorterStemmer()
corpus = []

for i in range(0,len(messages)):
    #removing special charachters
    review = re.sub('[^a-zA-Z]',' ',messages['title'][i])
    
    #converting all letters in lowercase
    review = review.lower()
    
    #spliting all the words
    review = review.split()
    
    #removing stopwords
    review = [(word) for word in review if not word in stopwords.words('english')]
    
    #joining the words with " ", creating the sentences and appending it in corpus
    review = ' '.join(review)

    corpus.append(review)
#all the words in corpus is represented in one hot representation
onehot_repr=[one_hot(words,voc_size) for words in corpus]

#Embedded representation of one hot representation
sent_length = 20
embedded_docs = pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)


#creating Neural network model
embedding_vector_feature = 40
model = Sequential()
model.add(Embedding(voc_size,embedding_vector_feature,input_length=sent_length))
model.add(Bidirectional(LSTM(100)))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

X_final = np.array(embedded_docs)
Y_final = np.array(Y)

X_train,X_test,Y_train,Y_test = train_test_split(X_final,Y_final, test_size=0.33,random_state=42)

#TRAINING THE MODEL
print("TRAINING THE MODEL")

model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=10,batch_size=64)

