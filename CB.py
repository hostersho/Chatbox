import numpy as np
import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Activation
# from keras.layers import Dropout
# from keras.optimizers import SGD
import random
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

# load json
read = open('CBPhoVan.json')
file = json.load(read)

# Tokenize json
words=[]
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']

for file in file['intents']:
    for pattern in file['patterns']:
        word = nltk.word_tokenize(pattern)        #tokenize each word
        words.extend(word)
        documents.append((word, file['tag']))   #add documents in the corpus
        if file['tag'] not in classes:          # add to our classes list
            classes.append(file['tag'])
# print(documents)

# lemmatize words to condense vocabulary
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))                  # sort classes
print (len(documents), "documents")                   # documents = combination between patterns and intents
print (len(classes), "classes", classes)              # classes = intents
print (len(words), "unique lemmatized words", words)  # words = all words, vocabulary

# Create pickle file for the lemmatized and tokenized version of the json
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# Start training the data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
random.shuffle(training)
training = np.array(training)
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data is created")

# 3 layer keras model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

# sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# rms = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# ad = tf.keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)   worst accuracy
# admax = tf.keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Training and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model is created")