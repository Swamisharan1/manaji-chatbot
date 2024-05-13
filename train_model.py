import requests
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import random
import json

# URLs of the intents file in your GitHub repo
intents_url = "https://raw.githubusercontent.com/Swamisharan1/manaji-chatbot/master/skincare.json"

# Fetch the intents file
response = requests.get(intents_url)
data = response.json()

# Preprocess the data
training = []
output = []

out_empty = [0 for _ in range(len(data["intents"]))]

for x, intent in enumerate(data["intents"]):
    for pattern in intent["patterns"]:
        tokenized_word = nltk.word_tokenize(pattern)
        words = [stemmer.stem(w.lower()) for w in tokenized_word]
        training.append(words)
        output_row = out_empty[:]
        output_row[x] = 1
        output.append(output_row)

# Convert training data to array
training = np.array(training)
output = np.array(output)

# Define the model
model = Sequential()
model.add(Dense(128, input_shape=(len(training[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(output[0]), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(training, output, epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save('model.h5')
