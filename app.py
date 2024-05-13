import requests
import streamlit as st
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tensorflow
import random
import json
import tflearn
import os

# URLs of the intents file and the trained model in your GitHub repo
intents_url = "https://raw.githubusercontent.com/Swamisharan1/manaji-chatbot/master/skincare.json"
model_url = "https://github.com/Swamisharan1/manaji-chatbot/raw/master/model.tflearn"

# Fetch the intents file
response = requests.get(intents_url)
data = response.json()

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

# Fetch the trained model
response = requests.get(model_url, allow_redirects=True)
open('model.tflearn', 'wb').write(response.content)

# Load the trained model
net = tflearn.input_data(shape=[None, len(words)])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(labels), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load("model.tflearn")

# Function to convert a sentence into a bag of words
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

# Streamlit app
st.title("Chatbot")
input_text = st.text_input("You: ")

if input_text:
    # Predict the category
    results = model.predict([bag_of_words(input_text, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    # Select a response from the category
    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']

    st.write(random.choice(responses))
