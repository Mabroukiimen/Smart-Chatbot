import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

# from sqlite3.dbapi2 import *
# from _sqlite3 import *
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import csv

with open("intents.json") as file:
    data = json.load(file)
print(data["intents"])

    # try:
    # with open("data.pickle", "rb") as f:
    # words, labels, training, output = pickle.load(f)
    # except:
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

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

    # with open("data.pickle", "wb") as f:
    # pickle.dump((words, labels, training, output), f)

    # this is the complete ai model

tensorflow.compat.v1.reset_default_graph()  # getting ready of all the previous settings

net = tflearn.input_data(shape=[None, len(training[
                                                  0])])  # defines the input shape expected to our model #in this case all the inputs have the same lengths that's why we entered  len(training[0])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")  # the output layer will have 6 neurons because we have 6 tags in our json file
net = tflearn.regression(net)
'''
we will pick a random response from the highest predicted model 

'''

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)  # n_epoch is the number of times it's gonna see the same data  #the more it sees the data, the better it gets at classifying #batch_size represents the number of training samples to work through
model.save("model.tflearn")  # the model is going to be saved as model.tflearn


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)



app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
@app.route('/api/search/smart-agent/search/<inp>', methods=['GET', 'POST'])
def chat(inp):
    results = model.predict([bag_of_words(inp, words)])
    # print(results)
    results_index = numpy.argmax(results)  # this gives us the index of our greatest value in our list "results"
    tag = labels[results_index]
    # print(tag)

    if results[0][results_index] > 0.5:
        for tg in data["intents"]:

            if tg['tag'] == tag:
                responses = tg['responses']

        result = random.choice(responses)
        message = {"answer": [{"_type": "dialog", "message": result}]}  # dialog

    else:
        message = {"answer": [{"_type": "error", "message": "Sorry, i don't get what you mean"}]}

    return jsonify(message)



app.run(debug=True, port=9090)









