from flask import Flask, request, jsonify
from flask_cors import CORS

import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from tensorflow.keras.models import load_model

model = load_model('chatbot_model.h5')
import json
import random

intents = json.loads(open('intents.json', encoding="utf-8").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


def clean_up_sentence(sentence):
    # tokenize the pattern- split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word -create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for words that exist in sentence
def bow(sentence, words, show_details=True):
    # tokenizing patterns
    sentence_words = clean_up_sentence(sentence)
    # bag of words - lmatrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % word)
    return (np.array(bag))


def predict_class(sentence, model):
    # filter below  threshold predictions
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sorting strength probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


app = Flask(__name__)
cors = CORS(app)


@app.route('/askAssistant', methods=['GET', 'POST'])
def answerQuestion():
    if request.method == 'POST':
        question = request.form['question']
    elif request.method == 'GET':
        question = request.args.get('question', 'World')
    else:
        response = {
            'erreur': 'Incorrect Method'
        }
        return jsonify(response)

    if question != '':
        answer = chatbot_response(question)
        response = {
            'answer': f'{answer}'
        }
    else:
        response = {
            'answer': 'No Question was asked!'
        }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
