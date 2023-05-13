import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers.legacy import SGD
#from tensorflow.keras.optimizers import SGD
import random

with open('intents.json') as file:
  intents = json.loads(file.read())

intents

words = []
classes = []
ignore_words = ['?', '!']
documents = []

words=[]
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']
for intent in intents['intents']:
   for pattern in intent['patterns']:
       #tokenize each word
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        #add documents in the corpus
        documents.append((word, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
          classes.append(intent['tag'])
print(documents)

   
#lemmatize/lower pour chaque mot /anullé duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_words]
words = sorted(list(set(words)))
#sortclasses
classes = sorted(list(set(classes)))
#documents = combination between patterns and intents
print(len(documents), "documents")
#classe = intent
print(len(classes), "classes", classes)

#words = all words, vocabulary
print(len(words), "unique lemmatized words", word)
pickle.dump(words,open('words.pkl', 'wb'))
pickle.dump(classes,open('classes.pkl','wb'))


# training data
training = []
# création d'une liste vide: pour l'output
output_empty = [0] * len(classes)
# training set, bag of words pour chaque phrase
for doc in documents:
    # initialisation du bag of words
    bag = []
    # tokenisation: listes des mots tokeniser pour le modèle
    pattern_words = doc[0]
    # lemmatisation pour chaque mot - créer un mot de base en éliminant les suffixes, 
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # création du tableau bag of word avec valeur de 1, si le mot est trouvé  dans le pattern actuel
    
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
    # output '0' pour chaque tag et un '1' pour le tag actuel(pour chaque pattern c'est a dire chaque élément de réponse)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)
#training = np.array(training)
# create training and testing lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("la création de Data Training est terminé")


# deep neural networds model

model = Sequential()

model.add(Dense(300, input_shape=(len(train_x[0]),), activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(80, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(len(train_y[0]), activation='softmax'))


sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("modèle créer")




