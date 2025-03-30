# Author: Malashchuk Vladyslav
# File: preProcessing.py
# Description: This file contains the implementation of preprocessing functions for the chatbot.

import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
# Initialize the PorterStemmer to perform word stemming (reducing words to their base/root form).
stemer = PorterStemmer()

# Function to tokenize a sentence into individual words
# Tokenization splits a sentence into words or punctuation marks.
# Example: "I love coding!" -> ['I', 'love', 'coding', '!']
# This is useful for further processing like stemming or lemmatization.
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Function to apply stemming to a given word
# Stemming reduces words to their base form.
# Example: "running" -> "run", "played" -> "play"
def stem(word):
    return stemer.stem(word.lower())

# Function to create a bag of words representation for a given tokenized sentence
# and a list of all words in the dataset.
# The bag of words model represents a sentence as a vector of word counts or binary values.
# Each position in the vector corresponds to a word in the vocabulary.
# If a word is present in the sentence, its position in the vector is set to 1 (or its count).
# Otherwise, it is set to 0.
# Example: "I love coding!" with vocabulary ['I', 'love', 'coding', '!'] -> [1, 1, 1, 0]
# This is useful for converting text data into a numerical format suitable for machine learning models.
def bag_of_words(tokenized_sentence,all_words):
     # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
