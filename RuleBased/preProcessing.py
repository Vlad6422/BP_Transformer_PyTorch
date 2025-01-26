import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
# Initialize the PorterStemmer to perform word stemming (reducing words to their base/root form).
stemer = PorterStemmer()

# Function to tokenize a sentence into individual words
# Tokenization splits a sentence into words or punctuation marks.
# Example: "I love coding!" -> ['I', 'love', 'coding', '!']
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Function to apply stemming to a given word
# Stemming reduces words to their base form.
# Example: "running" -> "run", "played" -> "play"
def stem(word):
    return stemer.stem(word.lower())


def bag_of_words(tokenized_sentence,all_words):
     # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
