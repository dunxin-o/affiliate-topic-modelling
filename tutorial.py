import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import spacy
import nltk
import re
import string
import pandas as pd
import numpy as np
import gensim
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
import json
import cleantext

from gensim import corpora
from nltk.corpus import stopwords
stopwords_list = stopwords.words('english')

data = []
for line in open('reviews_Office_Products_5.json', 'r'):
    data.append(json.loads(line))

df = pd.DataFrame(data)

reviews = df['reviewText']
reviews = cleantext.clean(reviews)


'''
Forming Bigrams & Trigrams
'''

bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = nltk.collocations.BigramCollocationFinder.from_documents(
    [review.split() for review in reviews]
)
finder.apply_freq_filter(50)
# Pointwise mutual information
trigram_scores = finder.score_ngrams(trigram_measures.pmi)