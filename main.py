import pprint
import re
import pandas as pd
import pickle
import nltk
from nltk import (
        pos_tag,
        word_tokenize,
        FreqDist,
        WordNetLemmatizer,
        WhitespaceTokenizer,
        RegexpTokenizer,
    )
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

corpus_path = '/Users/Dun/repo/scrapy_webscraping/contents.json'
data = pd.read_json(corpus_path, lines=True)
data = data[~data['content'].isna()]
data['content length'] = data.apply(lambda x: len(x['content'].split()), axis=1)
# Exclude links with less than 50 words
data = data[data['content length']>=50]

# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')

# Process data
# Note that CountVectorizer does the following sequence:
#   1. Preprocessing
#   2. Tokenizer
#   3. N-grams generation
#   4. Stop words removal

class customTokenizer(object):

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        tokenize = RegexpTokenizer(r"[a-zA-Z']+").tokenize
        # tokenizer = WhitespaceTokenizer().tokenize
        # output = tokenize(doc)
        output = []
        for token, tag in pos_tag(tokenize(doc)):
            pos = tag[0].lower()
            if pos in ['n', 'v']:
                if len(token) > 1:
                    output.append(self.wnl.lemmatize(token, 'n').lower())

        return output

def get_top_words(topic_word_weights, top_n=10):
    output = pd.DataFrame(index=range(topic_word_weights.shape[0]), columns=range(top_n))
    for i in range(topic_word_weights.shape[0]):
        arr = topic_word_weights.iloc[i]
        arr = arr.sort_values(ascending=False)
        output.iloc[i] = arr.index.values[:top_n]

    return output

vectorizer = CountVectorizer(
    stop_words=stopwords.words('english'),
    ngram_range=(1,2),
    max_df=0.9,
    min_df=0.05,
    tokenizer=customTokenizer()
)

data_processed = vectorizer.fit_transform(data['content'])
corpus_vocabulary = vectorizer.get_feature_names_out()

X_train, X_test = train_test_split(data_processed, test_size=0.1, random_state=0)
LDA = LatentDirichletAllocation(
        n_components=10,
        max_iter=200,
        random_state=0,
        doc_topic_prior=0.08,
        topic_word_prior=0.08
    )
LDA.fit(X_train)
topic_word_weightages = pd.DataFrame(LDA.components_, columns=vectorizer.get_feature_names_out())
out = get_top_words(topic_word_weightages, top_n=20)
for i in range(out.shape[0]):
    print(out.iloc[i])


etl_filename = 'etl.pkl'
model_filename = 'model.pkl'

with open(etl_filename, 'wb') as f:
    pickle.dump(vectorizer, f)

with open(model_filename, 'wb') as f:
    pickle.dump(LDA, f)


