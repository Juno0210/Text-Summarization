# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 07:17:56 2023

@author: Mick
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
import torch
import transformers
# import tensorflow as tf
from scipy.sparse.linalg import svds

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer,
                                             TfidfVectorizer)
from transformers import BertTokenizer, BertModel

from nltk.corpus import stopwords
from string import punctuation


#Here are the documents that we will be using:
with open('sample.txt', encoding='utf-8') as f:
    lines = f.readlines()

DOCUMENT = str(lines)

DOCUMENT = re.sub(r'\n|\r', ' ', DOCUMENT)
DOCUMENT = re.sub(r' +', ' ', DOCUMENT)
DOCUMENT = DOCUMENT.strip()

sentences = sent_tokenize(DOCUMENT)

#Set stop words to english
stop_words = set(stopwords.words('english') + list(punctuation))


'''
We will first remove all special characters, 
excess whitespace, and stopwords from the documents.
'''

def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = word_tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)

norm_sentences = normalize_corpus(sentences)


#TF-IDF keyword extract
tfidf_vectorizor = TfidfVectorizer(min_df=0., max_df=1., ngram_range=(1,3), use_idf=True)
tfidf_matrix = tfidf_vectorizor.fit_transform(norm_sentences)
tfidf_matrix = tfidf_matrix.toarray()

#Get feature names
feature_names = tfidf_vectorizor.get_feature_names_out()
tfidf_matrix_df = tfidf_matrix.T

# pd.DataFrame(tfidf_matrix.toarray(), index=feature_names).head(10)
pd.DataFrame(np.round(tfidf_matrix_df, 2), index=feature_names).head(10)

# tfidf_matrix_df = tfidf_matrix_df.rename(columns = {0: 'score'})
# tfidf_matrix_df['word'] = tfidf_matrix_df.index
# tfidf_matrix_df = tfidf_matrix_df.sort_values('score', ascending = False)


#BERT encorder
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
marked_doc = "[CLS]" + DOCUMENT + "[SEP]"
sent_embds = tokenizer.tokenize(marked_doc)
segments_ids = [1] * len(sent_embds)

# Map the token strings to their vocabulary indeces.
indexed_tokens = tokenizer.convert_tokens_to_ids(sent_embds)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

def low_rank_svd(matrix, singular_count=2):
    u, s, vt = svds(matrix, k=singular_count)
    return u, s, vt

num_sentences = 5
num_topics = 5

u, s, vt = low_rank_svd(tfidf_matrix_df, singular_count=num_topics)  

term_topic_mat, singular_values, topic_document_mat = u, s, vt

sv_threshold = 0.5
min_sigma_value = max(singular_values) * sv_threshold
singular_values[singular_values < min_sigma_value] = 0

salience_scores = np.sqrt(np.dot(np.square(singular_values), 
                                 np.square(topic_document_mat)))
print(salience_scores)

top_sentence_indices = (-salience_scores).argsort()[:num_sentences]
top_sentence_indices.sort()
print(type(top_sentence_indices))
for index in top_sentence_indices:
    print(len(norm_sentences))
    print (norm_sentences[index])
