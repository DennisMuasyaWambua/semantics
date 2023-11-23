from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
import nltk


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

text = "I would like to inquire about the prices of your fridges"

tokens = word_tokenize(text)
print(tokens)
print("\n")
tags = pos_tag(tokens)
print(pos_tag(tokens))
print("\n")
print(ne_chunk(tags))

