import re
import xarray as xr
import nltk
from nltk import bigrams
from nltk.corpus import stopwords
from nltk.probability import FreqDist
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
nltk.download('maxent_ne_chunker')
import spacy


stop_words = set(stopwords.words("english"))
vocab = set(nltk.corpus.words.words())

def netcdf_to_bag_of_words(path):
    ds = xr.open_dataset(path)
    
    tokenized_words =  []#str(path).split('/')[:-1]
    for attribute in ds.attrs:
        corpus = ds.attrs[attribute]

        # filter out non-strings
        if not isinstance(corpus, str):
            continue
        
        tokenizer = nltk.RegexpTokenizer(r"(\w+[.|\w]\w+@\w+[.]\w+[.|\w+]\w+|(?:[1-9]\d{3}-(?:(?:0[1-9]|1[0-2])-(?:0[1-9]|1\d|2[0-8])|(?:0[13-9]|1[0-2])-(?:29|30)|(?:0[13578]|1[02])-31)|(?:[1-9]\d(?:0[48]|[2468][048]|[13579][26])|(?:[2468][048]|[13579][26])00)-02-29)T(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:Z|[+-][01]\d:[0-5]\d)|\w+)")
        tokens = tokenizer.tokenize(corpus)
 
        # filter out stopwords
        tokens = map(lambda t: t.lower(), tokens)
        tokens = list(tokens)
        tokens = filter(lambda t: t not in stop_words, tokens)
        tokens = filter(lambda t: len(t) > 1, tokens)
        tokens = list(tokens)

        tokenized_words.extend(tokens)

    return FreqDist(tokenized_words)