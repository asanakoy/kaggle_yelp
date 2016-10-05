import pandas as pd
import json
import os
import numpy as np
from nltk.tokenize import RegexpTokenizer
from data_reader import *


def extract_features(df, filepath):
    with open(os.path.join(DATA_ROOT, 'swearWords.txt')) as f:
        swear_words = f.readlines()
        swear_words = map(lambda x: x.strip(), swear_words)

    import datetime as dt
    text_lower = df.text.str.lower()
    age = df['date'] - dt.datetime.now().date()
    age = map(lambda x: x.days, age)

    feats = pd.DataFrame(index=df.index)
    feats['age'] = age
    feats['stars'] = df.stars.astype(int)
    print 'Pos Smiles ...'
    pos_smiles = [r'\(:', r':\)', r'=\)', r'\(=', r'\^\^', r':-\)', r'\(-:', r':D', 'xD', ':-D', ':P', ':p', ':3',
                  r':\]', r'\[:', '8D', '=D']
    pos_smiles.extend([x.replace(':', ';') for x in pos_smiles if x.count(':') == 1])
    pos_smile_pattern = r'(?:{})'.format(pos_smiles[0])
    for sm in pos_smiles[1:]:
        pos_smile_pattern += r'|(?:{})'.format(sm)
    feats['pos_smiles'] = df.text.str.count(pos_smile_pattern)

    print 'Neg Smiles ...'
    neg_smiles = ([x.replace(')', '(') for x in pos_smiles if x.count(')') == 1])
    neg_smiles.extend([x.replace('(', ')') for x in pos_smiles if x.count('(') == 1])
    neg_smiles.extend([x.replace('[', ']') for x in pos_smiles if x.count('[') == 1])
    neg_smiles.extend([x.replace(']', '[') for x in pos_smiles if x.count(']') == 1])
    neg_smiles.extend([':c', ';c'])
    neg_smile_pattern = r'(?:{})'.format(neg_smiles[0])
    for sm in neg_smiles[1:]:
        neg_smile_pattern += r'|(?:{})'.format(sm)
    feats['neg_smiles'] = df.text.str.count(neg_smile_pattern)
    feats['smiles'] = feats['pos_smiles'] + feats['neg_smiles']

    print 'keywords'
    keywords = [r'\$', r'http://www', 'low price', 'high price', 'price', 'expensive', 'cheap', r'!!+', r'\?\?+',
                r'[^!]![^!]', r'[^\?]\?[^\?]', r'\.\.+', 'love']
    for i, pattern in enumerate(keywords):
        feats['keyword_{}'.format(i)] = text_lower.str.count(pattern)

    print 'sentences_cnt ...'
    feats['sentences_cnt'] = df.text.str.count(r'\w\.(?:[^.0-9]|$)').apply(lambda x: x if x != 0 else 1)
    print 'word_count ...'
    feats['word_count'] = df.text.apply(count_words).astype(float)
    feats['text_len'] = df.text.apply(len).astype(float)
    assert np.all(feats['text_len'] > 0)
    feats['avg_sentence_words'] = feats['word_count'] / feats['sentences_cnt']
    feats['avg_word_len'] = (feats['text_len'] - feats['word_count']) / feats['word_count']
    feats['avg_word_len'] = feats['avg_word_len'].apply(lambda x: 0 if np.isinf(x) else x)
    feats['capitals_cnt'] = df.text.str.count('[A-Z]').astype(float)
    feats['capitals_cnt_over_sent_cnt'] = feats['capitals_cnt'] / feats['sentences_cnt']
    feats['capitals_cnt_over_text_len'] = feats['capitals_cnt'] / feats['text_len']

    print 'swear_count ...'
    swear_pattern = r'(?:\W{}(?:\W|$))'.format(swear_words[0])
    for sm in swear_words[1:]:
        swear_pattern += r'|(?:\W{}(?:\W|$))'.format(sm)
    feats['swear_count'] = df.text.str.count(swear_pattern)

    print 'Saving...'
    feats.to_hdf(filepath, 'df', mode='w')
    return feats


def count_words(text, tokenizer=RegexpTokenizer(r'\w+')):
    import re
    text = re.sub(r'\d+\.\d+', '',  text)
    tokens = tokenizer.tokenize(text)
    return len(tokens)

