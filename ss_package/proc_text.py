import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')


def punc_remover(df, new_col, old_col):
    df[new_col] = df[old_col]
    for punctuation in string.punctuation:
        df[new_col] = df[new_col].map(lambda x: x.replace(punctuation, ''))
    return df

def lower_case(df, new_col):
    df[new_col] = df[new_col].map(lambda x: x.lower())
    return df

def remove_num(df, new_col):
    df[new_col] = df[new_col].map(lambda x: ''.join(word for word in x if not word.isdigit()))
    return df

def remove_stop(df, new_col):
    stop_words = set(stopwords.words('english'))
    df[new_col] = df[new_col].map(word_tokenize)
    df[new_col] = df[new_col].map(lambda x: ' '.join(w for w in x if not w in stop_words))
    return df

def lemmatize(df, new_col):
    df[new_col] = df[new_col].map(word_tokenize)
    lemmatizer = WordNetLemmatizer()
    df[new_col] = df[new_col].map(lambda x: ' '.join(lemmatizer.lemmatize(word) for word in x))
    return df

def all_text(df, new_col, old_col):
    punc_remover(df, new_col, old_col)
    lower_case(df, new_col)
    remove_num(df, new_col)
    remove_stop(df, new_col)
    lemmatize(df, new_col)
    df = df.drop(columns=old_col)
    print(df.shape)
    print(df.head())
    return df


if __name__ == "__main__":
    all_text(df)
