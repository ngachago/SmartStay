import pandas as pd
from nltk.tokenize import word_tokenize

def vocab_richness(text):
    tokens = word_tokenize(text)
    total_length = len(tokens)
    unique_words = set(tokens)
    unique_word_length = len(unique_words)
    return unique_word_length / total_length

def richness_col(df, col, new_col):
    df_1 = df[df[col] != '']
    df_1[new_col] = df_1[col].apply(vocab_richness)
    print(df_1.shape)
    print(df_1.head())
    return df_1


if __name__ == "__main__":
    richness_col(df, col, new_col)
