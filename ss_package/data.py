import pandas as pd
from urllib.parse import urlparse
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')


def load_data():
    df_1 = pd.read_csv('data/Datafiniti_Hotel_Reviews.csv')
    df_2 = pd.read_csv('data/Datafiniti_Hotel_Reviews_Jun19.csv')
    df_2 = df_2.drop(columns='reviews.dateAdded')
    df = pd.concat([df_1, df_2], ignore_index=True)
    print(df.shape)
    print(df.head())
    return df


class ProcCols:

    def __init__(self, df=df):
        self.df = df

    def missing_data(self):
        self.df = self.df.drop(columns=['reviews.userCity', 'reviews.userProvince'], inplace=True)
        self.df = self.df.dropna(inplace=True)
        return self.df

    def season(self, month):
        if month >= 1 and month <= 2:
            return 'winter'
        if month >= 3 and month <= 5:
            return 'spring'
        if month >= 6 and month <= 8:
            return 'summer'
        if month >= 9 and month <= 11:
            return 'fall'
        if month == 12:
            return 'winter'

    def time_data(self):
        self.df = self.df.drop(columns=['dateAdded', 'dateUpdated', 'reviews.dateSeen'], inplace=True)
        self.df['reviews.date'] = pd.to_datetime(self.df['reviews.date'])
        self.df['month'] = self.df['reviews.date'].dt.month
        self.df['year'] = self.df['reviews.date'].dt.year
        self.df = self.df.drop(columns='reviews.date', inplace=True)
        self.df['visit_season'] = self.df['month'].map(self.season)
        return self.df

    def review_map(self, rating):
        if rating >= 4:
            return 'great'
        if rating == 3:
            return 'ok'
        if rating < 3:
            return 'bad'

    def short_url(self, url):
        parsed = urlparse(url)
        short_url = parsed.netloc
        return short_url.replace('www.', '')

    def misc_processing(self):
        self.df = self.df.drop(columns='country')
        self.df['cat_rating'] = self.df['reviews.rating'].map(self.review_map)
        self.df['short_source'] = self.df['reviews.sourceURLs'].map(self.short_url)
        self.df = self.df.drop(columns=['sourceURLs', 'websites', 'address', 'keys', \
            'reviews.sourceURLs', 'primaryCategories', 'reviews.username'])
        return self.df

    def all_cols(self):
        self.df = self.missing_data()
        self.df = self.time_data()
        self.df = self.misc_processing()
        return self.df


class ProcText:

    def __init__(self, df=df, new_col, old_col):
        self.df = df
        self.new_col = new_col
        self.old_col = old_col

    def punc_remover(self):
        self.df[self.new_col] = self.df[self.old_col]
        for punctuation in string.punctuation:
            self.df[self.new_col] = self.df[self.new_col].map(lambda x: x.replace(punctuation, ''))
        return self.df

    def lower_case(self):
        self.df[self.new_col] = self.df[self.new_col].map(lambda x: x.lower())
        return self.df

    def remove_num(self):
        self.df[self.new_col] = self.df[self.new_col].map(lambda x: ''.join(word for word in x if not word.isdigit()))
        return self.df

    def remove_stop(self):
        stop_words = set(stopwords.words('english'))
        self.df[self.new_col] = self.df[self.new_col].map(word_tokenize)
        self.df[self.new_col] = self.df[self.new_col].map(lambda x: ' '.join(w for w in x if not w in stop_words))
        return self.df

    def lemmatize(self):
        self.df[self.new_col] = self.df[self.new_col].map(word_tokenize)
        lemmatizer = WordNetLemmatizer()
        self.df[self.new_col] = self.df[self.new_col].map(lambda x: ' '.join(lemmatizer.lemmatize(word) for word in x))
        return self.df

    def all_text(self):
        self.punc_remover()
        self.lower_case()
        self.remove_num()
        self.remove_stop()
        self.lemmatize()
        self.df = self.df.drop(columns=self.old_col, inplace=True)
        return self.df


class FeatEng:

    def __init__(self, df=df, col='clean_review', new_col='review_richness'):
        self.df = df
        self.col = col
        self.new_col = new_col

    def vocab_richness(self, text):
        tokens = word_tokenize(text)
        total_length = len(tokens)
        unique_words = set(tokens)
        unique_word_length = len(unique_words)
        return unique_word_length / total_length

    def richness_col(self):
        self.df = self.df[self.df[col] != '']
        self.df[self.new_col] = self.df[self.col].apply(self.vocab_richness)
        return self.df
