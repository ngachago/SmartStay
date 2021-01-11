import pandas as pd
from urllib.parse import urlparse


def missing_data(df):
    df_1 = df.drop(columns=['reviews.userCity', 'reviews.userProvince'])
    df_2 = df_1.dropna()
    return df_2

def season(month):
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

def time_data(df):
    df_1 = df.drop(columns=['dateAdded', 'dateUpdated', 'reviews.dateSeen'])
    df_1['reviews.date'] = pd.to_datetime(df_1['reviews.date'])
    df_1['month'] = df_1['reviews.date'].dt.month
    df_1['year'] = df_1['reviews.date'].dt.year
    df_2 = df_1.drop(columns='reviews.date')
    df_2['visit_season'] = df_2['month'].map(season)
    return df_2

def review_map(rating):
    if rating > 3:
        return 'great'
    if rating == 3:
        return 'ok'
    if rating < 3:
        return 'bad'

def short_url(url):
    parsed = urlparse(url)
    short_url = parsed.netloc
    return short_url.replace('www.', '')

def misc_processing(df):
    df_1 = df.drop(columns='country')
    df_1['cat_rating'] = df_1['reviews.rating'].map(review_map)
    df_1['short_source'] = df_1['reviews.sourceURLs'].map(short_url)
    df_2 = df_1.drop(columns=['sourceURLs', 'websites', 'address', 'keys', \
        'reviews.sourceURLs', 'primaryCategories', 'reviews.username'])
    return df_2

def all_cols(df):
    df_1 = missing_data(df)
    df_2 = time_data(df_1)
    df_3 = misc_processing(df_2)
    print(df_3.shape)
    print(df_3.head())
    return df_3


if __name__ == "__main__":
    all_cols(df)
