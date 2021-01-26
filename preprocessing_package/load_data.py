import pandas as pd


def load_data():
    df_1 = pd.read_csv('data/Datafiniti_Hotel_Reviews.csv')
    df_2 = pd.read_csv('data/Datafiniti_Hotel_Reviews_Jun19.csv')
    df_2 = df_2.drop(columns='reviews.dateAdded')
    df = pd.concat([df_1, df_2], ignore_index=True)
    print(df.shape)
    print(df.head())
    return df


if __name__ == "__main__":
    load_data()
