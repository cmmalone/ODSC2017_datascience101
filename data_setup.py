import os
import pandas as pd


def df_from_csv():
    """
    Order the well training features (well_data.csv)
    and associated labels (well_labels.csv) by date
    and return a dataframe with features and labels
    joined together
    """
    X_df = pd.read_csv("well_data.csv")
    y_df = pd.read_csv("well_labels.csv")
    df = pd.merge(X_df, y_df, on="id")

    # in the raw data, dates are encoded as strings
    # change to datetime for ordering
    date_strings = df['date_recorded'].tolist()
    date_strings = pd.Series(date_strings)
    date_datetimes = pd.to_datetime(date_strings)

    df['date_recorded'] = date_datetimes
    return df


def split_df(df):
    """
    Split the dataset into six approximately equal sized slices
    """
    df_list = []
    df_list.append(df.sort_values(by='date_recorded')[:10000])
    df_list.append(df.sort_values(by='date_recorded')[10000:20000])
    df_list.append(df.sort_values(by='date_recorded')[20000:30000])
    df_list.append(df.sort_values(by='date_recorded')[30000:40000])
    df_list.append(df.sort_values(by='date_recorded')[40000:50000])
    df_list.append(df.sort_values(by='date_recorded')[50000:])
    return df_list


def main():
    df = df_from_csv()
    df_list = split_df(df)

    # store resulting data files as follows:
    # datasets/dataset_1.csv
    # datasets/dataset_2.csv
    # etc.
    if not os.path.exists("datasets/"):
        os.makedirs("datasets/")
    for ii, df in enumerate(df_list):
        df.to_csv("datasets/dataset_"+str(ii+1)+".csv")


if __name__=="__main__":
    main()
