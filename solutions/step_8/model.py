from pathlib import Path
import logging
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_cols_to_dummy():
    cols_to_dummy = ["basin", "region", "lga", "recorded_by",
                "extraction_type", "extraction_type_group",
                "extraction_type_class",
                "management", "management_group",
                "payment", "payment_type",
                "water_quality", "quality_group", "quantity", "quantity_group",
                "source", "source_type", "source_class",
                "waterpoint_type", "waterpoint_type_group"]
    return cols_to_dummy


def get_cols_to_drop():
    cols_to_drop = ["date_recorded", "funder", "installer", "wpt_name",
                    "subvillage", "ward", "public_meeting", "permit",
                    "scheme_name", "scheme_management"]
    return cols_to_drop


def handle_data_files(infile_name):
    """
    For a given infile name, open the file into a 
    pandas Dataframe and determinately split into training
    and testing subsets. Save those as
    `train_X.csv` and `test_X.csv` in the `datasets`
    directory (where `X` is the integer index of the
    dataset and should be present in infile_name)

    Params
    ------
    infile_name : str
        name of the input dataset, standard format
        is dataset_X.csv, where "X" is an integer index
    
    Returns
    -------
    train_df, pd.DataFrame
        80% of input data, for training
    test_df, pd.DataFrame
        20% of input data, for testing
    """
    # assume filename format: dataset_X.csv where
    # X is an integer that indexes the file
    df = pd.read_csv(infile_name)

    dataset_id = infile_name.split(".csv")[0].split("_")[1]
    train_name = "../../datasets/train_{}.csv".format(dataset_id)
    test_name = "../../datasets/test_{}.csv".format(dataset_id)

    train_file = Path(train_name)
    if train_file.is_file():
        logger.info("found {}, retrieving".format(train_name))
        train_df = pd.read_csv(train_name)
        test_df = pd.read_csv(test_name)
    else:
        logger.info("{} not found, creating".format(train_name))
        df = pd.read_csv(infile_name)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        # index=False prevents an unwanted column named "Unnamed: 0"
        # from being appended 
        # https://stackoverflow.com/questions/36519086/pandas-how-to-get-rid-of-unnamed-column-in-a-dataframe
        train_df.to_csv(train_name, index=False)
        test_df.to_csv(test_name, index=False)

    return train_df, test_df


def encode_labels(df, col="status_group"):
    """
    Takes a df with an index and the status_group column
    filled with strings ("functional", "functional needs repair",
    "non-functional") and returns integer-encoded labels
    column

    Params
        df: pd.DataFrame

    Returns
        pd.DataFrame, with integer-encoded "status_group" column
    """
    labels = df[col].tolist()
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    df[col] = labels_encoded
    logger.info("encoding labels complete")
    return df


def build_model(df_train):
    """
    Build a model with df_train as the training data.
    
    Params
    ------
    df_train, pd.DataFrame
        training dataset

    Returns
    -------
    sklearn.ModelPipeline
        trained model
    """
    cols_to_dummy = get_cols_to_dummy()
    cols_to_drop = get_cols_to_drop()
    df_train = df_train.drop(cols_to_drop, axis=1)
    logger.info("dropped columns")
    logger.info(cols_to_drop)

    # dummying!
    for column_name in cols_to_dummy:
        column = np.array(df_train[column_name].tolist())
        le = LabelEncoder()
        int_column = le.fit_transform(column).reshape(-1, 1)
        enc = OneHotEncoder(handle_unknown="ignore")
        new_column = enc.fit_transform(int_column).toarray()   
   
        # once we've dummied, want to remove original col 
        df_train.drop(column_name, axis=1, inplace=True)
        logger.debug("dummying column {}".format(column_name))
        logger.debug("  values: {}".format(le.classes_))

        for ii in range(len(new_column[0])):
            this_column_name = column_name+" "+str(le.classes_[ii])
            df_train[this_column_name] = new_column[:,ii]


    df_y = df_train[["id", "status_group"]]
    data_y = encode_labels(df_y)["status_group"].tolist()
    df_X = df_train.drop("status_group", axis=1)
    data_X = df_X.as_matrix()   
 
    kbest = SelectKBest(k=100)
    rf = RandomForestClassifier()
    steps = [('feature_selection', kbest),
             ('random_forest', rf)]

    pipeline = Pipeline(steps)
    
    logger.info("attempting to fit model")
    pipeline.fit(data_X, data_y)

    return pipeline


def run_predictions(model, test_df):
    """
    Run predictions for the model on the test_df

    Params
    ------
    test_df, pd.DataFrame
        testing dataset

    Returns
    -------
    list of int
        class label predictions for test_df
    """
    # fill in stuff
    return predictions



def main():
    file_name = "../../datasets/dataset_1.csv"
    train_df, test_df = handle_data_files(file_name)
    
    model = build_model(train_df)
    # comment this out for now because it'll crash
    # predictions = run_predictions(model, test_df)
    logger.info("All done!")


if __name__=="__main__":
    main()
