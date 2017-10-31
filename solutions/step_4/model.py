from pathlib import Path
import logging
import pandas as pd

from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def main():
    file_name = "../../datasets/dataset_1.csv"
    train_df, test_df = handle_data_files(file_name)
    logger.info("All done!")


if __name__=="__main__":
    main()
