import pandas as pd


def main():
    file_name = "../../datasets/dataset_1.csv"
    df = pd.read_csv(file_name)
    print(df.head())


if __name__=="__main__":
    main()
