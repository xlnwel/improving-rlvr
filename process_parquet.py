import pandas as pd


def read_parquet(data_path):

    df = pd.read_parquet(data_path)

    return df


if __name__ == "__main__":
    data_path = "data/math/train.parquet"
    df = read_parquet(data_path)

    print(df.head())
    print(len(df))

    data_path = "data/math/test.parquet"
    df = read_parquet(data_path)

    print(df.head())
    print(len(df))


    data_path = "data/aime2024/test.parquet"
    df = read_parquet(data_path)
    print(df)
    print(len(df))