import math
import json
import re
from pynvml import *
import pandas as pd
from loguru import logger
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


def reduce_columns(df, rating_col, review_col, reduce_cols=True):
    """
    Reduces the columns of the dataframe to only the rating and review column.
    """
    logger.info("Reducing columns to only rating and review")
    if reduce_cols:
        df = df[[rating_col, review_col]]
    df = df.rename(columns={rating_col: "rating", review_col: "review"})
    return df


def reduce_ratings(df, col="Rating"):
    df[col] = df[col].apply(lambda x: math.ceil(int(x)/2))
    return df


def remove_words(df, words):
    df["Review"] = df.apply(lambda row: " ".join(
        [word for word in str(row["Review"]).split() if word not in words]), axis=1)

    df["Review"] = df["Review"].apply(
        lambda row: re.sub(r'([^a-zA-Z\s])\1+', r'\1', row))


def preprocess_df(df):
    df["Review"] = df.apply(lambda row: row["Review"].replace(
        "\n", "\\n").replace("/><br", ""), axis=1)

    # df["Sentiment"] = df["Sentiment"].apply(
    #     lambda val: 1 if val == "positive" else 0)


def tag_reviews(df):
    df['review'] = df.apply(
        lambda row: f"Review of {row['rating']}. {row['review']}", axis=1)


def alternative_tag_reviews(df, rating_dict):
    df["Review"] = df.apply(lambda row: rating_dict[math.ceil(
        row["Rating"]/2)] + ". " + row["Review"], axis=1)
    reduce_ratings(df)


def get_bert_dataset(df, text_col, label_col, test_size=0.2):
    """
        Returns a DatasetDict with the following structure:
        {
            "train": Dataset({
                features: ['text', 'label']
                num_rows: int
            }),
            "test": Dataset({
                features: ['text', 'label']
                num_rows: int
            })
        }
        Args:
            data_path (str): Path to the dataset
    """
    # shuffle rows
    df = df.rename(columns={text_col: "text", label_col: "label"})
    df['label'] = df['label'] - 1
    dataset = Dataset.from_pandas(df)

    # Ensure that the labels are in the correct format (integers)
    dataset = dataset.map(lambda example: {'label': int(example['label'])})

    return dataset.train_test_split(test_size=test_size, shuffle=True)


def get_gpt2_dataset(df, text_col="review", label_col="rating", test_size=0.2):
    """
        Returns a DatasetDict with the following structure:
        {
            "train": Dataset({
                features: ['rating', 'review']
                num_rows: int
            }),
            "test": Dataset({
                features: ['rating', 'review']
                num_rows: int
            })
        }
        Args:
            data_path (str): Path to the dataset
    """
    # dataset_size = sample_size / 5 if sample_size is not None else 0
    # new_df = pd.DataFrame(columns=[text_col, label_col])
    # for i in range(1, 6):
    #     rating_samples = df[df[label_col] == i].sample(
    #         dataset_size) if sample_size is not None else df[df[label_col] == i]
    #     new_df = pd.concat([new_df, rating_samples], ignore_index=True)

    tag_reviews(df)
    train_df, test_df = train_test_split(df, test_size=test_size, shuffle=True)

    return DatasetDict({
        "train": Dataset.from_pandas(train_df).remove_columns(["__index_level_0__"]),
        "test": Dataset.from_pandas(test_df).remove_columns(["__index_level_0__"])
    })


def get_review_dataset(data_path, for_model="gpt", separator=";", label_col="rating", text_col="review", test_size=0.2):

    df = pd.read_csv(data_path, sep=separator, on_bad_lines="skip")

    df = df[df["review"].notna()]

    """
    dataset_size = int(sample_size / 5) if sample_size is not None else 0
    new_df = pd.DataFrame(columns=[text_col, label_col])
    for i in range(1, 6):
        rating_samples = df[df[label_col] == i].sample(
            dataset_size) if sample_size is not None else df[df[label_col] == i]
        new_df = pd.concat([new_df, rating_samples], ignore_index=True)

    df = new_df
    """

    if for_model == "gpt":
        return get_gpt2_dataset(df, text_col, label_col, test_size)
    elif for_model == "bert":
        return get_bert_dataset(df, text_col, label_col, test_size)


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def get_args(base_folder, config_file, trainer_args_file):

    if base_folder.split("/")[-1] == "gpt2":
        default_base_checkpoint = "distilgpt2"
    elif base_folder.split("/")[-1] == "bert":
        default_base_checkpoint = "bert-base-uncased"

    config_args = json.load(open(config_file))
    config_args["dataset_path"] = config_args["dataset"] if "dataset" in config_args else "datasets/joined_movies/preprocessed_no_validation.csv"
    config_args["token_checkpoint"] = config_args["token_checkpoint"] if "token_checkpoint" in config_args else default_base_checkpoint
    config_args["model_checkpoint"] = config_args["model_checkpoint"] if "model_checkpoint" in config_args else default_base_checkpoint
    config_args["layers_to_freeze"] = config_args["layers_to_freeze"] if "layers_to_freeze" in config_args else 5
    config_args["folder_save"] = config_args["folder_save"] if "folder_save" in config_args else "preprocessed"
    config_args["max_tokenizer_length"] = config_args["max_length"] if "max_length" in config_args else 200
    config_args["use_callback"] = config_args["use_callback"] if "use_callback" in config_args else False
    config_args["dataset_size"] = config_args["dataset_size"] if "dataset_size" in config_args else 50_000
    config_args["parameter"] = config_args["parameter"] if "parameter" in config_args else "epochs"
    config_args["test_size"] = config_args["test_size"] if "test_size" in config_args else 0.2

    training_args = json.load(open(trainer_args_file))
    training_args["output_dir"] = f"{base_folder}/checkpoints/{config_args['folder_save']}"
    model_dir = f"{base_folder}/models/{config_args['folder_save']}"
    return config_args, training_args, model_dir
