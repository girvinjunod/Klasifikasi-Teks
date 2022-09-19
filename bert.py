import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import logging
import pickle
import numpy as np
import pandas as pd

from tensorflow.keras.layers import Input, Dropout, Dense, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from utils.constants import TRAIN_PATH, TEST_PATH, MAPS

from transformers import (
    BertConfig,
    BertTokenizerFast,
    TFAutoModel,
)


def preprocess(train):
    no, yes = train["label"].value_counts()

    diff = abs(no - yes)

    if no > yes:
        no_indices = train.index[train["label"] == "no"]
        indices = np.random.choice(no_indices, diff, replace=False)
    elif yes > no:
        yes_indices = train.index[train["label"] == "yes"]
        indices = np.random.choice(yes_indices, diff, replace=False)

    return train.drop(indices)


def getTokenizer():
    return BertTokenizerFast.from_pretrained(
        pretrained_model_name_or_path="bert-base-uncased",
        config=BertConfig.from_pretrained("bert-base-uncased"),
    )


def tokenize(tokenizer, x) -> dict:
    res = tokenizer(
        text=list(x),
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="tf",
    )

    return {
        "input_ids": res["input_ids"],
        "token_type_ids": res["token_type_ids"],
        "attention_mask": res["attention_mask"],
    }


def generateModel():
    bert = TFAutoModel.from_pretrained("bert-base-uncased")

    bert_layer = bert.bert
    input_ids = Input(shape=(512,), name="input_ids", dtype="int32")
    token_type_ids = Input(shape=(512,), name="token_type_ids", dtype="int32")
    attention_mask = Input(shape=(512,), name="attention_mask", dtype="int32")
    inputs = {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
    }

    x = bert_layer(inputs)[0]
    x = LSTM(128)(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=x)
    model.layers[3].trainable = False
    return model


def main():
    logging.basicConfig(format="[INFO] %(message)s", level=logging.INFO)
    logging.info("Loading Data")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    logging.info("Preprocessing Data")
    train = preprocess(train)

    tokenizer = getTokenizer()

    logging.info("Tokenizing Input")
    x_train = tokenize(tokenizer, train["text_a"])
    y_train = train["label"].replace(MAPS)

    x_test = tokenize(tokenizer, test["text_a"])
    y_test = test["label"].replace(MAPS)

    logging.info("Loading model")
    model = generateModel()
    logging.info("Model Summary")
    model.summary()

    optimizer = Adam(learning_rate=5e-5)

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    logging.info("Training Model")
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=8,
        epochs=5,
        verbose=True,
        validation_data=(x_test, y_test),
    )

    logging.info("Saving Model")
    model.save_weights("bert.h5", model)
    logging.info("Model saved")


if __name__ == "__main__":
    main()
