import argparse
import logging
import os
import pickle
import re

import pandas as pd
from emoji import demojize
from tensorflow.keras.optimizers import Adam
from transformers import BertConfig, BertTokenizerFast, TFBertForSequenceClassification

from utils.constants import MAPS, TEST_PATH, TRAIN_PATH

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def preprocess(x_train):
    preprocess_func = [
        lambda x: x.lower(),
        lambda x: re.sub(r"@[^a-zA-Z0-9_]+", "@USER", x),
        lambda x: re.sub(r"#[^a-zA-Z0-9_]+", "@HASHTAG", x),
        lambda x: re.sub(r"http[sS]\S+", "HTTPURL", x),
        lambda x: re.sub(r"www.\S+", "HTTPURL", x),
        lambda x: re.sub(r"bit.ly\S+", "HTTPURL", x),
        lambda x: re.sub(r"\s+[a-zA-Z]\S+", "", x),
        lambda x: demojize(x),
        lambda x: x.strip(),
    ]

    x_train = [x_train.apply(func) for func in preprocess_func]


def getTokenizer(model_name):
    return BertTokenizerFast.from_pretrained(pretrained_model_name_or_path=model_name)


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


def generateModel(model_name):
    config = BertConfig.from_pretrained(model_name)
    config.num_labels = 2
    model = TFBertForSequenceClassification.from_pretrained(model_name, config=config)

    model.layers[0].trainable = False

    return model


def main(batch_size, epoch, learning_rate, model_name):
    logging.basicConfig(format="[INFO] %(message)s", level=logging.INFO)

    logging.info(f"Learning rate : {learning_rate}")
    logging.info(f"Batch size    : {batch_size}")
    logging.info(f"Epoch         : {epoch}")

    logging.info("Loading Data")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    logging.info("Preprocessing Data")
    preprocess(train["text_a"])

    tokenizer = getTokenizer(model_name)

    logging.info("Tokenizing Input")
    x_train = tokenize(tokenizer, train["text_a"])
    y_train = train["label"].replace(MAPS)

    x_test = tokenize(tokenizer, test["text_a"])
    y_test = test["label"].replace(MAPS)

    logging.info("Loading model")
    model = generateModel(model_name)
    logging.info("Model Summary")
    model.summary()

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    logging.info("Training Model")
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epoch,
        verbose=True,
        validation_data=(x_test, y_test),
    )

    logging.info("Saving Model")
    model.save_weights(
        f"dump/bert_lr-{learning_rate}_bs-{batch_size}_ep-{epoch}.h5", model
    )
    logging.info("Model saved")

    logging.info("Saving Prediction")
    y_pred = model.predict(x_test, batch_size=batch_size, verbose=True)
    pickle.dump(
        y_pred,
        open(f"dump/y_pred_lr-{learning_rate}_bs-{batch_size}_ep-{epoch}.pkl", "wb"),
    )
    logging.info("Prediction Saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", required=False, help="Batch Size")

    parser.add_argument("--epoch", required=False, help="Epoch")

    parser.add_argument("--learning_rate", required=False, help="Learning Rate")

    parser.add_argument("--model", required=False, help="Model")

    args = parser.parse_args()

    main(
        32 if args.batch_size is None else args.batch_size,
        2 if args.epoch is None else args.epoch,
        5e-5 if args.learning_rate is None else args.learning_rate,
        "indobenchmark/indobert-lite-base-p2" if args.model is None else args.model,
    )
