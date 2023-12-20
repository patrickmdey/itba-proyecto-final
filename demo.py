import sys
from time import sleep
import pandas as pd
import os
from termcolor import colored
from loguru import logger
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
    AutoModelForCausalLM,
)

SLEEP_TIME = 4

NUM_LABELS = {
    "LABEL_0": 1,
    "LABEL_1": 2,
    "LABEL_2": 3,
    "LABEL_3": 4,
    "LABEL_4": 5,
}


def generte_review(generator_model, rating, tokenizer, iter, tot):
    logger.info(f"Generando | Reseña de puntaje {rating} ({iter}/{tot})")

    input_text = f"Review of {rating}."
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    generator_model.eval()
    outputs = generator_model.generate(input_ids, do_sample=True, max_length=100).to(
        device
    )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def classify_review(classification_pipeline, review):
    label = classification_pipeline(review)[0]["label"]
    return NUM_LABELS[label]


def reset_files():
    if not os.path.exists("out"):
        os.mkdir("out")

    with open("out/correct_reviews.csv", "w") as correct_file, open(
        "out/wrong_reviews.csv", "w"
    ) as wrong_file:
        correct_file.write("real;pred;review\n")
        wrong_file.write("real;pred;review\n")
        correct_file.close()
        wrong_file.close()


def sleep_for_read():
    print("Momento de leer ", end="")
    sleep(1)
    # print one dot every second horizontally
    for i in range(1, SLEEP_TIME):
        print(".", end="", flush=True)
        sleep(1)
    print()


def print_with_enter(msg, color):
    print()
    print(colored(msg, color))
    print()


if __name__ == "__main__":
    logger.info("DEMO: Generate review and classify with it")
    gpt_checkpoint = "distilgpt2"
    gpt_model_path = "models/best_gpt2"

    bert_model_path = "models/best_bert"
    bert_tokenizer = "distilbert-base-uncased"

    device = 0

    logger.info("Cargando | Classifier tokenizer")
    classifier_tokenizer = AutoTokenizer.from_pretrained(bert_tokenizer)
    classifier_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    logger.info("Cargando | Classifier model")
    classifier_model = AutoModelForSequenceClassification.from_pretrained(
        bert_model_path, num_labels=5, pad_token_id=classifier_tokenizer.pad_token_id
    ).to(0)

    classification_pipeline = TextClassificationPipeline(
        model=classifier_model, tokenizer=classifier_tokenizer, device=device
    )

    logger.info("Cargando | GPT-2 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(gpt_checkpoint)

    logger.info("Cargando | GPT-2 model")
    current_model = AutoModelForCausalLM.from_pretrained(
        gpt_model_path, pad_token_id=tokenizer.eos_token_id
    ).to(device)

    #### DEMO ####

    # reset files before begining
    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        reset_files()

    rating = input("Ingresa un puntaje del 1 al 5: ")

    while not rating.isdigit() or int(rating) not in range(1, 6):
        rating = input("El puntaje debe ser un entero entre 1 y 5: ")

    with open(f"out/correct_reviews_of_{rating}.csv", "a") as correct_file, open(
        f"out/wrong_reviews_of_{rating}.csv", "a"
    ) as wrong_file:
        try_counter = 0

        rating = int(rating)

        iters = 0
        wrongly_predicted = 0
        correctly_predicted = 0

        while iters < 10:
            review = generte_review(current_model, rating, tokenizer, iters + 1, 10)
            review = review[0].replace(f"Review of {rating}. ", "")

            # print_with_enter(review, "yellow")

            # sleep_for_read()

            predicted = classify_review(classification_pipeline, review)
            if predicted != rating:
                wrongly_predicted += 1
                print_with_enter(
                    f"Predicción Erronea | Real: {rating} | Predicha: {predicted}",
                    "red",
                )

                wrong_file.write(f"{rating};{predicted};{review}\n")
            else:
                correctly_predicted += 1
                print_with_enter(
                    f"Predicción correcta | Real: {rating} | Predicha: {predicted}",
                    "green",
                )

                correct_file.write(f"{rating};{predicted};{review}\n")

            iters += 1
            # os.system("cls" if os.name == "nt" else "clear")

        print(f"Correctly predicted: {correctly_predicted}")
        print(f"Wrongly predicted: {wrongly_predicted}")

        wrong_file.close()
        correct_file.close()
