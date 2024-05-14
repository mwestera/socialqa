import click
import jsonlines
import logging
import tempfile
import shutil
import torch
from transformers import pipeline
from tqdm import tqdm
import csv
import re
import sys
import os
@click.command(help="Add subjectivity and concreteness scores to sentences in a JSONL file.")
@click.argument("sentences", type=click.File('r'), default=sys.stdin)
def main(sentences):
    logging.basicConfig(level=logging.INFO)
    batch_size = 16

    classify_subjectivity = make_subjectivity_classifier(batch_size)
    classify_concreteness = make_concreteness_classifier()

    sentences_buffer = []
    ids = []
    try:
        with tempfile.NamedTemporaryFile(delete=False, mode='w', newline='', encoding='utf-8') as temp_file:
            writer = jsonlines.Writer(temp_file)
            reader = jsonlines.Reader(sentences)
            for sentence in tqdm(reader, desc="Processing sentences"):
                sentences_buffer.append(sentence['text'])
                ids.append(sentence)
                if len(sentences_buffer) >= batch_size:
                    subjectivity_scores = classify_subjectivity(sentences_buffer)
                    concreteness_scores = classify_concreteness(sentences_buffer)
                    for sentence, subj_score, conc_score in zip(ids, subjectivity_scores, concreteness_scores):
                        sentence['subjectivity'] = subj_score
                        sentence['concreteness'] = conc_score
                        writer.write(sentence)
                    sentences_buffer = []
                    ids = []
            # Process remaining sentences if any
            if sentences_buffer:
                subjectivity_scores = classify_subjectivity(sentences_buffer)
                concreteness_scores = classify_concreteness(sentences_buffer)
                for sentence, subj_score, conc_score in zip(ids, subjectivity_scores, concreteness_scores):
                    sentence['subjectivity'] = subj_score
                    sentence['concreteness'] = conc_score
                    writer.write(sentence)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        shutil.move(temp_file.name, sentences.name if sentences != sys.stdin else 'updated_sentences.jsonl')
        logging.info(f"Updated file and added new data at {sentences.name if sentences != sys.stdin else 'updated_sentences.jsonl'}")

def make_subjectivity_classifier(batch_size=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = pipeline("text-classification", model="cffl/bert-base-styleclassification-subjective-neutral", device=device)
    def classify_subjectivity(texts):
        if isinstance(texts, str):
            texts = [texts]
        return [result['score'] for result in model(texts)]
    return classify_subjectivity

def make_concreteness_classifier():
    concreteness_ratings = {}
    file_path = os.path.join(os.path.dirname(__file__), 'auxiliary', 'Concreteness_ratings_Brysbaert_et_al_BRM.tsv')
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter="\t")
        next(reader)  # Skip header
        for row in reader:
            concreteness_ratings[row[0].lower()] = float(row[2])
    word_re = re.compile(r'\b\w+\b')
    def classify_concreteness(texts):
        scores = []
        for text in texts:
            word_ratings = [concreteness_ratings.get(word.lower(), 0) for word in word_re.findall(text)]
            rated = [rating for rating in word_ratings if rating > 0]
            scores.append(sum(rated) / len(rated) if rated else 0)
        return scores
    return classify_concreteness


if __name__ == '__main__':
    main()