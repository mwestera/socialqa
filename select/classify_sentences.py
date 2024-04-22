import logging

from transformers import pipeline
import sys
import click
import json

import csv
import re

from importlib.resources import open_text


"""
Add subjectivity and concreteness scores to sentences.jsonl.

Subjectivity based on a transformer model. Concreteness simply by averaging wors scores from a lookup table (Brysbaert et al.) 

Example:

$ cat collected/sentences_conspiracy.jsonl | python classify_sentences.py > collected/sentences_conspiracy_scores.jsonl
"""


@click.command(help="Add information to sentences.jsonl, namely subjectivity and abstractness.")
@click.argument("sentences", type=click.File('r'), default=sys.stdin)
def main(sentences):

    logging.basicConfig(level=logging.INFO)

    classify_subjectivity = make_subjectivity_classifier()
    classify_concreteness = make_concreteness_classifier()

    try:
        for n, line in enumerate(sentences):
            sentence = json.loads(line)
            sentence['subjectivity'] = classify_subjectivity(sentence['text'])
            sentence['concreteness'] = classify_concreteness(sentence['text'])
            print(json.dumps(sentence))
    except KeyboardInterrupt:
        logging.info('Keyboard interrupt!')

    logging.info(f'Computed scores for {n} sentences.')


def make_subjectivity_classifier():
    model = pipeline(
        task="text-classification",
        model="cffl/bert-base-styleclassification-subjective-neutral",
        top_k=None,
    )

    def classify_subjectivity(text):
        return model(text)[0][0]['score']

    return classify_subjectivity


def make_concreteness_classifier():
    concreteness_ratings = {}
    with open_text('auxiliary', 'Concreteness_ratings_Brysbaert_et_al_BRM.tsv') as file:
        reader = csv.reader(file, delimiter="\t")
        next(reader)    # skip columns
        for row in reader:
            concreteness_ratings[row[0]] = float(row[2])

    word_re = re.compile(r'\b\w+\b')

    def classify_concreteness(text):
        word_ratings = [concreteness_ratings.get(w, None) for w in word_re.findall(text.lower())]
        if not word_ratings:
            return None
        return sum(filter(lambda x: x is not None, word_ratings)) / len(word_ratings)

    return classify_concreteness


if __name__ == '__main__':
    main()