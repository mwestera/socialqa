from sentence_transformers import SentenceTransformer
from transformers import pipeline

import logging
import sys
import click
import json

import csv

"""
Compute embeddings for sentences.jsonl, writing them to a separate .csv file based on sentence id.

Example:

$ cat collected/sentences_conspiracy.jsonl | python embed_sentences.py > collected/sentences_conspiracy_embs.csv
"""


@click.command(help="Compute embeddings for sentences jsonl, outputting them in .csv format.")
@click.argument("sentences", type=click.File('r'), default=sys.stdin)
def main(sentences):

    logging.basicConfig(level=logging.INFO)

    # 384 dimensional; there seems to be no lower-dim model: https://www.sbert.net/docs/pretrained_models.html
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    writer = csv.writer(sys.stdout)

    n = 0

    for line in sentences:
        sentence = json.loads(line)
        id, text = sentence['id'], sentence['text']
        emb = model.encode(text, show_progress_bar=False)
        writer.writerow([id, *emb])
        n += 1

    logging.info(f'Computed {n} embeddings.')


if __name__ == '__main__':
    main()