from sentence_transformers import SentenceTransformer

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


@click.command(help="Add information to sentences.jsonl, namely subjectivity and abstractness.")
@click.argument("sentences", type=click.File('r'), default=sys.stdin)
def main(sentences):

    logging.basicConfig(level=logging.INFO)

    # 384 dimensional; there seems to be no lower-dim model: https://www.sbert.net/docs/pretrained_models.html
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    ids = []
    texts = []

    for n, line in enumerate(sentences):
        sentence = json.loads(line)
        id, text = sentence['id'], sentence['text']
        texts.append(text)
        ids.append(id)

    writer = csv.writer(sys.stdout)

    for id, emb in zip(ids, model.encode(texts, show_progress_bar=True)):
        writer.writerow([id, *emb])



if __name__ == '__main__':
    main()