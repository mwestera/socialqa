from sentence_transformers import SentenceTransformer
import logging
import sys
import click
import json
import os
import csv
import json
import logging
import sys
import click
import json
import os
import csv
import jsonlines
import tqdm
import torch
from sentence_transformers import SentenceTransformer

@click.command(help="Add information to sentences.jsonl, namely subjectivity and abstractness.")
@click.argument("sentences", type=click.Path(), required=True, default='None')
@click.argument("post_file_name",type=str, default=sys.stdin)
def main(sentences, post_file_name):
    logging.basicConfig(level=logging.INFO)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Initialize the SentenceTransformer model on the specified device
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)
    ids = []
    texts = []
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
        embeddings_file = config['dir'] + f"/{post_file_name}_embeddings.csv"
    # Remove the file extension and add the new suffix


    # Read and parse each line of sentences
    with jsonlines.open(sentences) as reader:
        for n, sentence in enumerate(tqdm.tqdm(reader)): 
            id, text = sentence['id'], sentence['text']
            texts.append(text)
            id.append(ids)

    # Open a file to write the embeddings
    with open(embeddings_file, 'w', newline='') as file:
        writer = csv.writer(file)

        # Encode texts to embeddings and write each as a row in the file
        for idx, (id, emb) in enumerate(zip(ids, model.encode(texts, show_progress_bar=True))):
            writer.writerow([id, *emb])
if __name__ == '__main__':
    main()
