import csv
import os
import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
import torch
import click
import sys

@click.command(help="")
@click.argument("infile", type=str, default=sys.stdin)
@click.option("--qa", help="Do we use QA model or RTE , True=QA, False=RTE", type=bool, required=False, is_flag=False, default=None)
def main(infile, qa):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Check if CUDA is available

    if qa:
        model_name = "ahotrod/albert_xxlargev1_squad2_512"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    else:  
        model_name = "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

    model.to(device)  # Move model to GPU if available

    process_and_write_scores(infile, tokenizer, model, qa, device)

def process_and_write_scores(infile, tokenizer, model, qa, device, batch_size=32):
    root, file_extension = os.path.splitext(infile)
    outfile = f"{root}_scores{file_extension}"

    with open(infile, 'r', encoding='utf-8') as read_file, \
         open(outfile, 'w', newline='', encoding='utf-8') as write_file:
        
        tsv_reader = csv.DictReader(read_file, delimiter='\t')
        fieldnames = tsv_reader.fieldnames + ['score']  # Extend fieldnames to include 'score'
        writer = csv.DictWriter(write_file, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()

        batch = []
        for row in tqdm.tqdm(tsv_reader):
            batch.append(row)
            if len(batch) >= batch_size:
                process_batch(batch, tokenizer, model, writer, qa, device)
                batch = []

        if batch:
            process_batch(batch, tokenizer, model, writer, qa, device)

def process_batch(batch, tokenizer, model, writer, qa, device):
    sentence1 = [b['sentence1'] for b in batch]
    sentence2 = [b['sentence2'] for b in batch]

    inputs = tokenizer(sentence1, sentence2, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Move input tensors to GPU

    with torch.no_grad():
        outputs = model(**inputs)
        if qa:
            predictions = torch.softmax(outputs.start_logits + outputs.end_logits, dim=1)[:, 1]  # Example logic for QA
        else:
            predictions = torch.softmax(outputs.logits, dim=1)[:, 0]  # Example logic for RTE

    for i, row in enumerate(batch):
        row['score'] = predictions[i].item()
        writer.writerow(row)

if __name__ == "__main__":
    main()

