import csv
import os
import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModelForQuestionAnswering
import torch
import click
import sys
@click.command(help="")
@click.argument("infile",type=str, default=sys.stdin)
@click.option("--qa", help="Max how many QA items per user.", type=bool, required=False, is_flag=False, default=None)
def main(infile, qa):
    # Assuming the setup for tokenizer and model is done as shown previously
    qa=True
    rte= 1-qa
    if(qa==True):
        model_name = "ahotrod/albert_xxlargev1_squad2_512" 
        model= AutoModelForQuestionAnswering.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif(rte==True):
        tokenizer = AutoTokenizer.from_pretrained("ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli")
        model = AutoModelForSequenceClassification.from_pretrained("ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli")

    #infile = 'pipeline/temp12345_qa.tsv'

    # Process and write scores
    process_and_write_scores(infile, tokenizer, model, rte, qa)

def process_and_write_scores(infile, tokenizer, model, rte, qa, chunk_size=100):
    """
    Process the file in chunks and write scores incrementally to avoid memory issues.
    """
    root, file_extension = os.path.splitext(infile)
    outfile = f"{root}_scores{file_extension}"
    
    with open(infile, 'r', encoding='utf-8') as read_file, \
         open(outfile, 'w', newline='', encoding='utf-8') as write_file:
        
        tsv_reader = csv.DictReader(read_file, delimiter='\t')
        fieldnames = tsv_reader.fieldnames + ['score']  # Extend fieldnames to include 'score'
        
        writer = csv.DictWriter(write_file, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        
        for row in tqdm.tqdm(tsv_reader):
            # Assuming the model_predict function returns a score for a sentence pair
            if(qa):
                score=model_predict_qa(row['sentence1'], row['sentence2'], tokenizer, model)
            elif(rte):
                score = model_predict_rte(row['sentence1'], row['sentence2'], tokenizer, model)
            row['score'] = score
            writer.writerow(row)

def model_predict_rte(sentence1, sentence2, tokenizer, model):
    """
    A mock function for making predictions. Replace with actual prediction logic.
    """
    inputs = tokenizer(sentence1, sentence2, return_tensors="pt", padding=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=1)
    # Assuming we're interested in the first label probability
    score = predictions[0, 0].item()
    return score
def find_best_token_index(predictions, input_ids, tokenizer, skip_tokens):
    # Sort the predictions in descending order of probability and get the indices
    sorted_indices = torch.argsort(predictions, descending=True)
    for idx in sorted_indices[0]:  # Iterate over indices of sorted predictions
        # Ensure the idx is used to index input_ids correctly as a list
        token_ids = input_ids[idx].unsqueeze(0)  # Add a dimension to make it iterable
        token = tokenizer.convert_ids_to_tokens(token_ids)[0]
        if token not in skip_tokens:
            return idx.item(), predictions[0, idx].item()
    return None, None  # In case all tokens are skip tokens, which should not happen


def model_predict_qa(sentence1, sentence2, tokenizer, model):
    inputs = tokenizer(sentence1, sentence2, return_tensors="pt", padding=True,  max_length=512)
    outputs = model(**inputs)

    predictions_start = torch.softmax(outputs.start_logits, dim=1)
    predictions_end = torch.softmax(outputs.end_logits, dim=1)

    # Define skip tokens
    skip_tokens = ["[CLS]", "[SEP]"]

    # Find the best start and end indices, skipping over the special tokens
    answer_start, prob_start = find_best_token_index(predictions_start, inputs['input_ids'][0], tokenizer, skip_tokens)
    answer_end, prob_end = find_best_token_index(predictions_end, inputs['input_ids'][0], tokenizer, skip_tokens)
    # Ensure we have valid start and end
    if answer_start is not None and answer_end is not None and answer_start <= answer_end:
        # Convert tokens to answer string
        average_prob = (prob_start + prob_end) / 2
        return average_prob
    else:
        # There is no answer
        return 0

if __name__=="__main__":
    main()
