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
        fieldnames = tsv_reader.fieldnames + ['score'] 
        
        writer = csv.DictWriter(write_file, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        
        for row in tqdm.tqdm(tsv_reader):
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

    score = predictions[0, 0].item()
    return score
def find_best_token_index(predictions, input_ids, tokenizer, skip_tokens):

    sorted_indices = torch.argsort(predictions, descending=True)
    for idx in sorted_indices[0]:
        token_ids = input_ids[idx].unsqueeze(0) 
        token = tokenizer.convert_ids_to_tokens(token_ids)[0]
        if token not in skip_tokens:
            return idx.item(), predictions[0, idx].item()
    return None, None 


def model_predict_qa(sentence1, sentence2, tokenizer, model):
    #sentence1=sentence1.lower()
    #sentence2=sentence2.lower()
    inputs = tokenizer(sentence1, sentence2, return_tensors="pt", padding=True,  max_length=512)
    outputs = model(**inputs)

    predictions_start = torch.softmax(outputs.start_logits, dim=1)
    predictions_end = torch.softmax(outputs.end_logits, dim=1)

    # Define skip tokens
    skip_tokens = ["[CLS]", "[SEP]"]
    answer_start, prob_start = find_best_token_index(predictions_start, inputs['input_ids'][0], tokenizer, skip_tokens)
    answer_end, prob_end = find_best_token_index(predictions_end, inputs['input_ids'][0], tokenizer, skip_tokens)
    # Ensure we have valid start and end
    if answer_start is not None and answer_end is not None and answer_start <= answer_end:
        average_prob = (prob_start + prob_end) / 2
        return average_prob
    else:
        return 0

if __name__=="__main__":
    main()
