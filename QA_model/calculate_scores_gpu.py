import csv
import os
import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModelForQuestionAnswering
import torch
import click
import sys
import torch
import time

"""
Calculate scores for the input file using a QA or Entailment model.

Example:
$ python calculate_scores_gpu.py --model_type qa infile.tsv
"""

@click.command(help="")
@click.argument("infile",type=str, default=sys.stdin)
@click.argument("model_type", type=str)
def main(infile, model_type):
    """
    Main function for calculating scores.

    Args:
        infile (str): Path to the input file.
        qa (bool): Flag indicating whether the input file is for question answering.
        model_type (str): Type of model to use for prediction.

    Returns:
        None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load the model and tokenizer based on the model type
    if model_type == "qa":
        model_name = "ahotrod/albert_xxlargev1_squad2_512" 
        model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif model_type == "rte":
        model_name = "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Process and write scores
    process_and_write_scores(infile, tokenizer, model, model_type)
    return

def process_and_write_scores(infile, tokenizer, model, model_type, batch_size=100):
    """
    Process the file in chunks and write scores incrementally to avoid memory issues.
    """
    root, file_extension = os.path.splitext(infile)

    outfile = f"{root}_scores{file_extension}"
    
    # Open the input and output files
    with open(infile, 'r', encoding='utf-8') as read_file, \
         open(outfile, 'w', newline='', encoding='utf-8') as write_file:
        
        # Read file
        tsv_reader = csv.DictReader(read_file, delimiter='\t')
        fieldnames = tsv_reader.fieldnames + ['score'] 
        
        # Write file
        writer = csv.DictWriter(write_file, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        
        batch_rows=[]
        batch_texts=[]
        batch_qas=[]

        for row in tqdm.tqdm(tsv_reader):
            # Check if the index is a string and convert it to an integer else it will not be processed
            if(type(row["index"])==str):
                try:
                    row["index"]=int(row["index"])
                except:
                    continue
            # If the model type is QA
            if model_type=="qa":
                qa_list = eval(row['pivot_positions'])  # Convert the string representation of the list to an actual list
                if(qa_list[0]>1000):
                    new_post, new_qa_list = reduce_context_size(400, qa_list, row)
                    question = row['question']
                    post = new_post
                    pivot = row['pivot']
                    batch_texts.append((question, post, pivot))
                    batch_qas.append(new_qa_list)

                else:
                    question = row['question']
                    pivot = row['pivot']
                    post = row['post']
                    batch_texts.append((question, post, pivot))
                    batch_qas.append(qa_list)

            # If the model type is Entailment
            elif model_type  == 'rte':
                sentence1 = row['sentence1']
                sentence2 = row['sentence2']
                batch_texts.append((sentence1, sentence2))

            batch_rows.append(row)

            # Predict after batch size is full
            if len(batch_rows) >= batch_size:
                if model_type=='qa':
                    scores = model_predict_qa([texts[0] for texts in batch_texts], [texts[1] for texts in batch_texts], [texts[2] for texts in batch_texts], batch_qas, tokenizer, model)
                else:
                    scores = model_predict_rte([texts[0] for texts in batch_texts], [texts[1] for texts in batch_texts], tokenizer, model)
                for row, score in zip(batch_rows, scores):
                    row['score'] = score
                    writer.writerow(row)
                batch_rows, batch_texts, batch_qas= [], [], []  # Reset for next batch

        # Last iteration can include samples size < batch_size
        if batch_rows:
            if model_type =='qa':
                scores = model_predict_qa([texts[0] for texts in batch_texts], [texts[1] for texts in batch_texts], [texts[2] for texts in batch_texts], batch_qas, tokenizer, model)
            else:
                scores = model_predict_rte([texts[0] for texts in batch_texts], [texts[1] for texts in batch_texts], tokenizer, model)
            for row, score in zip(batch_rows, scores):
                row['score'] = score
                writer.writerow(row)


def reduce_context_size(context_start, qa_list, row):
    """
    Reduces the context size of the post and updates the QA list accordingly.
    """
    post = row['post']

    post= post[qa_list[0][0]-context_start:]
    qa_list[0] = (context_start, context_start+ (qa_list[0][1]-qa_list[0][0]))

    return post, qa_list


def find_sublist_in_list(big_list, sublist):
    """
    Find the index of a sublist in a list.
    """
    sublist_length = len(sublist)
    for i in range(len(big_list)):
        if big_list[i:i+sublist_length] == sublist:
            return i
    return -1

def find_pivot(tokenizer, pivot, input_ids):
    """
    Find the pivot tokens in the post tokens.
    """
    # Encode the pivot
    encoded_pivot = tokenizer.encode(pivot)[1:-1]  # Ignore the first and last token
    # Convert the tokenized input to a list
    input_tokens = input_ids.tolist()
    # Check if the encoded pivot is in the tokenized input
    pivot_index = find_sublist_in_list(input_tokens, encoded_pivot)

    start_token_pos = pivot_index

    end_token_pos = start_token_pos + len(encoded_pivot) -1

    return start_token_pos, end_token_pos

def model_predict_rte(sentence1s, sentence2s, tokenizer, model):
    """
    Make predictions on batches of sentence pairs using an entailment model.
    
    Args:
        sentence1s (list of str): The first sentences in the pairs.
        sentence2s (list of str): The second sentences in the pairs.
        tokenizer: The tokenizer used to encode the inputs.
        model: The entailment model used for prediction.
    
    Returns:
        list of float: The list of probabilities indicating entailment.
    """
    # Ensure the input is in batch form (list of sentences)
    if isinstance(sentence1s, str):
        sentence1s = [sentence1s]
    if isinstance(sentence2s, str):
        sentence2s = [sentence2s]

    # Tokenize the batch of sentence pairs
    inputs = tokenizer(sentence1s, sentence2s, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Move to the appropriate device (e.g., GPU if available)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Perform prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)[:, 0]  # Get probabilities for the 'entailment' class = 0

    # Convert tensor to list of probabilities
    return predictions.tolist()


def model_predict_qa(questions, posts, pivots, qa_lists, tokenizer, model):
    """
    Predicts the probabilities of answers for given lists of questions and contexts using a QA model.
    
    Args:
        questions (list of str): List of input questions.
        contexts (list of str): List of input contexts.
        qa_lists (list of list of tuples): List of lists of tuples representing the character positions of the answers in the context.
        tokenizer: The tokenizer used to encode the inputs.
        model: The QA model used for prediction.

    Returns:
        list of list: A list of lists containing probabilities corresponding to each answer in the qa_lists.
    """

    # Tokenize batch of questions and contexts
    inputs = tokenizer(questions, posts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Move to device

    # Perform prediction
    with torch.no_grad():
        start_time = time.time()
        # Make prediction
        outputs = model(**inputs)
        end_time = time.time()

        print(f"Time taken for batch model prediction: {end_time - start_time} seconds")

        # Softmax converts to probabilities
        predictions_start = torch.softmax(outputs.start_logits, dim=1)
        predictions_end = torch.softmax(outputs.end_logits, dim=1)

    probabilities = []
    # Process each question-context pair in the batch
    for idx, input_ids in enumerate(inputs['input_ids']):

        # Call find_pivot with the current inputs
        start_token, end_token = find_pivot(tokenizer, pivots[idx], input_ids)

        # Pivot not found in post = Error
        if start_token == -1:
            probabilities.append(0)
            continue

        max_average_prob = 0

        # Calculate the best answer combination between start and end token
        for i in range(start_token -1, end_token + 1):
            for j in range(i, end_token + 2):

                average_prob = (predictions_start[idx, i] + predictions_end[idx, j]) / 2 
                if average_prob > max_average_prob:
                    max_average_prob = average_prob

        probabilities.append(max_average_prob.item())

    return probabilities

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

