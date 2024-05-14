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
$ python calculate_scores_cpu.py --model_type qa infile.tsv

"""

@click.command(help="")
@click.argument("infile",type=str, default=sys.stdin)
@click.option("--model_type", help="Which model to use for prediction", choices=["qa", "ent"], type=str, required=True, is_flag=False, default=None)

def main(infile, qa, model_type):
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
        model = AutoModelForQuestionAnswering.from_pretrained(model_name, device=device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif model_type == "ent":
        model_name = "ynie/albert-xxlarge-v2"
        model = AutoModelForSequenceClassification.from_pretrained(model_name, device=device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Process and write scores
    process_and_write_scores(infile, tokenizer, model, qa=qa)

def process_and_write_scores(infile, tokenizer, model, rte, qa, chunk_size=100):
    """
    Process the file in chunks and write scores incrementally to avoid memory issues.
    """
    root, file_extension = os.path.splitext(infile)

    outfile = f"{root}_scores{file_extension}"
    
    # Open the input and output files
    with open(infile, 'r', encoding='utf-8') as read_file, \
         open(outfile, 'w', newline='', encoding='utf-8') as write_file:
        
        tsv_reader = csv.DictReader(read_file, delimiter='\t')
        fieldnames = tsv_reader.fieldnames + ['score'] 
        
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
            if(qa):
                qa_list = eval(row['pivot_positions'])  # Convert the string representation of the list to an actual list
                if(qa_list[0][0]>1000):
                    new_post, new_qa_list = reduce_context_size(400, qa_list, row)
                    question = row['question']
                    post = new_post
                    batch_texts.append((question, post))
                    batch_qas.append(new_qa_list)

                else:
                    question = row['question']
                    post = row['post']
                    batch_texts.append((question, post))
                    batch_qas.append(qa_list)

            # If the model type is Entailment
            elif(rte):
                sentence1 = row['sentence1']
                sentence2 = row['sentence2']
                batch_texts.append((sentence1, sentence2))

            batch_rows.append(row)

            if len(batch_rows) >= chunk_size:
                if qa:
                    scores = model_predict_qa([texts[0] for texts in batch_texts], [texts[1] for texts in batch_texts], batch_qas, tokenizer, model)
                else:
                    scores = model_predict_rte([texts[0] for texts in batch_texts], [texts[1] for texts in batch_texts], tokenizer, model)
                for row, score in zip(batch_rows, scores):
                    row['score'] = score
                    writer.writerow(row)
                batch_rows, batch_texts = [], []  # Reset for next batch

        if batch_rows:
            if qa:
                scores = model_predict_qa([texts[0] for texts in batch_texts], [texts[1] for texts in batch_texts], batch_qas, tokenizer, model)
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
        predictions = torch.softmax(outputs.logits, dim=1)[:, 1]  # Get probabilities for the 'entailment' class

    # Convert tensor to list of probabilities
    return predictions.tolist()


def find_best_token_index(predictions, input_ids, tokenizer, skip_tokens):

    sorted_indices = torch.argsort(predictions, descending=True)
    for idx in sorted_indices[0]:
        token_ids = input_ids[idx].unsqueeze(0) 
        token = tokenizer.convert_ids_to_tokens(token_ids)[0]
        if token not in skip_tokens:
            return idx.item(), predictions[0, idx].item()
    return None, None 

def find_pivot(tokenizer, post, inputs, start_char, end_char):
    """
    Finds the pivot tokens in the input tokens that correspond to a given substring in the post.

    Args:
        tokenizer (Tokenizer): The tokenizer used to encode the post.
        post (str): The input post.
        inputs (Tensor): The input tokens.
        start_char (int): The starting character index of the substring in the post.
        end_char (int): The ending character index of the substring in the post.

    Returns:
        tuple: A tuple containing the start and end token indices that correspond to the substring in the post.
    """

    encoded_sentence = tokenizer.encode(post[start_char:end_char])
    start_token = None
    end_token = None
    # Iterate through the input tokens
    for i in range(len(inputs.input_ids[0])):
        # Check if the current token matches the start of the encoded sentence
        # Start and end of the encoded sentence could be slightly different when taking sentence separately, middle should be unchanged
        if inputs.input_ids[0][i+2:i+len(encoded_sentence)-1].tolist()== encoded_sentence[2:-1]:
            start_token = i+1
            end_token = i + len(encoded_sentence) - 1
            break
    return start_token, end_token


def model_predict_qa(questions, contexts, qa_lists, tokenizer, model):
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
    # Ensure inputs are in batch form
    if isinstance(questions, str):
        questions = [questions]
    if isinstance(contexts, str):
        contexts = [contexts]

    # Tokenize batch of questions and contexts
    inputs = tokenizer(questions, contexts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Move to device

    # Perform prediction
    with torch.no_grad():
        start_time = time.time()
        outputs = model(**inputs)
        end_time = time.time()
        print(f"Time taken for batch model prediction: {end_time - start_time} seconds")

        predictions_start = torch.softmax(outputs.start_logits, dim=1)
        predictions_end = torch.softmax(outputs.end_logits, dim=1)

    all_probabilities = []
    # Process each question-context pair in the batch
    for index, (start_chars, end_chars) in enumerate(qa_lists):
        probabilities = []
        for start_char, end_char in zip(start_chars, end_chars):
            start_token, end_token = find_pivot(contexts[index], start_char, end_char, tokenizer, inputs.encodings[index])
            if start_token is None or end_token is None:
                probabilities.append(0)
                continue

            valid_start = max(0, min(start_token, predictions_start.size(1) - 1))
            valid_end = max(valid_start, min(end_token, predictions_end.size(1) - 1))

            max_average_prob = 0
            for i in range(valid_start, valid_end + 1):
                for j in range(i, valid_end + 1):
                    average_prob = (predictions_start[index, i] + predictions_end[index, j]) / 2
                    if average_prob > max_average_prob:
                        max_average_prob = average_prob
            probabilities.append(max_average_prob)
        all_probabilities.append(probabilities)

    return all_probabilities

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

