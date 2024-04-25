import csv
import os
import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModelForQuestionAnswering
import torch
import click
import sys
import torch
import time
@click.command(help="")
@click.argument("infile",type=str, default=sys.stdin)
@click.option("--qa", help="Max how many QA items per user.", type=bool, required=False, is_flag=False, default=None)
def main(infile, qa):
    """
    Main function for calculating scores.

    Args:
        infile (str): Path to the input file.
        qa (bool): Flag indicating whether the input file is for question answering.

    Returns:
        None
    """
    # Assuming the setup for tokenizer and model is done as shown previously
    rte=False
    if "qa" in infile:
        qa = True
    elif "rte" in infile:
        rte=True
        
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
    outfile = f"{root}_scores{file_extension}_2"
    
    with open(infile, 'r', encoding='utf-8') as read_file, \
         open(outfile, 'w', newline='', encoding='utf-8') as write_file:
        
        tsv_reader = csv.DictReader(read_file, delimiter='\t')
        fieldnames = tsv_reader.fieldnames + ['score'] 
        
        writer = csv.DictWriter(write_file, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        
        for row in tqdm.tqdm(tsv_reader):
            # Check if the index is a string and convert it to an integer else it will not be processed
            if(type(row["index"])==str):
                try:
                    row["index"]=int(row["index"])
                except:
                    continue

            if(qa):
                qa_list = eval(row['pivot_positions'])  # Convert the string representation of the list to an actual list
                max_score = 0
                print("size post: ",len(row['post']))
                if(qa_list[0][0]>1000):

                    context_start=400
                    post = row['post']

                    post= post[qa_list[0][0]-context_start:]
                    qa_list[0] = (context_start, context_start+ (qa_list[0][1]-qa_list[0][0]))
                    probabilities = model_predict_qa(row['question'], post, qa_list, tokenizer, model)
                else:
                    probabilities = model_predict_qa(row['question'], row['post'], qa_list, tokenizer, model)
                for score in probabilities:
                    if score > max_score:
                        max_score = score
                row['score'] = max_score
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
    """
    encoded_sentence = tokenizer.encode(post[start_char:end_char])
    start_token = None
    end_token = None
    threshold = 0.8  # Set your threshold here

    # Iterate through the input tokens
    for i in range(len(inputs.input_ids[0])):
        # Extract the corresponding part from the input tokens
        token_part = inputs.input_ids[0][i+1:i+len(encoded_sentence)-1].tolist()

        # Calculate the number of matching tokens
        matching_tokens = sum(1 for a, b in zip(token_part, encoded_sentence[1:-1]) if a == b)

        # Check if the number of matching tokens exceeds the threshold
        if matching_tokens / len(encoded_sentence[1:-1]) >= threshold:
            start_token = i+1
            end_token = i + len(encoded_sentence) - 1
            break

    return start_token, end_token
    """

    encoded_sentence = tokenizer.encode(post[start_char:end_char])
    start_token = None
    end_token = None
    # Iterate through the input tokens
    for i in range(len(inputs.input_ids[0])):
        # Check if the current token matches the start of the encoded sentence
        if inputs.input_ids[0][i+2:i+len(encoded_sentence)-1].tolist()== encoded_sentence[2:-1]:
            start_token = i+1
            end_token = i + len(encoded_sentence) - 1
            break
    return start_token, end_token


def model_predict_qa(sentence1, post, qa_list, tokenizer, model):
    """
    Predicts the probabilities of answers for a given question-answer list using a QA model.

    Args:
        sentence1 (str): The input sentence or context.
        post (str): The input post or question.
        qa_list (list): A list of tuples representing the character positions of the answers in the post.
        tokenizer: The tokenizer used to encode the inputs.
        model: The QA model used for prediction.

    Returns:
        list: A list of probabilities corresponding to each answer in the qa_list.
    """
    with torch.no_grad():
        start_time = time.time()
        inputs = tokenizer(sentence1, post, return_tensors="pt", padding=True, max_length=512, truncation=True)
        outputs = model(**inputs)
        end_time = time.time()
        print(f"Time taken for model prediction: {end_time - start_time} seconds")

    # Get softmax of logits
    predictions_start = torch.softmax(outputs.start_logits, dim=1)
    predictions_end = torch.softmax(outputs.end_logits, dim=1)

    # Obtain the mappings from tokens to characters
    token_to_chars = inputs.encodings[0].offsets
    probabilities = []

    for start_char, end_char in qa_list:
        start_token, end_token = find_pivot(tokenizer, post, inputs, start_char, end_char)
        if start_token is None or end_token is None:
            print(start_char,end_char)
            print(start_token, end_token)
            print(post[start_char:end_char])
            print("Pivot is outside of the tokens. Shifting the post...")

            # Calculate the pivot's position relative to the start of the post

            # Calculate the new start and end positions of the post in terms of tokens
            #new_start = max(0, start_char - 256)

            # Extract the new post
            #new_post = post[new_start:]

            # Recalculate the start and end characters of the answer
            #new_start_char = max(0, start_char - 256)
            #new_end_char = end_char - 256

            # Call the function recursively with the new post and answer positions
            #probabilities.append(model_predict_qa(sentence1, new_post, [(new_start_char, new_end_char)], tokenizer, model)[0])
            continue

        valid_start = max(0, min(start_token, predictions_start.size(1) - 1))
        valid_end = max(valid_start, min(end_token, predictions_end.size(1) - 1))

        # Compute the maximum average probability in the range
        max_average_prob = 0
        for i in range(valid_start, valid_end + 1):
            for j in range(i, valid_end + 1):
                average_prob = (predictions_start[0, i] + predictions_end[0, j]) / 2
                if average_prob > max_average_prob:
                    max_average_prob = average_prob
        probabilities.append(max_average_prob)

    return probabilities

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

