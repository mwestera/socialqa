import os
os.environ['HF_HOME'] = '/home/s3382001/data1/'
import csv
import os
import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModelForQuestionAnswering
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import torch
import click
import sys
import torch
import random
import time
import pandas as pd
"""
Calculate scores for the input file using a QA or Entailment model.

Example:
$ python calculate_scores_cpu.py --model_type qa infile.tsv

"""

@click.command(help="")
@click.argument("infile",type=str, default=sys.stdin)
@click.argument("model_type", type=str)
@click.argument("similarities_file", type=str, default=sys.stdin)
@click.argument("similarities_frac", type=str, default='0.10')
def main(infile, model_type, similarities_file, similarities_frac):
    """
    Main function for calculating scores.

    Args:
        infile (str): Path to the input file.
        qa (bool): Flag indicating whether the input file is for question answering.
        model_type (str): Type of model to use for prediction.

    Returns:
        None
    """
    threshold_value=get_top_percentage_threshold_approx(similarities_file, similarities_frac, model_type,sample_size=10000)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load the model and tokenizer based on the model type
    if model_type == "qa":
        model_name_1 = "google/flan-t5-large"
        model_1 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)
        tokenizer_1 = AutoTokenizer.from_pretrained("google/flan-t5-large")
        model_name_2 = "ahotrod/albert_xxlargev1_squad2_512" 
        model_2 = AutoModelForQuestionAnswering.from_pretrained(model_name_2).to(device)
        tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2) 
    elif model_type == "rte":
        model_1 = AutoModelForSeq2SeqLM.from_pretrained("google/t5_xxl_true_nli_mixture").to(device)
        tokenizer_1 = AutoTokenizer.from_pretrained("google/flan-t5-large")
        #model_name_2 = "soumyasanyal/nli-entailment-verifier-xxl"
        #model_2 = AutoModelForSeq2SeqLM.from_pretrained(model_name_2).to(device)
        #tokenizer_2 = AutoTokenizer.from_pretrained('google/flan-t5-xxl')
        model_2=AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)
        tokenizer_2 = tokenizer_1
    # Process and write scores
    create_html_with_threshold(infile, similarities_file, threshold_value, model_type)
    process_and_write_scores(infile, similarities_file,tokenizer_1, model_1, tokenizer_2, model_2, model_type, threshold_value)

def process_and_write_scores(infile,sim_file, tokenizer_1, model_1, tokenizer_2, model_2, model_type, threshold_value, chunk_size=1):
    """
    Process the file in chunks and write scores incrementally to avoid memory issues.
    """
    root, file_extension = os.path.splitext(infile)

    outfile = f"{root}_scores{file_extension}"
    
    # Open the input and output files
    with open(infile, 'r', encoding='utf-8') as read_file, \
         open(outfile, 'w', newline='', encoding='utf-8') as write_file:
        
        tsv_reader = csv.DictReader(read_file, delimiter='\t')
        fieldnames = tsv_reader.fieldnames +['score albert']+ ['best answer albert']+ ['score flan-t5']+['score nli-entailment-verifier-xxl'] 
        
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
                    print("index is not a integer", flush=True)
                    continue
            # If the model type is QA
            if model_type=="qa":
                qa_list = eval(row['pivot_positions'])  # Convert the string representation of the list to an actual list                  
                if(qa_list[0]>1000):
                    new_post, new_qa_list = reduce_context_size(400, qa_list, row)
                    question = row['question']
                    post = new_post
                    pivot = row['pivot']
                    #Find similarity score here
                    score_sim = get_similarity(sim_file, row["question_id"], row["pivot_id"], model_type)
                    print(score_sim, flush=True)
                    if score_sim<threshold_value:
                      row['score flan-t5'] = 0
                      row['score albert'] = 0
                      row['best answer albert'] = "Not Similar"
                      writer.writerow(row)
                    else:
                      batch_texts.append((question, post, pivot))
                      batch_qas.append(new_qa_list)
                      batch_rows.append(row)
                else:
                    question = row['question']
                    pivot = row['pivot']
                    post = row['post']
                    #Find similarity score here
                    score_sim = get_similarity(sim_file, row["question_id"], row["pivot_id"], model_type)
                    print(score_sim, flush=True)
                    if  score_sim<threshold_value:
                      row['score flan-t5'] = 0
                      row['score albert'] = 0
                      row['best answer albert'] = "Not Similar"
                      writer.writerow(row)

                    else:
                      batch_texts.append((question, post, pivot))
                      batch_rows.append(row)
                      batch_qas.append(qa_list)

            # If the model type is Entailment
            elif model_type  == 'rte':
                sentence1 = row['sentence1']
                sentence2 = row['sentence2']
                score_sim = get_similarity(sim_file, row["pivot_id"], row["entailment_id"], model_type)
                #Find similarity score here
                if score_sim<threshold_value:
                  row['score nli-entailment-verifier-xxl'] = 0
                  row['score flan-t5'] = 0 
                  writer.writerow(row)
                else:
                  batch_texts.append((sentence1, sentence2))
                  batch_rows.append(row)
            if len(batch_texts) >= chunk_size:
                if model_type=='qa':
                    scores_flan, scores_alb, answers_alb = model_predict_qa([texts[0] for texts in batch_texts], [texts[1] for texts in batch_texts], [texts[2] for texts in batch_texts], batch_qas, tokenizer_1, model_1, tokenizer_2, model_2)
                    for idx,(row, score) in enumerate(zip(batch_rows, scores_flan)):
                        print("enter potential target", flush=True)
                        row['score flan-t5'] = scores_flan[idx]
                        row['score albert'] = scores_alb[idx]
                        row['best answer albert'] = answers_alb[idx]
                        writer.writerow(row)
                else:
                    scores_nli,  scores_flan = model_predict_rte([texts[0] for texts in batch_texts], [texts[1] for texts in batch_texts], tokenizer_1, model_1, tokenizer_2, model_2)
                    for idx,(row, score) in enumerate(zip(batch_rows, scores_nli)):
                        print("enter potential target", flush=True)
                        row['score nli-entailment-verifier-xxl'] = scores_nli[idx]
                        row['score flan-t5'] = scores_flan[idx]
                        writer.writerow(row)
                batch_rows, batch_texts, batch_qas= [], [], []  # Reset for next batch

        if batch_rows:
            if model_type =='qa':
                scores_flan, scores_alb, answers_alb = model_predict_qa([texts[0] for texts in batch_texts], [texts[1] for texts in batch_texts], [texts[2] for texts in batch_texts], batch_qas, tokenizer_1, model_1, tokenizer_2, model_2)
                for idx,(row, score) in enumerate(zip(batch_rows, scores_alb)):
                    row['score flan-t5'] = scores_flan[idx]
                    row['score albert'] = scores_alb[idx]
                    row['best answer albert'] = answers_alb[idx]
                    writer.writerow(row)
            else:
                scores_nli, scores_flan = model_predict_rte([texts[0] for texts in batch_texts], [texts[1] for texts in batch_texts], tokenizer_1, model_1, tokenizer_2, model_2)
                for idx,(row, score) in enumerate(zip(batch_rows, scores_nli)):
                    row['score nli-entailment-verifier-xxl'] = scores_nli[idx]
                    row['score flan-t5'] = scores_flan[idx]
                    writer.writerow(row)
def subsample_from_quantiles(file_path, num_quantiles, sample_size_per_quantile):
    """
    Takes a subsample from each quantile of the 'similarity' column in the file.

    Args:
        file_path (str): Path to the TSV file containing similarity data.
        num_quantiles (int): Number of quantiles to divide the data into.
        sample_size_per_quantile (int): Number of samples to take from each quantile.

    Returns:
        pandas.DataFrame: A DataFrame containing the subsampled data.
    """

    # Read the data in chunks
    all_data = []
    for chunk in pd.read_csv(file_path, sep='\t', header=None, names=['id1', 'id2', 'similarity'], chunksize=10000):
        all_data.append(chunk)

    # Concatenate the chunks into a single DataFrame
    all_data = pd.concat(all_data, ignore_index=True)

    # Calculate quantile boundaries
    quantile_bins = pd.qcut(all_data['similarity'], q=num_quantiles, labels=False)

    # Subsample from each quantile
    subsamples = []
    for quantile in range(num_quantiles):
        quantile_data = all_data[quantile_bins == quantile]
        subsample = quantile_data.sample(n=min(sample_size_per_quantile, len(quantile_data)))
        subsamples.append(subsample)

    # Concatenate the subsamples
    return pd.concat(subsamples, ignore_index=True)

def find_sentences_for_ids(pairs_file, id1, id2, model_type):
    """
    Finds the sentences corresponding to given IDs in a pairs file.

    Args:
        pairs_file (str): Path to the file containing sentence pairs.
        id1 (str): The first ID to search for.
        id2 (str): The second ID to search for.

    Returns:
        tuple: A tuple containing the two sentences corresponding to the IDs, or None if not found.
    """
    if model_type =="qa":
      id2_name = 'pivot_id'
      id1_name = 'question_id'
      sentence1_col = "question"
      sentence2_col = "pivot"
    else:
      id1_name = 'pivot_id'
      id2_name = 'entailment_id'
      sentence1_col = "sentence1"
      sentence2_col = "sentence2"
    sentence1 = None
    sentence2 = None

    with open(pairs_file, 'r', encoding='utf-8') as read_file:
      tsv_reader = csv.DictReader(read_file, delimiter='\t')
      for line in tsv_reader:
            current_id1 = line[id1_name]
            current_id2 = line[id2_name]
            if current_id1 == id1:
                sentence1 = line[sentence1_col]
            if current_id2 == id2:
                sentence2 = line[sentence2_col]

            if sentence1 is not None and sentence2 is not None:
                break
    return sentence1, sentence2
    
def create_html_with_threshold(pairs_file, sim_file, threshold_value, model_type, num_quantiles=10, sample_size_per_quantile=20):
    """
    Creates an HTML file showing subsampled data from each quantile with colors.

    Args:
        file_path (str): Path to the TSV file containing similarity data.
        num_quantiles (int): Number of quantiles to divide the data into.
        sample_size_per_quantile (int): Number of samples to take from each quantile.
        output_file (str): Path to the output HTML file.
    """

    subsampled_data = subsample_from_quantiles(sim_file, num_quantiles, sample_size_per_quantile)

    # Generate a color palette for the quantiles
    import colorsys
    colors = [colorsys.hsv_to_rgb(i / num_quantiles, 0.8, 0.8) for i in range(num_quantiles)]
    colors = ['#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]
    output_file = sim_file.replace('.tsv', f'_{model_type}_sample.html')

    with open(output_file, 'w') as f:
        f.write('<html><body><table border="1">\n')
        f.write('<tr><th>Sentence 1</th><th>Sentence 2</th><th>Similarity</th><th>Quantile</th></tr>\n')

        # Calculate quantile bins for the subsampled data
        quantile_bins_subsampled = pd.qcut(subsampled_data['similarity'], q=num_quantiles, labels=False)

        for _, row in subsampled_data.iterrows():
            quantile = quantile_bins_subsampled.loc[row.name]
            color = 'green' if row["similarity"] >= threshold_value else 'red'
            sentence1, sentence2 = find_sentences_for_ids(pairs_file, row['id1'], row['id2'], model_type)
            f.write(f'<tr style="background-color: {color}"><td>{sentence1}</td><td>{sentence2}</td><td>{row["similarity"]}</td><td>{quantile}</td></tr>\n')

def get_similarity(file_path, id1, id2, model_type):
    chunk_size = 10000  # Read file in chunks of 10000 lines
        # Perform reservoir sampling to get a sample of the similarity values
    sample_chunk = pd.read_csv(file_path, sep='\t', header=None, names=['id1', 'id2', 'similarity'], nrows=10)
    
    # Determine the types of 'id1' and 'id2' columns
    id1_type = sample_chunk['id1'].dtype
    id2_type = sample_chunk['id2'].dtype

    # Convert the input IDs to the same type
    id1_converted = id1_type.type(id1)
    id2_converted = id2_type.type(id2)    
    for chunk in pd.read_csv(file_path, sep='\t', header=None, names=['id1', 'id2', 'similarity'], chunksize=chunk_size):
        # Search for the pair of IDs in the current chunk
        match = chunk[(chunk['id1'] == id1_converted) & (chunk['id2'] == id2_converted)]
        match_reverse = chunk[(chunk['id1'] == id2_converted) & (chunk['id2'] == id1_converted)] 
        if not match.empty:
            return float(match.iloc[0]['similarity'])
        elif not match_reverse.empty:
            return float(match_reverse.iloc[0]['similarity'])
    
    return None  # Return None if the pair is not found

def reservoir_sampling(file_path, sample_size):
    sample = []
    chunk_size = 10000  # Read file in chunks of 10000 lines

    for chunk in pd.read_csv(file_path, sep='\t', header=None, names=['id1', 'id2', 'similarity'], chunksize=chunk_size):
        for index, row in chunk.iterrows():
            if len(sample) < sample_size:
                sample.append(row['similarity'])
            else:
                # Replace elements with gradually decreasing probability
                r = random.randint(0, index + 1)
                if r < sample_size:
                    sample[r] = row['similarity']
    
    return sample

def get_top_percentage_threshold_approx(file_path, percentage,model_type, sample_size=10000):
    # Perform reservoir sampling to get a sample of the similarity values
    sample = reservoir_sampling(file_path, sample_size)    

    # Sort the sample
    sample_sorted = sorted(sample, reverse=True)
    
    # Calculate the index for the top X percentage
    top_percentage_index = int(len(sample_sorted) * float(percentage))
    
    # Get the threshold value at the calculated index
    threshold_value = sample_sorted[top_percentage_index - 1]
    print(f"Threshold value for similarities : {threshold_value}")
    return float(threshold_value)

def reduce_context_size(context_start, qa_list, row):
        """
        Reduces the context size of the post and updates the QA list accordingly.
        """
        post = row['post']
        post= post[qa_list[0]-context_start:]
        qa_list = (context_start, context_start+ (qa_list[1]-qa_list[0]))

        return post, qa_list

def model_predict_rte(premises, hypotheses, tokenizer_1, model_1, tokenizer_2, model_2):
    """
    Calculate RTE scores for a chunk of data.
    """
    
    scores_nli = []
    scores_flan = []
    for premise, hypothesis in zip(premises, hypotheses):
        prompt = f"premise: {premise} hypothesis: {hypothesis}"
        input_ids = tokenizer_2(prompt, return_tensors='pt').input_ids.to(model_2.device)
        
        with torch.no_grad():
            pos_ids = tokenizer_2('1').input_ids
            neg_ids = tokenizer_2('0').input_ids
            pos_id = pos_ids[0]
            neg_id = neg_ids[0]
            
            logits = model_1(input_ids, decoder_input_ids=torch.zeros((input_ids.size(0), 1), dtype=torch.long).to(model_2.device)).logits
            pos_logits = logits[:, 0, pos_id]
            neg_logits = logits[:, 0, neg_id]
            posneg_logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits.unsqueeze(-1)], dim=1)
            score_ent = torch.nn.functional.softmax(posneg_logits, dim=1)[:, 0]

        scores_nli.append(score_ent.item())
        with torch.inference_mode():
            output = self.model.generate(input_ids, max_new_tokens=10)
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)

        if len(result) > 1:
            result = result[0]
        if result not in ["0", "1"]:
            print(f'warning: NLI AutoAIS returned "{result}" instead of 0 or 1')
        scores_flan.append(result)
    return scores_nli, scores_flan



def find_best_token_index(predictions, input_ids, tokenizer, skip_tokens):

    sorted_indices = torch.argsort(predictions, descending=True)
    for idx in sorted_indices[0]:
        token_ids = input_ids[idx].unsqueeze(0) 
        token = tokenizer.convert_ids_to_tokens(token_ids)[0]
        if token not in skip_tokens:
            return idx.item(), predictions[0, idx].item()
    return None, None

def find_sublist_in_list(big_list, sublist):
    sublist_length = len(sublist)
    for i in range(len(big_list)):
        if big_list[i:i+sublist_length] == sublist:
            return i
    return -1

def find_pivot(tokenizer, pivot, input_ids):
    # Encode the pivot
    encoded_pivot = tokenizer.encode(pivot)[1:-1]  # Ignore the first and last token

    # Convert the tokenized input to a list
    input_tokens = input_ids.tolist()
    # Check if the encoded pivot is in the tokenized input
    pivot_index = find_sublist_in_list(input_tokens, encoded_pivot)

    start_token_pos = pivot_index
    end_token_pos = start_token_pos + len(encoded_pivot) -1

    return start_token_pos, end_token_pos



def model_predict_qa(questions, posts, pivots, qa_lists, tokenizer_1, model_1, tokenizer_2, model_2):
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
        # Tokenize batch of questions and contexts for both models
    inputs2 = tokenizer_2(questions, posts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs2 = {k: v.to(model_2.device) for k, v in inputs2.items()}  # Move to device

    # Perform prediction for model2
    with torch.no_grad():
        outputs2 = model_2(**inputs2)
        predictions_start2 = torch.softmax(outputs2.start_logits, dim=1)
        predictions_end2 = torch.softmax(outputs2.end_logits, dim=1)

    probabilities_flan = []
    probabilities_alb = []
    best_answers_alb = []
    # Process each question-context pair in the batch
    for idx, input_ids in enumerate(inputs2['input_ids']):
        # Create a new inputs dictionary that contains only the current elemen

        # Call find_pivot with the current inputs
        start_token, end_token = find_pivot(tokenizer_2, pivots[idx], input_ids)

        if start_token == -1:
            probabilities_alb.append(0)
            best_answers_alb.append("")
            continue

        max_average_prob = 0
        best_start, best_end = -1, -1

        for i in range(start_token, end_token +1):
            for j in range(i, end_token + 1):
                average_prob = (predictions_start2[idx, i] + predictions_end2[idx, j]) / 2 
                if average_prob > max_average_prob:
                    max_average_prob = average_prob
                    best_start, best_end = i, j


        probabilities_alb.append(max_average_prob.item())
        input_sequence = tokenizer_2.decode(input_ids[best_start:best_end+1])
        best_answers_alb.append(input_sequence)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for idx, (question,answer) in enumerate(zip(questions,pivots)):
        # Create a new inputs dictionary that contains only the current elemen

        # Call find_pivot with the current inputs
        prompt = f"Question: {question}\nAnswer: {answer}\nDoes this answer correctly respond to the question?\nAnswer:"
        input_ids = tokenizer_1(prompt, return_tensors='pt').input_ids.to(device)
        with torch.no_grad():
            pos_ids = tokenizer_1('Yes', return_tensors='pt').input_ids.to(device)
            neg_ids = tokenizer_1('No', return_tensors='pt').input_ids.to(device)
            pos_id = pos_ids[0, 0]
            neg_id = neg_ids[0, 0]
           
            logits = model_1(input_ids, decoder_input_ids=torch.zeros((input_ids.size(0), 1), dtype=torch.long).to(model_1.device)).logits
            pos_logits = logits[:, 0, pos_id]
            neg_logits = logits[:, 0, neg_id]
            posneg_logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits.unsqueeze(-1)], dim=1)
            scores_qa = torch.nn.functional.softmax(posneg_logits, dim=1)[:, 0]
        probabilities_flan.append(scores_qa.item())
    return probabilities_flan, probabilities_alb, best_answers_alb

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

