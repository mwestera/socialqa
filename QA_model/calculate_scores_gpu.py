import os
import csv
import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM
import click
import sys
import torch
import random
import time
import pandas as pd
import numpy as np
"""
Calculate scores for the input file using a QA or Entailment model.

Example:
$ python calculate_scores_cpu.py infile.tsv --model_type qa --post_file posts_conservative_v1.jsonl
--similarities_file_n_cont similarities_n_cont.tsv
--similarities_file_cont similarities_file_cont.tsv
--similarities_frac 0.10

"""
# Sets environment where to load model, since LLM's are quite large
os.environ['HF_HOME'] = '/home/s3382001/data1/'


@click.command(help="")
@click.argument("infile",type=str, default=sys.stdin)
@click.argument("model_type", type=str)
@click.argument("post_file",type=str, default=sys.stdin)
@click.argument("similarities_file_n_cont", type=str, default=sys.stdin)
@click.argument("similarities_file_cont", type=str, default=sys.stdin)
@click.argument("similarities_frac", type=str, default='0.10')
def main(infile, model_type,post_file,similarities_file_n_cont, similarities_file_cont, similarities_frac):
    """
    Main function for calculating scores.

    Args:
        infile (str): Path to the input file.
        qa (bool): Flag indicating whether the input file is for question answering.
        model_type (str): Type of model to use for prediction.

    Returns:
        None
    """
    # Get threshold values for {similarities_frac}% of most similar embedding pairs
    threshold_value_nc=get_top_percentage_threshold_approx(similarities_file_n_cont, similarities_frac, model_type,sample_size=10000)
    threshold_value_c=get_top_percentage_threshold_approx(similarities_file_cont, similarities_frac, model_type,sample_size=10000)

    # Plot results in html values and give qunatile samples.
    create_html_with_threshold(infile, post_file, similarities_file_n_cont, similarities_file_cont , threshold_value_nc, model_type, contextual=False)
    create_html_with_threshold(infile, post_file, similarities_file_n_cont, similarities_file_cont , threshold_value_c, model_type, contextual=True)

    # Try using GPU, pass model to GPU
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
        model_2=AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)
        tokenizer_2 = tokenizer_1
        
    # Process and write scores
    #process_and_write_scores(infile, similarities_file,tokenizer_1, model_1, tokenizer_2, model_2, model_type, threshold_value)

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
            if type(row["index"])==str:
                try:
                    row["index"]=int(row["index"])
                except:
                    print("index is not a integer", flush=True)
                    continue
                
            # If the model type is QA
            if model_type == "qa":
                qa_list = eval(row['pivot_positions'])  # Convert the string representation of the list to an actual list                  

                # Pivot outside first 1000 characters
                if qa_list[0]>1000:

                    # If context is too large only pass 400 characters
                    new_post, new_qa_list = reduce_context_size(400, qa_list, row)
                    
                    question = row['question']
                    post = new_post
                    pivot = row['pivot']
                    
                    #Find similarity score here
                    score_sim = get_similarity(sim_file, row["question_id"], row["pivot_id"], model_type)

                    # Embedding similarity does not pass threshold
                    if score_sim<threshold_value:
                      row['score flan-t5'] = 0
                      row['score albert'] = 0
                      row['best answer albert'] = "Not Similar"
                      writer.writerow(row)

                    # Embedding similarity does pass threshold
                    else:
                      batch_texts.append((question, post, pivot))
                      batch_qas.append(new_qa_list)
                      batch_rows.append(row)
                      
                # With pivot in first 1000 characters
                else:
                    question = row['question']
                    pivot = row['pivot']
                    post = row['post']
                    #Find similarity score here
                    score_sim = get_similarity(sim_file, row["question_id"], row["pivot_id"], model_type)

                    # Embedding similarity does not pass threshold
                    if  score_sim<threshold_value:

                      #Write to file
                      row['score flan-t5'] = 0
                      row['score albert'] = 0
                      row['best answer albert'] = "Not Similar"
                      writer.writerow(row)
                      
                    # Embedding similarity does pass threshold
                    else:
                      # Store results
                      batch_texts.append((question, post, pivot))
                      batch_rows.append(row)
                      batch_qas.append(qa_list)

            # If the model type is Entailment
            elif model_type  == 'rte':
                
                sentence1 = row['sentence1']
                sentence2 = row['sentence2']
                score_sim = get_similarity(sim_file, row["pivot_id"], row["entailment_id"], model_type)
                
                # Embedding similarity does not pass threshold
                if score_sim<threshold_value:
                  row['score nli-entailment-verifier-xxl'] = 0
                  row['score flan-t5'] = 0 
                  writer.writerow(row)

                # Embedding similarity does pass threshold
                else:
                  # Store results
                  batch_texts.append((sentence1, sentence2))
                  batch_rows.append(row)

            # Process accepted pairs in batches (chunk size)
            if len(batch_texts) >= chunk_size:
                
                if model_type=='qa':

                    # Model tests for Albert and LLM, if one is preferred over the other save time by deleting code for the other
                    scores_flan, scores_alb, answers_alb = model_predict_qa([texts[0] for texts in batch_texts], [texts[1] for texts in batch_texts], [texts[2] for texts in batch_texts], batch_qas, tokenizer_1, model_1, tokenizer_2, model_2)

                    for idx,(row, score) in enumerate(zip(batch_rows, scores_flan)):
                        # Write to file
                        row['score flan-t5'] = scores_flan[idx]
                        row['score albert'] = scores_alb[idx]
                        row['best answer albert'] = answers_alb[idx]
                        writer.writerow(row)
                        
                else:
                    # Model is tested for two LLM's, if one is preferred over the other save time by deleting code for the other
                    scores_nli,  scores_flan = model_predict_rte([texts[0] for texts in batch_texts], [texts[1] for texts in batch_texts], tokenizer_1, model_1, tokenizer_2, model_2)

                    for idx,(row, score) in enumerate(zip(batch_rows, scores_nli)):
                        # Write to file
                        row['score nli-entailment-verifier-xxl'] = scores_nli[idx]
                        row['score flan-t5'] = scores_flan[idx]
                        writer.writerow(row)
                        
                batch_rows, batch_texts, batch_qas= [], [], []  # Reset for next batch
                
        # Last batch does not have to meet batch size
        if batch_rows:
            
            if model_type =='qa':
                
                scores_flan, scores_alb, answers_alb = model_predict_qa([texts[0] for texts in batch_texts], [texts[1] for texts in batch_texts], [texts[2] for texts in batch_texts], batch_qas, tokenizer_1, model_1, tokenizer_2, model_2)

                for idx,(row, score) in enumerate(zip(batch_rows, scores_alb)):
                    # Write to file
                    row['score flan-t5'] = scores_flan[idx]
                    row['score albert'] = scores_alb[idx]
                    row['best answer albert'] = answers_alb[idx]
                    writer.writerow(row)
            else:

                scores_nli, scores_flan = model_predict_rte([texts[0] for texts in batch_texts], [texts[1] for texts in batch_texts], tokenizer_1, model_1, tokenizer_2, model_2)

                for idx,(row, score) in enumerate(zip(batch_rows, scores_nli)):
                    # Write to file
                    row['score nli-entailment-verifier-xxl'] = scores_nli[idx]
                    row['score flan-t5'] = scores_flan[idx]
                    writer.writerow(row)
                    
def subsample_from_quantiles(file_path,file_path_2, num_quantiles, sample_size_per_quantile):
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
    for chunk in pd.read_csv(file_path, sep='\t', header=None, names=['post1_id', 'post2_id','id1', 'id2', 'similarity_1'], chunksize=10000):
        all_data.append(chunk)

    # Concatenate the chunks into a single DataFrame
    all_data = pd.concat(all_data, ignore_index=True)

    # Calculate quantile boundaries
    quantile_bins = pd.qcut(all_data['similarity_1'], q=num_quantiles, labels=False)

    # Subsample from each quantile
    subsamples = []
    for quantile in range(num_quantiles):
        quantile_data = all_data[quantile_bins == quantile]
        if len(quantile_data) > 0:
            # Sort the quantile data by similarity_1
            quantile_data_sorted = quantile_data.sort_values(by='similarity_1')
            
            # Take the best 10 samples (highest similarity)
            best_samples = quantile_data_sorted.tail(sample_size_per_quantile)
            
            # Take the worst 10 samples (lowest similarity)
            worst_samples = quantile_data_sorted.head(sample_size_per_quantile)
            
            # Combine best and worst samples
            combined_samples = pd.concat([best_samples, worst_samples])
            subsamples.append(combined_samples)
    subsampled_data = pd.concat(subsamples, ignore_index=True)
    additional_data = pd.read_csv(file_path_2, sep='\t', header=None, names=['post1_id', 'post2_id','id1', 'id2', 'similarity_2'])
    # Merge the subsampled data with the additional data on id1 and id2 using a left join
    combined_data = pd.merge(subsampled_data, additional_data, on=['id1', 'id2'], how='left', suffixes=('', '_y'))

    # Drop the duplicated post1_id and post2_id columns from the right DataFrame
    combined_data = combined_data.drop(['post1_id_y', 'post2_id_y'], axis=1)
    # Concatenate the subsamples
    return combined_data

def read_posts(posts_file):
    """
    Read the posts from the file and return them as a DataFrame.
    """
    df = pd.read_json(posts_file, lines=True)
    df['created'] = pd.to_datetime(df['created'])

    # for convenience:
    df['user_post_author_id'] = df['author_id']
    df['text'] = df['selftext']
    df['text'] = df['text'].fillna(df['body'])

    df['text'] = df['text'].replace(to_replace=r'\s+', value=r' ', regex=True)  # TODO: Or do we want to keep newlines for splitting?

    return df


def get_post_dict(posts):
    """
    Code returns a dict where the post's text can be retrieved from the id key.
    Args:
       posts : posts file
    Returns:
       dict: dictionary with text (value) for id (key)
    """
    id_text_dict = {}
    for _, row in posts.iterrows():
        post_id = row['id']
        if row['type'] == 'submission':
            post_id = row['name']
            title = row.get("title", "")
            selftext = row.get("selftext", "")
            if isinstance(title, (float, np.float64)) and np.isnan(title):
              title = ""
            if isinstance(selftext, (float, np.float64)) and np.isnan(selftext):
              selftext = ""
            id_text_dict[post_id] = {
                'text': title + " "+ selftext,
                'start_end': len(title)+1}
        elif row['type'] == 'comment':
            post_id = row['id']
            body = row.get("body", "")
            id_text_dict[post_id] = {
                'text': body,
                'start_end': 0}
        submission = row['submission']
        if pd.notnull(submission):
            post_id = submission.get("name")
            title = submission.get("title", "")
            if isinstance(title, (float, np.float64)) and np.isnan(title):
              title = ""
            if isinstance(selftext, (float, np.float64)) and np.isnan(selftext):
              selftext = ""
            id_text_dict[post_id] = {
                'text': title + " " + submission.get("selftext"),
                'start_end': len(title)+1
            }
        parent = row['parent']
        if pd.notnull(parent):
            post_id = parent.get("id","")
            selftext= parent.get("selftext","")
            title = parent.get("title", "")
            if parent.get("type") == "submission":
                id_text_dict[post_id] = {
                    'text': title + " "+ selftext,
                    'start_end': len(title)+1
                }
            elif parent.get("type") == "comment":
                id_text_dict[post_id] = {
                    'text': parent.get("body"),
                    'start_end': 0}
        replies = row['replies']
        if replies is None or (isinstance(replies, (float, np.float64)) and np.isnan(replies)):
            replies = []
        elif not isinstance(replies, list):
            replies = []
        # If replies is a list and not empty, proceed
        if pd.notnull(replies).any():
            for reply in replies:
                post_id=reply.get("id")
                id_text_dict[post_id] = {
                    'text': reply.get("body"),
                    'start_end': 0
                }    
    return id_text_dict

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
    
def create_html_with_threshold(pairs_file,posts_file, sim_file_nc, sim_file_c, threshold_value, model_type, num_quantiles=10, sample_size_per_quantile=20, contextual=False):
    """
    Creates an HTML file showing subsampled data from each quantile with colors.

    Args:
        file_path (str): Path to the TSV file containing similarity data.
        num_quantiles (int): Number of quantiles to divide the data into.
        sample_size_per_quantile (int): Number of samples to take from each quantile.
        output_file (str): Path to the output HTML file.
    """
    if contextual:
      sim_file_scores =sim_file_c
      sim_file_add = sim_file_nc
    else:
      sim_file_add =sim_file_c
      sim_file_scores = sim_file_nc 
    subsampled_data = subsample_from_quantiles(sim_file_scores, sim_file_add, num_quantiles, sample_size_per_quantile)
    # Generate a color palette for the quantiles
    import colorsys
    colors = [colorsys.hsv_to_rgb(i / num_quantiles, 0.8, 0.8) for i in range(num_quantiles)]
    colors = ['#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]
    output_file = sim_file_scores.replace('.tsv', f'c:{contextual}_{model_type}_sample.html')
    posts = read_posts(posts_file)
    posts_dict = get_post_dict(posts)

    with open(output_file, 'w') as f:
        f.write('<html><body><table border="1">\n')
        f.write('<tr><th>Sentence 1</th><th>Sentence 2</th><th>Post 1</th><th>Post 2</th><th>Similarity Contextual</th><th>Similarity Non Contextual</th><th>Quantile</th></tr>\n')

        # Calculate quantile bins for the subsampled data
        quantile_bins_subsampled = pd.qcut(subsampled_data['similarity_1'], q=num_quantiles, labels=False)

        for _, row in subsampled_data.iterrows():
            quantile = quantile_bins_subsampled.loc[row.name]
            color = 'green' if row["similarity_1"] >= threshold_value else 'red'
            sentence1, sentence2 = find_sentences_for_ids(pairs_file, row['id1'], row['id2'], model_type)
            post1_id, post2_id  = row['post1_id'], row['post2_id']
            post1 = posts_dict[post1_id]['text']
            post2 = posts_dict[post2_id]['text']
            f.write(f'<tr style="background-color: {color}"><td>{sentence1}</td><td>{sentence2}</td><td>{post1}</td><td>{post2}</td><td>{row["similarity_1"]}</td><td>{row["similarity_2"]}</td><td>{quantile}</td></tr>\n')

def get_similarity(file_path, id1, id2, model_type):
    chunk_size = 10000  # Read file in chunks of 10000 lines
    # Perform reservoir sampling to get a sample of the similarity values
    sample_chunk = pd.read_csv(file_path, sep='\t', header=None, names=['post1_id','post2_id','id1', 'id2', 'similarity'], nrows=10)
    
    # Determine the types of 'id1' and 'id2' columns
    id1_type = sample_chunk['id1'].dtype
    id2_type = sample_chunk['id2'].dtype

    # Convert the input IDs to the same type
    id1_converted = id1_type.type(id1)
    id2_converted = id2_type.type(id2)    
    for chunk in pd.read_csv(file_path, sep='\t', header=None, names=['post1_id','post2_id','id1', 'id2', 'similarity'], chunksize=chunk_size):
        # Search for the pair of IDs in the current chunk
        match = chunk[(chunk['id1'] == id1_converted) & (chunk['id2'] == id2_converted)]
        match_reverse = chunk[(chunk['id1'] == id2_converted) & (chunk['id2'] == id1_converted)] 
        if not match.empty:
            return float(match.iloc[0]['similarity'])
        elif not match_reverse.empty:
            return float(match_reverse.iloc[0]['similarity'])
    
    return None  # Return None if the pair is not found

def reservoir_sampling(file_path, sample_size):
    """
    Samples similarities to make a rough estimate on the similarity threshold
    """
    sample = []
    chunk_size = 10000  # Read file in chunks of 10000 lines

    for chunk in pd.read_csv(file_path, sep='\t', header=None, names=['post1_id','post2_id','id1', 'id2', 'similarity'], chunksize=chunk_size):
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
    """
    Calculates threshold at top 1-{percetage}% mark
    """
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

        # Prompt as specificied in huggingface for this model
        prompt = f"premise: {premise} hypothesis: {hypothesis}"
        input_ids = tokenizer_2(prompt, return_tensors='pt').input_ids.to(model_2.device)
        
        with torch.no_grad():
            
            # Output of model from prompt should be either 1 or 0 and sum up to 1.
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
            output = model_1.generate(input_ids, max_new_tokens=10)
        result = tokenizer_2.decode(output[0], skip_special_tokens=True)

        if len(result) > 1:
            result = result[0]
        if result not in ["0", "1"]:
            print(f'warning: NLI AutoAIS returned "{result}" instead of 0 or 1')
        scores_flan.append(result)
    return scores_nli, scores_flan

def find_sublist_in_list(big_list, sublist):
    """
    Look for overlapping tokens between pivot and text
    """
    sublist_length = len(sublist)
    for i in range(len(big_list)):
        if big_list[i:i+sublist_length] == sublist:
            return i
    return -1

def find_pivot(tokenizer, pivot, input_ids):
    """
    Finds tokens of pivot within context
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
    inputs2 = tokenizer_2(questions, posts, padding=True, truncation=True, return_tensors="pt", max_length=512)

    # Pass to GPU
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

        # Pivot not found in text
        if start_token == -1:
            probabilities_alb.append(0)
            best_answers_alb.append("")
            continue

        max_average_prob = 0
        best_start, best_end = -1, -1

        # Get most probable answer
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

    # This code tries QA using an LLM, if one wants to test a model, otherwise delete code
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

