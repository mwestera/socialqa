import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import sys
import tqdm
import argparse
#!/usr/bin/python

from transformers import RobertaTokenizerFast, RobertaModel
from transformers import AutoModel, AutoTokenizer

import torch
import itertools
import sys
import argparse
import csv
from typing import Iterable, Callable, List
import logging
import numpy as np
import jsonlines
import json
import torch
"""
A CLI wrapper around transformers to compute contextualized span embeddings for lines in input, yielding .csv output.

A 'contextualized span embedding' is the embedding of a given span of text, but crucially as processed by a model 
that also saw some surrounding text. Concretely, it computes token embeddings for the full text, then averages 
the embeddings of tokens inside the span, ignoring token embeddings outside the span.

Input is a csv of triples sentence,start,stop, or (with --bracket) lines like "this is a sentence and [I'd like to embed this span] and ignore this."

Will use SpanBERT by default, though note that it is designed for subsentential (<10 words) spans. A solid alternative might be roberta-base, or the original Bert.
"""

def bracketed_reader(lines):
    for line in lines:
        line = line.strip()
        start = line.index('[')
        end = line.index(']') - 1
        text = line.replace('[', '').replace(']', '')
        yield {'text': text, 'start': start, 'end': end}


def csv_reader(lines):
    csvreader = csv.DictReader(lines, fieldnames=['text', 'start', 'end'])
    for d in csvreader:
        d['start'], d['end'] = int(d['start']), int(d['end'])
        yield d

def sentence_id_to_post(posts):
    id_text_dict = {}
    for _, row in posts.iterrows():
        post_id = row['id']
        if row['type'] == 'submission':
            post_id = row['name']
            title = row.get("title", "")
            text = row.get("text","")
            selftext = row.get("selftext", "")
            if isinstance(title, (float, np.float64)) and np.isnan(title):
              title = ""
            if isinstance(selftext, (float, np.float64)) and np.isnan(selftext):
              selftext = ""
            id_text_dict[post_id] = {
                'text': text,
                'start_end': len(title)+1 }
        elif row['type'] == 'comment':
            post_id = row['id']
            text = row.get("text","")
            body = row.get("body", "")
            id_text_dict[post_id] = {'text':text,'start_end': 0}
        submission = row['submission']
        if pd.notnull(submission):
            post_id = submission.get("name")
            text = submission.get("text","")
            title = submission.get("title", "")
            if isinstance(title, (float, np.float64)) and np.isnan(title):
              title = ""
            if isinstance(selftext, (float, np.float64)) and np.isnan(selftext):
              selftext = ""
            id_text_dict[post_id] = {
                'text': text,
                'start_end': len(title)+1
            }
        parent = row['parent']
        if pd.notnull(parent):
            post_id = parent.get("id")
            if parent.get("type") == "submission":
                title = parent.get("title", "")
                text = parent.get("text", "")
                selftext = parent.get("selftext", "")
                id_text_dict[post_id] = {
                    'text': text,
                    'start_end': len(title)+1
                }
            elif parent.get("type") == "comment":
                id_text_dict[post_id] = {
                    'text': parent.get("text"),
                    'start_end': 0
                }
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
def make_contextualized_sentence_transformer(model_name: str, hidden_states_to_use: List[int]) -> Callable:
    # tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    # model = RobertaModel.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    def contextualized_sentence_transformer(spans: Iterable[dict]):
        """
        As input, takes an iterable of dictionaries, each with keys 'text', 'start', and 'end'.
        """
        texts = [t['text'] for t in spans]
        starts = [t['start'] for t in spans]
        ends = [t['end'] for t in spans]
        encoded_input = tokenizer(texts, return_tensors='pt', return_offsets_mapping=True, padding=True)
        encoded_input = {key: tensor.to(device) for key, tensor in encoded_input.items()}

        # mask any tokens outside the span

        span_mask = [[m and start <= s < end for m, (s, e) in zip(mask, offsets)] for (mask, offsets, start, end) in zip(encoded_input['attention_mask'], encoded_input['offset_mapping'], starts, ends)]
        span_mask_tensor = torch.tensor(span_mask, dtype=torch.bool).to(device)    # torch.Size([3, 18])

        # dimensions in comments are for an example batch of 3 sentences, longest 18 tokens, with 4 hidden layers requested.

        output = model(input_ids=encoded_input['input_ids'], attention_mask=encoded_input['attention_mask'], output_hidden_states=True)

        hidden_states = [output['hidden_states'][h] for h in hidden_states_to_use]  # 4  x  torch.Size([3, 18, 768])
        hidden_states_stacked = torch.stack(hidden_states)  # torch.Size([4, 3, 18, 768])

        # set tokens outside the span to nan:
        hidden_states_stacked_masked = hidden_states_stacked.masked_fill(~span_mask_tensor.unsqueeze(0).unsqueeze(-1), torch.nan) # torch.Size([4, 3, 18, 768])

        # for remaining (non-nan) tokens, average first over hidden states then over tokens.
        mean_hidden_state = torch.nanmean(hidden_states_stacked_masked, dim=0)  # torch.Size([3, 18, 768])
        span_embeddings = mean_hidden_state.nanmean(dim=-2)    # torch.Size([3, 768])

        return span_embeddings

    return contextualized_sentence_transformer

# Included here, since only available in Python 3.12...
def batched(iterable, n):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch

def read_posts(posts_file):
    """
    Read the posts from the file and return them as a DataFrame.
    """
    logging.info('Reading posts.')
    df = pd.read_json(posts_file, lines=True)
    df['created'] = pd.to_datetime(df['created'])

    # for convenience:
    df['user_post_author_id'] = df['author_id']
    df['text'] = df['selftext']
    df['text'] = df['text'].fillna(df['body'])

    df['text'] = df['text'].replace(to_replace=r'\s+', value=r' ', regex=True)  # TODO: Or do we want to keep newlines for splitting?

    logging.info(f'Read {len(df)} posts.')
    return df


def subsample_text(text, start, end, max_length=512):
    if len(text) > 2 * max_length:
        mid_point = (start + end) // 2
        half_span = max_length // 2
        new_start = max(mid_point - half_span, 0)
        new_end = min(new_start + max_length, len(text))
        new_text = text[new_start:new_end]

        # Adjust start and end according to the new text
        start_shift = start - new_start
        end_shift = end - new_start
        start = max(start_shift, 0)
        end = min(end_shift, max_length)
        return new_text, start, end
    else:
        return text, start, end


def main( sentence_file, posts_file, output_file, hidden_layers=[1,11,12]):
    model = "google-bert/bert-base-uncased"

    # Change the hidden layers used for 
    hidden = hidden_layers
    user_posts= read_posts(posts_file)
    id_to_text = sentence_id_to_post(user_posts) 
    spans = []
    texts = []
    ids = []
    batch_size = 4  # Set your batch size here

    model = make_contextualized_sentence_transformer(model, hidden)
    with open(output_file, 'a', newline='') as file:  # Use 'a' to append
      writer = csv.writer(file)    
      with jsonlines.open(sentence_file) as reader:
        for n, sentence in enumerate(tqdm.tqdm(reader)):
            span = {} 
            id, text, start, end, post_id = sentence['id'], sentence['text'], sentence['start'], sentence['end'], sentence['post_id']
            post_text = id_to_text[post_id]['text']
            sampled_text, sampled_start, sampled_end = subsample_text(post_text, start, end)

            texts.append(text)
            ids.append(id)
            span['text'] = sampled_text
            span['start'] = sampled_start
            span['end'] = sampled_end
            spans.append(span)
            
            if len(spans) >= batch_size:
                embs = model(spans)
                for id, emb in zip(ids, embs):
                  writer.writerow([id, emb.tolist()])
                spans = []
                texts = []
                ids = []
        if len(spans) > 0:
            embs = model(spans)
            with open(output_file, 'a', newline='') as file:  # Use 'a' to append
                writer = csv.writer(file)
                for id, emb in zip(ids, embs):
                    writer.writerow([id, emb.tolist()])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute similarities between posts.')
    parser.add_argument('sentence_file', type=str, help='Path to the input TSV file.')
    parser.add_argument('posts_file', type=str, help='Path to the input TSV file.')
    parser.add_argument('output_file', type=str, help='Path to save the output TSV file.')
    parser.add_argument('hidden_layers', type=list, default = [1,11,12], help='Path to save the output TSV file.')
    args = parser.parse_args()
    main(args.sentence_file, args.posts_file, args.output_file, args.hidden_layers)
