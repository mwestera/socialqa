import sys
import click
import random
import itertools
import csv
import ast
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import termplotlib
import logging
import tqdm
import re
import json
from typing import List, Tuple
from sklearn.metrics.pairwise import paired_cosine_distances
from scoring_methods import get_scalers, calculate_question_score, calculate_pivot_score, filter_QA_pair, rank_QA_pair, filter_RTE_pair, rank_RTE_pair

# TODO: finish implementation of Squad-format output

"""
Reads posts and sentences and uses these to create QA and entailment tasks of approximately desired magnitude.

Example:
$ python make_tasks.py collected/sentences_conspiracy.jsonl collected/posts_conspiracy_top_anon.jsonl --n_qa 1000 --n_rte 1000 --pdf report.pdf

NOTE: The metrics, scores etc. for filtering/selecting items are in the separate scoring_methods.py.
"""

@click.command(help="")
@click.argument("sentences", type=click.File('r'), default=sys.stdin)
@click.argument("posts", type=click.File('r'), default=None)
@click.argument("post_file_name",type=str, default=sys.stdin)
@click.option('--questions_frac', default=0.10, help='Fraction of questions to use.')
@click.option('--pivots_frac', default=0.10, help='Fraction of pivots to use.')
@click.option('--entailments_frac', default=0.10, help='Fraction of entailments to use.')
@click.option("--n_qa", help="Max how many QA items per user.", type=int, required=False, is_flag=False, default=None)
@click.option("--n_rte", help="Max how many RTE items per user.", type=int, required=False, is_flag=False, default=None)
@click.option("--pdf", help="Path to pdf file to write report to.", type=click.Path(dir_okay=False), required=False, is_flag=False, default=None)
@click.option("--seed", help="Random seed to use.", type=int, required=False, is_flag=False, default=None)
def main(sentences, posts, post_file_name, questions_frac, pivots_frac, entailments_frac, n_qa, n_rte, pdf, seed):
    """
    Main function for generating QA and RTE pairs.

    Args:
        sentences (str): Path to the file containing sentences.
        posts (str): Path to the file containing posts.
        questions_frac (float): Fraction of questions to use.
        pivots_frac (float): Fraction of pivots to use.
        entailments_frac (float): Fraction of entailments to use.
        n_qa (int): Number of QA pairs to generate.
        n_rte (int): Number of RTE pairs to generate.
        pdf (str): Path to the PDF file to save the report.
        seed (int): Seed for random number generation.

    Returns:
        None
    """
    logging.basicConfig(level=logging.INFO)

    seed = seed or random.randint(0, 999999)
    random.seed(seed)
    logging.info(f'Seed: {seed}')
    
    outfile_QA = f'pairs_qa_{post_file_name}.tsv'   # TODO: Make these cmd line args?
    outfile_RTE = f'pairs_rte_{post_file_name}.tsv'

    embeddings_nc = get_embeddings(f'{post_file_name}_embeddings.csv')
    embeddings_c = get_embeddings(f'{post_file_name}_posts_embeddings.tsv')
    report = PdfPages(pdf) if pdf else None

    all_sentences = read_sentences(sentences)
    user_posts = read_posts(posts)

    add_sentence_scores(all_sentences)

    
    questions = all_sentences.dropna(subset=['question_score'])
    pivots = all_sentences[all_sentences['from_user'] == False].dropna(subset=['pivot_score'])
    entailments = all_sentences[all_sentences['from_user'] == True].dropna(subset=['pivot_score'])

    write_to_html(questions, 'question_score')
    write_to_html(pivots, 'pivot_score')
    write_to_html(entailments, 'pivot_score')

    estimate_exhaustive_task_sizes(questions, pivots, entailments)

    # Filter low scoring sentences
    questions_thresholded = threshold_df_by_frac(questions, 'question_score', questions_frac, by='user_post_author_id')
    questions_thresholded = questions_thresholded.drop_duplicates()

    pivots_thresholded = threshold_df_by_frac(pivots, 'pivot_score', pivots_frac, by='user_post_author_id')
    pivots_thresholded = pivots_thresholded.drop_duplicates()

    entailments_thresholded = threshold_df_by_frac(entailments, 'pivot_score', entailments_frac, by='user_post_author_id')
    entailments_thresholded = entailments_thresholded.drop_duplicates()

    estimate_exhaustive_task_sizes(questions_thresholded, pivots_thresholded, entailments_thresholded)

    logging.info('Composing QA pairs.')
    
    pairs_QA = select_pairs(questions_thresholded,
                            pivots_thresholded,
                            group_by='user_post_author_id',
                            n=n_qa,
                            filter=filter_QA_pair,
                            ranker=rank_QA_pair)
    
    logging.info(f'Selected {len(pairs_QA)} QA pairs.')

    calculate_similarity(pairs_QA, embeddings_nc, post_file_name+"_nc_qa", include_posts=True)
    calculate_similarity(pairs_QA, embeddings_c, post_file_name+"_c_qa", include_posts=True)

    pairs_QA = (sorted(pairs_QA, key=lambda pair: rank_QA_pair(pair)) if rank_QA_pair else pairs_QA)[:n_qa]

    logging.info(f'Selected {len(pairs_QA)} QA pairs, after filter')
    write_to_html_pairs(pairs_QA, QA=True)

    logging.info('Composing RTE pairs.')
    pairs_RTE= select_pairs(pivots_thresholded,
                             entailments_thresholded,
                             group_by='user_post_author_id',
                             n=n_rte,
                             filter=filter_RTE_pair,
                             ranker=rank_RTE_pair)
    logging.info(f'Selected {len(pairs_RTE)} RTE pairs.')

    calculate_similarity(pairs_RTE, embeddings_nc, post_file_name+"_nc_rte", include_posts=True)
    calculate_similarity(pairs_RTE, embeddings_c, post_file_name+"_c_rte", include_posts=True)

    pairs_RTE = (sorted(pairs_RTE, key=lambda pair: rank_RTE_pair(pair)) if rank_RTE_pair else pairs_RTE)[:n_rte]

    logging.info(f'Selected {len(pairs_RTE)} RTE pairs, after filter')

    # Write to files
    write_to_html_pairs(pairs_RTE, QA=False)
    write_qa_pairs(pairs_QA, user_posts, outfile_QA)
    write_ent_pairs(pairs_RTE,user_posts, outfile_RTE)

    # Interpolate between red and green
def interpolate_color(score, min_val, max_val):
    """
    Calculate color based on interpolation between red and green.
    Args:
    - score: Current value
    - min_val: Minimum value for scaling
    - max_val: Maximum value for scaling

    Returns:
    - A CSS color string.
    """
    norm_val = (score - min_val) / (max_val - min_val) if max_val > min_val else 0.5
    red = int(255 * (1 - norm_val))
    green = int(255 * norm_val)
    blue = 0
    return f"rgb({red},{green},{blue})"

def style_text(score, min_score, max_score):
    """
    Style the text based on the score.
    """
    color = interpolate_color(score, min_score, max_score)
    return f'background-color: {color};'

def write_to_html(posts, score_column):
    """
    Write a styled DataFrame to an HTML file.

    Args:
        posts (pandas.DataFrame): The DataFrame containing the posts data.
        score_column (str): The name of the column representing the scores.
        text_column (str): The name of the column representing the text.

    Returns:
        None
    """
    posts.reset_index(drop=True, inplace=True)
    posts[score_column] = pd.to_numeric(posts[score_column], errors='coerce')

    min_score = np.nanmin(posts[score_column])
    max_score = np.nanmax(posts[score_column])

    # Sort by the score column and take the top and bottom 100 entries
    # Calculate quantiles
    posts['Quantile'] = pd.qcut(posts[score_column], q=10, duplicates='drop')

    # Select 10 from each quantile
    selected_entries = posts.groupby('Quantile').apply(lambda group: group.head(10))

    # Reset index
    selected_entries.reset_index(drop=True, inplace=True)

    # Apply the styling function to the DataFrame
    styled = selected_entries.style.apply(lambda row: ['background-color: {}'.format(interpolate_color(row[score_column], min_score, max_score))]*len(row), axis=1)
    # Write the styled DataFrame to an HTML file
    styled.to_html(f'{score_column}_scores.html')

def write_to_html_pairs(pairs, QA):
    data = []
    for idx, (sent1, sent2) in enumerate(pairs):
        data.append({
            'Sentence 1 Text': sent1.text,
            'Sentence 1 Question Score': sent1.question_score,
            'Sentence 1 Pivot Score': sent1.pivot_score,
            'Sentence 1 Subreddit Name': sent1.subreddit_name,
            'Sentence 2 Text': sent2.text,
            'Sentence 2 Question Score': sent2.question_score,
            'Sentence 2 Pivot Score': sent2.pivot_score,
            'Sentence 2 Subreddit Name': sent2.subreddit_name,
            'post1': sent1,
            'post2': sent2
        })

    df = pd.DataFrame(data)
    print(df.head())
    df['Score'] = df.apply(lambda row: rank_QA_pair((row['post1'], row['post2'])) if QA else rank_RTE_pair((row['post1'], row['post2'])), axis=1)
    df.reset_index(drop=True, inplace=True)
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')

    # Calculate quantiles
    df['Quantile'] = pd.qcut(df['Score'], q=10, duplicates='drop')

    # Select 10 from each quantile
    df = df.groupby('Quantile').apply(lambda group: group.head(10))

    # Reset index
    df.reset_index(drop=True, inplace=True)

    # Determine color based on scores
    min_score = np.nanmin(df['Score'])
    max_score = np.nanmax(df['Score'])
    # Convert DataFrame to HTML
    df['Color'] = df['Score'].apply(lambda score: interpolate_color(score, min_score, max_score) if pd.notnull(score) else 'white')
    df = df.drop(['post1', 'post2'], axis=1)
    # Convert DataFrame to HTML with styling
    def apply_color(row):
        return [f"background-color: {row['Color']};" for _ in row]
    
    styled_html = df.style.apply(apply_color, axis=1)

    if QA:
        styled_html.to_html('pairs_QA_scores.html')
    else:
        styled_html.to_html('pairs_RTE_scores.html')
    return

def read_sentences(sentences_file):
    """
    Read the sentences from the file and return them as a DataFrame.
    """
    logging.info('Reading sentences.')
    df = pd.read_json(sentences_file, lines=True)
    df['user_post_created'] = pd.to_datetime(df['user_post_created'])
    logging.info(f'Read {len(df)} sentences.')
    return df


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


def add_sentence_scores(all_sentences, scale=False):
    """
    For a dataframe with sentences, adds two columns: pivot_score and question_score.
    The idea is that only the highest-scoring ones will be used (in the QA and entailment tasks), to save compute.
    """
    logging.info('Computing sentence scores.')
    scaler_votes = get_scalers(all_sentences)
    all_sentences['pivot_score'] = all_sentences.apply(calculate_pivot_score, axis=1, args=(scaler_votes,))
    all_sentences['question_score'] = all_sentences.apply(calculate_question_score, axis=1, args=(scaler_votes,))

    if scale:
        all_sentences['pivot_score'] = scale_min_max(all_sentences['pivot_score'])
        all_sentences['question_score'] = scale_min_max(all_sentences['question_score'])


def estimate_exhaustive_task_sizes(questions, pivots, posts) -> None:
    """
    From three dataframes: if we did not do any further filtering, how many QA pairs and RTE pairs would we have?
    """
    n_qa = []
    n_rte = []
    for user, pivots_of_user in pivots.groupby('user_post_author_id'):
        questions_of_user = questions.loc[questions['user_post_author_id'] == user]
        posts_of_user = posts.loc[posts['user_post_author_id'] == user]
        n_qa.append(len(questions_of_user) * len(pivots_of_user))
        n_rte.append(len(pivots_of_user) * len(posts_of_user))
    sum_qa = sum(n_qa)
    sum_rte = sum(n_rte)
    logging.info(f'Exhaustive estimates: {sum_qa:,} QA pairs ; {sum_rte:,} RTE pairs )')


def threshold_df_by_frac(df, criterion, frac, by):
    """
    Finds the lowest score in the top fraction of scores, then removes all
    rows with a lower score than that.
    Since many rows may have the same score, this means the resulting rows will be MORE than the fraction of the original rows.
    """

    result = (df
            .sort_values(by=[by, criterion], ascending=False)
            .groupby([by])
            .apply(lambda group: group.loc[group[criterion] >= group.iloc[int(frac*len(group))][criterion]])
            .reset_index(drop=True)
            )

    logging.info(f'Thresholded {criterion} to {frac}: from {len(df):,} to {len(result):,}.')

    return result


def select_pairs(df1,
                 df2,
                 group_by : str = None,
                 n: int = None,
                 filter: callable = None,
                 ranker: callable = None) -> List[Tuple]:
    """
    Selecting pairs of items from two dataframes, optionally per sub-group, subject to filtering, ranking and
    keeping only the top n.

    Example at hand: if df1 has the questions, and df2 has the pivots, we can select the top-n-best question+pivot
    combinations per user.
    """

    pairs = []
    df1 = df1.drop_duplicates(subset='text')
    df2 = df2.drop_duplicates(subset='text')
    if group_by:
        for group_label, subdf1 in tqdm.tqdm(df1.groupby(group_by)):
            subdf2 = df2.loc[df2[group_by] == group_label]
            pairs_2 = select_pairs(subdf1, subdf2, group_by=None, n=n, filter=filter, ranker=ranker)
            pairs.extend(pairs_2)

    else:
        for pair in itertools.product(
            df1.itertuples(),
            df2.itertuples()
        ):
            if pair[0] == pair[1]:
                continue  # Skip this pair
            if filter is None or filter(pair):
                pairs.append(pair)
        
        random.shuffle(pairs)
    return pairs

def calculate_similarity(pairs, embeddings, post_file_name, include_posts=True):
    """
    Calculate the cosine similarity between the embeddings of the sentences in the pairs.
    """
    with open(f'{post_file_name}_similarities.tsv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for pair in pairs:
            sent1_id = pair[0].id
            sent2_id = pair[1].id
            post1_id = pair[0].post_id
            post2_id = pair[1].post_id
            sent1_embedding = embeddings.get(sent1_id)
            sent2_embedding = embeddings.get(sent2_id)

            if sent1_embedding is not None and sent2_embedding is not None:
                similarity = 1 - paired_cosine_distances([sent1_embedding], [sent2_embedding])[0]
                if include_posts:
                  writer.writerow([post1_id, post2_id, sent1_id, sent2_id, similarity])
                else:
                  writer.writerow([sent1_id, sent2_id, similarity])
    return 

def get_embeddings(file_path):
    """
    Read embeddings from a file and return them as a dictionary.
    """
    embeddings = {}
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip the header
        for row in reader:
            post_id = row[0]
            embedding = []
            if isinstance(row[1], str) and row[1].startswith('[') and row[1].endswith(']'):
                    # Parse string-encoded list
               try:
                 parsed_list = ast.literal_eval(row[1])
                    # If value is a list, flatten it and add to embedding
                 embedding.extend([float(v) for v in parsed_list])
               except ValueError as e:
                  logging.error(f'ValueError: {e} for row: {row}')
                  continue
            else:
                    # Otherwise, add the single float value to embedding
               embedding = [float(value) for value in row[1:]]
            embeddings[post_id] = np.array(embedding, dtype=float)
    return embeddings


def report_score_percentiles(posts, sentences, pdf=None):
    """
    Computes score percentiles, and then reports them (optionally to pdf) with a table and plot.
    """

    logging.info('Computing score percentiles.')

    posts_description = posts[['post_score']].describe(percentiles=[.9, .95, .98, .99, .995, .999])
    posts_nonna = pd.DataFrame(posts[['post_score']].count() / len(posts), columns=['% (of total inc. nan)']).transpose()
    posts_description = pd.concat([posts_description.iloc[0:1], posts_nonna, posts_description.iloc[2:]])

    sentences_description = sentences[['pivot_score', 'question_score']].describe(percentiles=[.9, .95, .98, .99, .995, .999])
    posts_nonna = pd.DataFrame(sentences[['pivot_score', 'question_score']].count() / len(sentences), columns=['% (of total inc. nan)']).transpose()
    sentences_description = pd.concat([sentences_description.iloc[0:1], posts_nonna, sentences_description.iloc[2:]])

    description = pd.concat([posts_description, sentences_description], axis=1)

    if pdf:
        write_table(description, pdf)

    plot_score_percentiles(posts, sentences, pdf)

    return description


def write_table(df, pdf=None):
    """
    A fairly generic table writing function, though currently used only for writing the score percentiles table.
    """
    logging.info(df.to_string(float_format='{:.2f}'.format))

    if pdf:
        logging.info(f'Writing table to PDF {pdf._filename}.')
        df = df.map(lambda x: f'{x:.2f}')
        df = df.reset_index()
        fig = plt.figure(figsize=(12, 6))
        ax = plt.subplot(111)
        ax.axis('off')
        c = df.shape[1]
        contents = [[n for n in row] for row in np.vstack([df.columns, df.values])]
        ax.table(cellText=contents,
                 cellColours=[['lightgray'] * c] + [['none'] * c] * 12,
                 bbox=[0, 0, 1, 1])
        pdf.savefig()
        fig.clear()


def plot_score_percentiles(posts, sentences, pdf=None):
    """
    Creates a figure with three plots, each by calling plot_score_distribution.
    """
    logging.info('Plotting score percentiles.')
    fig, (left, mid, right) = plt.subplots(1, 3, figsize=(20, 10))
    plot_score_distribution(sentences, 'pivot_score', group_by='user_post_author_id', ylim=(0, 5000), ax=left)
    plot_score_distribution(sentences, 'question_score', group_by='user_post_author_id', ylim=(0, 1000), ax=mid)
    plot_score_distribution(posts, 'post_score', group_by='author_id', ylim=(0, 500), ax=right)

    if pdf:
        pdf.savefig()
        fig.clear()
    else:
        plt.show()


def plot_score_distribution(df, score_label, group_by, ax=None, ylim=None):
    """
    Generates a seaborn lineplot for the chosen score_label, optionally grouped by, e.g., username, and rescaling.
    Also logs a terminal interface histogram.

    Called thrice by plot_score_percentiles.
    """
    df_nonna = df.dropna(subset=[score_label])
    score_counts = df_nonna.groupby(([group_by] if group_by else []) + [score_label]).agg(count=('id', 'count')).reset_index()

    ax = sns.lineplot(score_counts, x=score_label, y='count', hue=group_by, alpha=0.4, estimator=None, legend=False, ax=ax)
    sns.lineplot(score_counts, x=score_label, y='count', estimator='mean', color='black', linewidth=2, ax=ax)
    ax.set(title=f'Non-null, min-max-scaled {score_label} (N={len(df_nonna)}; {100*len(df_nonna) / len(df):.2f}%)')
    if ylim:
        ax.set_ylim(*ylim)

    # logging ascii histogram just because we can:
    scores = df[score_label]
    scores_nonna = df_nonna[score_label]
    counts, bin_edges = np.histogram(scores_nonna, bins=8)
    fig = termplotlib.figure()
    fig.hist(counts, bin_edges, force_ascii=False)
    logging.info(f'Non-null pivot scores (N={len(scores_nonna)} ({100*len(scores_nonna) / len(scores):.2f}%)):\n{fig.get_string()}\nmin={min(scores_nonna)}, max={max(scores_nonna)}, '
                  f'mean={sum(scores_nonna) / len(scores_nonna):.2f},')

def create_id_text_dict(posts):
    """
    Create a dictionary mapping post IDs to their text content.

    Parameters:
    - posts (DataFrame): The DataFrame containing the posts.

    Returns:
    - dict: A dictionary mapping post IDs to their text content.
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
            post_id = parent.get("id")
            if parent.get("type") == "submission":
                title = parent.get("title", "")
                selftext = parent.get("selftext", "")
                id_text_dict[post_id] = {
                    'text':title + " "+ selftext,
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

def scale_min_max(series: pd.Series):
    """
    Linearly scale a numerical series to the [0,1] interval with min-max scaling.
    """
    seriesmin = series.min()
    return (series - seriesmin) / (series.max() - seriesmin)

def write_pairs_to_list(pairs, outfile):
    with open(outfile, 'w') as file:
        tsv_writer = csv.writer(file, delimiter='\t')
        tsv_writer.writerow(['index', 'sentence1', 'sentence2'])
        for n, (pivot, post) in enumerate(pairs):
            tsv_writer.writerow([n, post.text, pivot.text])

def write_ent_pairs(pairs,posts,outfile):
    """
    Writes entailment pairs of sentences to a file in tsv format.

    Args:
        pairs (list): List of tuples containing pivot and post sentences.
        outfile (str): Path to the output file.

    Returns:
        None
    """
    id_text_dict = create_id_text_dict(posts)
    logging.info(f'Writing {len(pairs)} pairs to {outfile} in RTE format.')
    with open(outfile, 'w') as file:
        tsv_writer = csv.writer(file, delimiter='\t')
        tsv_writer.writerow(['index','pivot_user_id','pivot_id', 'entailment_user_id', 'entailment_id','sentence1','post1', 'sentence2','post2'])
        def replace_last_punctuation(text):
            return re.sub(r'([.?!])[^.?!]*$', ',', text)
        
        for n, (pivot, ent) in enumerate(pairs):
            new_post = ent.text
            new_pivot = pivot.text
            pivot_id = pivot.id
            entailment_id = ent.id
            post_id_1 = pivot.post_id
            post_id_2 = ent.post_id
            post_pivot_text = id_text_dict[post_id_1]['text']
            post_ent_text = id_text_dict[post_id_2]['text']
            pivot_user_id = pivot.user_post_author_id
            entailment_user_id = ent.user_post_author_id
                
            new_pivot = new_pivot.replace('\n', ' ')
            new_post = new_post.replace('\n', ' ')
            tsv_writer.writerow([n, pivot_user_id, pivot_id, entailment_user_id, entailment_id, new_pivot, post_pivot_text, new_post, post_ent_text])


def write_qa_pairs(pairs, posts, outfile):
    """
    Write question-answer pairs to a file in tsv format.

    Args:
        pairs (list): List of tuples containing question and pivot objects.
        posts (list): List of post objects.
        outfile (str): Path to the output file.

    Returns:
        None
    """
    logging.info("Creating ID to text dictionary.")
    id_text_dict = create_id_text_dict(posts)  # Create the ID to text dictionary

    logging.info(f'Writing {len(pairs)} pairs to {outfile} in tsv format.')
    with open(outfile, 'w', newline='') as file:
        tsv_writer = csv.writer(file, delimiter='\t')
        tsv_writer.writerow(['index','user_id_question','question_id','user_id_pivot','pivot_id','post_id','question', 'post', 'pivot','pivot_positions'])
        
        # Function to replace the last punctuation mark in a sentence with a comma
        def replace_last_punctuation(text):
            return re.sub(r'([.?!])[^.?!]*$', ',', text)

        pair_positions = {}
        pair_pivots_ids={}
        for idx, (question, pivot) in enumerate(pairs):
            new_question = question.text
            post_id = pivot.post_id
            pivot_id = pivot.id
            question_user_id = question.user_post_author_id
            pivot_user_id = pivot.user_post_author_id
            post_text = id_text_dict[post_id]['text']
            
            question_id = question.id
            post = id_text_dict.get(post_id)  # Get the text from the ID using the dictionary
            # Handle previous context for question, if available

            new_start= post_text.find(pivot.text)
            new_end = new_start + len(pivot.text)
            new_question = new_question.replace('\n', ' ')
            new_post = post_text.replace('\n', ' ')
            new_pivot = pivot.text.replace('\n', ' ')
            tsv_writer.writerow([idx, question_user_id, question_id, pivot_user_id, pivot_id, post_id, new_question, new_post, new_pivot, (new_start, new_end)])

if __name__ == '__main__':
    main()
