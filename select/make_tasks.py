import sys
import click
import random
import itertools
import csv
import time
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

from scoring_methods import get_scalers,calculate_post_score, calculate_question_score, calculate_pivot_score, filter_QA_pair, rank_QA_pair, filter_RTE_pair, rank_RTE_pair

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
@click.option("--n_qa", help="Max how many QA items per user.", type=int, required=False, is_flag=False, default=None)
@click.option("--n_rte", help="Max how many RTE items per user.", type=int, required=False, is_flag=False, default=None)
@click.option("--pdf", help="Path to pdf file to write report to.", type=click.Path(dir_okay=False), required=False, is_flag=False, default=None)
@click.option("--seed", help="Random seed to use.", type=int, required=False, is_flag=False, default=None)
def main(sentences, posts, n_qa, n_rte, pdf, seed):

    logging.basicConfig(level=logging.INFO)

    seed = seed or random.randint(0, 999999)
    random.seed(seed)
    logging.info(f'Seed: {seed}')

    # For each user, will use only the top fraction of pivots/questions/posts:
    questions_frac = .1
    pivots_frac = .05       # TODO: Make these cmd line args? Or config file?
    posts_frac = .5

    outfile_QA = f'{sentences}_pairs_qa.tsv'   # TODO: Make these cmd line args?
    outfile_RTE = f'{sentences}_pairs_rte.tsv'

    report = PdfPages(pdf) if pdf else None

    all_sentences = read_sentences(sentences)
    all_user_posts = read_posts(posts)

    add_sentence_scores(all_sentences)


    add_post_scores(all_user_posts)

    #report_score_percentiles(all_user_posts, all_sentences, report)

    questions = all_sentences.dropna(subset=['question_score'])
    pivots = all_sentences.dropna(subset=['pivot_score'])
    user_posts = all_user_posts.dropna(subset=['post_score'])

    write_to_html(questions, 'question_score', 'text')
    write_to_html(pivots, 'pivot_score', 'text')
    write_to_html(user_posts, 'post_score', 'text')

    # Pre-filter by time? Not super useful.
    # potential_pivots = extract_sorted_subrange_by(potential_pivots, group_by='user_post_author_id', sort_by='user_post_created', between=[.25, .75])
    # potential_questions = extract_sorted_subrange_by(potential_questions, group_by='user_post_author_id', sort_by='user_post_created', between=[0, .75])

    estimate_exhaustive_task_sizes(questions, pivots, user_posts)

    # Use only some highest scoring fraction of sentences/posts per user
    questions_thresholded = threshold_df_by_frac(questions, 'question_score', questions_frac, by='user_post_author_id')
    pivots_thresholded = threshold_df_by_frac(pivots, 'pivot_score', pivots_frac, by='user_post_author_id')

    # TODO: DONT threshold posts for now... because we probably don't want to accidentally
    #       delete the user posts from through which the user interacted with the questions and pivots...
    # user_posts_thresholded = threshold_df_by_frac(user_posts, 'post_score', posts_frac, by='user_post_author_id')
    user_posts_thresholded = user_posts  # temporary

    estimate_exhaustive_task_sizes(questions_thresholded, pivots_thresholded, user_posts_thresholded)

    logging.info('Composing QA pairs.')
    start_time = time.time()
    pairs_QA = select_pairs(questions_thresholded,
                            pivots_thresholded,
                            group_by='user_post_author_id',
                            n=n_qa,
                            filter=filter_QA_pair,
                            ranker=rank_QA_pair)
    end_time = time.time()
    logging.info(f'Selected {len(pairs_QA)} QA pairs. Duration: {end_time - start_time} seconds.')


    write_to_html_pairs(pairs_QA, QA=True)

    logging.info('Composing RTE pairs.')
    start_time = time.time()
    print("start QA pairs")
    pairs_RTE = select_pairs(pivots_thresholded,
                            user_posts_thresholded,
                            group_by='user_post_author_id',
                            n=n_rte,
                            filter=filter_RTE_pair,
                            ranker=rank_RTE_pair)
    end_time = time.time()
    logging.info(f'Selected {len(pairs_RTE)} RTE pairs. Duration: {end_time - start_time} seconds.')

    write_to_html_pairs(pairs_RTE, QA=False)

    write_qa_pairs(pairs_QA, user_posts, outfile_QA)
    write_ent_pairs(pairs_RTE, outfile_RTE)

    report.close()

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
    color = interpolate_color(score, min_score, max_score)
    return f'background-color: {color};'

def write_to_html(posts, score_column, text_column):
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
    top_entries = posts.nlargest(100, score_column)
    bottom_entries = posts.nsmallest(100, score_column)
    combined_entries = pd.concat([top_entries, bottom_entries])
    # Reset the index of the DataFrame
    combined_entries.reset_index(drop=True, inplace=True)

    # Apply the styling function to the DataFrame
    styled = combined_entries.style.apply(lambda row: pd.Series([f'background-color: {interpolate_color(row[score_column], min_score, max_score)};'], index=[text_column]), axis=1)

    # Write the styled DataFrame to an HTML file
    styled.to_html(f'{score_column}_scores.html')

def write_to_html_pairs(pairs, QA):

    # Calculate scores and add as a new column
    pairs['Score'] = pairs.apply(lambda row: rank_QA_pair(row) if QA else rank_RTE_pair(row), axis=1)
    pairs.reset_index(drop=True, inplace=True)
    pairs['Score'] = pd.to_numeric(pairs['Score'], errors='coerce')

    # Determine color based on scores
    min_score = np.nanmin(pairs['Score'])
    max_score = np.nanmax(pairs['Score'])
    pairs['Color'] = pairs['Score'].apply(lambda score: interpolate_color(score, min_score, max_score) if pd.notnull(score) else 'white')
    
    # Convert DataFrame to HTML
    styled_html = pairs.style.apply(lambda x: [f"background-color: {color};" for color in x['Color']], axis=0)
    if QA:
        styled_html.to_html('pairs_QA_scores.html')
    else:
        styled_html.to_html('pairs_RTE_scores.html')

def read_sentences(sentences_file):
    logging.info('Reading sentences.')
    df = pd.read_json(sentences_file, lines=True)
    df['user_post_created'] = pd.to_datetime(df['user_post_created'])
    logging.info(f'Read {len(df)} sentences.')
    return df


def read_posts(posts_file):
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


def add_post_scores(all_user_posts, scale=False):
    """
    For a dataframe with user posts, adds one column: post_score.
    The idea is that only the highest-scoring posts will be used (in the entailment task), to save compute.
    """
    logging.info('Computing post scores.')
    scaler_votes = get_scalers(all_user_posts)
    all_user_posts['post_score'] = all_user_posts.apply(calculate_post_score, axis=1, args=(scaler_votes,))

    if scale:
        all_user_posts['post_score'] = scale_min_max(all_user_posts['post_score'])


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

    if group_by:
        for group_label, subdf1 in tqdm.tqdm(df1.groupby(group_by)):
            subdf2 = df2.loc[df2[group_by] == group_label]
            pairs.extend(select_pairs(subdf1, subdf2, group_by=None, n=n, filter=filter, ranker=ranker))

    else:
        for pair in itertools.product(
            df1.itertuples(),
            df2.itertuples()
        ):
            if filter is None or filter(pair):
                pairs.append(pair)

        random.shuffle(pairs)
        pairs = (sorted(pairs, key=ranker) if ranker else pairs)[:n]

    return pairs


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


# def extract_sorted_subrange_by(df, sort_by, group_by, between):
#     """
#     I thought this would be useful for selecting, e.g., only posts from the middle 50% of the time a user was active.
#     """
#     start, end = between
#     result = (df
#               .sort_values(by=[group_by, sort_by])
#               .groupby([group_by])
#               .apply(lambda group: group.iloc[int(len(group)*start):
#                                               int(len(group)*end)]))
#     return result

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
            id_text_dict[post_id] = {
                'text': title + " "+ selftext,
                'start_end': len(title)+1
            }
        elif row['type'] == 'comment':
            post_id = row['id']
            body = row.get("body", "")
            id_text_dict[post_id] = {
                'text': body,
                'start_end': 0
            }

        submission = row['submission']
        if pd.notnull(submission):
            post_id = submission.get("name")
            title = submission.get("title", "")
            id_text_dict[post_id] = {
                'text': title + " " + submission.get("selftext"),
                'start_end': len(title)+1
            }

        parent = row['parent']
        if pd.notnull(parent):
            post_id = parent.get("id")
            if parent.get("type") == "submission":
                id_text_dict[post_id] = {
                    'text': parent.get("selftext"),
                    'start_end': 0
                }
            elif parent.get("type") == "comment":
                id_text_dict[post_id] = {
                    'text': parent.get("body"),
                    'start_end': 0
                }

        replies = row['replies']
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

def write_ent_pairs(pairs, outfile):
    """
    Writes entailment pairs of sentences to a file in tsv format.

    Args:
        pairs (list): List of tuples containing pivot and post sentences.
        outfile (str): Path to the output file.

    Returns:
        None
    """
    logging.info(f'Writing {len(pairs)} pairs to {outfile} in RTE format.')
    with open(outfile, 'w') as file:
        tsv_writer = csv.writer(file, delimiter='\t')
        tsv_writer.writerow(['index','pivot_id','sentence1', 'sentence2'])
        def replace_last_punctuation(text):
            return re.sub(r'([.?!])[^.?!]*$', ',', text)
        
        for n, (pivot, post) in enumerate(pairs):
            new_post = post.text
            new_pivot = pivot.text
            pivot_id = pivot.snippet_id
            # Handle previous context for post, if available
            if hasattr(post, 'previous') and post.previous is not None:
                prev_post= replace_last_punctuation(post.previous)
                new_post = prev_post + " " +new_post

            # Handle previous context for pivot, if available
            if hasattr(pivot, 'previous') and pivot.previous is not None:
                prev_pivot = replace_last_punctuation(pivot.previous)
                new_pivot = prev_pivot + " " +new_pivot
            new_pivot = new_pivot.replace('\t', ' ')  # Replace tabs with spaces
            new_post = new_post.replace('\t', ' ')  # Replace tabs with spaces
            new_pivot = re.sub(r'\s+', ' ', new_pivot)  # Replace multiple spaces with a single space
            new_post = re.sub(r'\s+', ' ', new_post)  # Replace multiple spaces with a single space
            tsv_writer.writerow([n, pivot_id,new_pivot, new_post])


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
        tsv_writer.writerow(['index','pivot_id','question', 'post', 'pivot_positions'])
        
        # Function to replace the last punctuation mark in a sentence with a comma
        def replace_last_punctuation(text):
            return re.sub(r'([.?!])[^.?!]*$', ',', text)

        pair_positions = {}
        pair_pivots_ids={}
        for (question, pivot) in pairs:
            new_question = question.text
            post_id = pivot.post_id
            pivot_id = pivot.snippet_id
            post = id_text_dict.get(post_id)['text']  # Get the text from the ID using the dictionary

            # Handle previous context for question, if available
            if hasattr(question, 'previous') and question.previous is not None:
                prev_question = replace_last_punctuation(question.previous)
                new_question = prev_question + ". " + new_question

            # Handle previous context for post, if available
            #new_question =new_question.replace('\t', ' ')  # Replace tabs with spaces
            #new_question = re.sub(r'\s+', ' ', new_question)  # Replace multiple spaces with a single space
            #new_post = post.replace('\t', ' ')  # Replace tabs with spaces
            #new_post = re.sub(r'\s+', ' ', new_post)  # Replace multiple spaces with a single space

            #pivot_text = pivot.text.replace('\t', ' ')  # Replace tabs with spaces
            #pivot_text = re.sub(r'\s+', ' ', pivot_text) 

            new_start= new_post.find(pivot.text)
            new_end = new_start + len(pivot.text)

            pair_key = (new_question, post_id)
            pair_post = (new_question,new_post)

            # Add the pivot positions to the dictionary
            if pair_key in pair_positions:
                pair_positions[pair_post].append((new_start, new_end))
            else:
                pair_positions[pair_post] = [(new_start, new_end)]
                pair_pivots_ids[pair_post] = pivot_id

        for index, ((new_question, new_post), positions) in enumerate(pair_positions.items()):
            tsv_writer.writerow([index, pair_pivots_ids[(new_question, new_post)], new_question, new_post, positions])

if __name__ == '__main__':
    main()
