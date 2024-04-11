import sys
import click
import random
import itertools
import csv

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import termplotlib
import logging
import tqdm

from typing import List, Tuple

from scoring_methods import get_post_score, get_question_score, get_pivot_score, filter_QA_pair, rank_QA_pair, filter_RTE_pair, rank_RTE_pair

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

    outfile_QA = 'temp12345_qa.jsonl'   # TODO: Make these cmd line args?
    outfile_RTE = 'temp12345_rte.tsv'

    report = PdfPages(pdf) if pdf else None

    all_sentences = read_sentences(sentences)
    all_user_posts = read_posts(posts)

    add_sentence_scores(all_sentences)
    add_post_scores(all_user_posts)

    report_score_percentiles(all_user_posts, all_sentences, report)

    questions = all_sentences.dropna(subset=['question_score'])
    pivots = all_sentences.dropna(subset=['pivot_score'])
    user_posts = all_user_posts.dropna(subset=['post_score'])

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
    pairs_QA = select_pairs(questions_thresholded,
                            pivots_thresholded,
                            group_by='user_post_author_id',
                            n=n_qa,
                            filter=filter_QA_pair,
                            ranker=rank_QA_pair)
    logging.info(f'Selected {len(pairs_QA)} QA pairs.')

    logging.info('Composing RTE pairs.')
    pairs_RTE = select_pairs(pivots_thresholded,
                             user_posts_thresholded,
                             group_by='user_post_author_id',
                             n=n_rte,
                             filter=filter_RTE_pair,
                             ranker=rank_RTE_pair)
    logging.info(f'Selected {len(pairs_RTE)} RTE pairs.')

    write_pairs_to_squad_format(pairs_QA, user_posts, outfile_QA)
    write_pairs_to_rte_format(pairs_RTE, outfile_RTE)

    report.close()


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
    all_sentences['pivot_score'] = all_sentences.apply(get_pivot_score, axis=1)
    all_sentences['question_score'] = all_sentences.apply(get_question_score, axis=1)

    if scale:
        all_sentences['pivot_score'] = scale_min_max(all_sentences['pivot_score'])
        all_sentences['question_score'] = scale_min_max(all_sentences['question_score'])


def add_post_scores(all_user_posts, scale=False):
    """
    For a dataframe with user posts, adds one column: post_score.
    The idea is that only the highest-scoring posts will be used (in the entailment task), to save compute.
    """
    logging.info('Computing post scores.')
    all_user_posts['post_score'] = all_user_posts.apply(get_post_score, axis=1)

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
    logging.info(f'Exhaustive estimates: {sum_qa:,} QA pairs (avg. {sum_qa/len(n_qa):,.2f} per user); {sum_rte:,} RTE pairs (avg. {sum_rte/len(n_rte):,.2f})')


def threshold_df_by_frac(df, criterion, frac, by):
    """
    Finds the lowest score in the top fraction of scores, then removes all
    rows with a lower score than that.
    Since many rows may have the same score, this means the resulting rows will be MORE than the fraction of the original rows.
    """

    result = (df
              .sort_values(by=[by, criterion], ascending=False)
              .groupby([by])
              .apply(lambda group: group.loc[group[criterion] >= group.iloc[int(frac*len(group))][criterion]], include_groups=False)
              .reset_index(drop=False)
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


def scale_min_max(series: pd.Series):
    """
    Linearly scale a numerical series to the [0,1] interval with min-max scaling.
    """
    seriesmin = series.min()
    return (series - seriesmin) / (series.max() - seriesmin)


def write_pairs_to_rte_format(pairs, outfile):
    """
    RTE format:
    index sentence1 sentence2 label
    0 No Weapons of Mass Destruction Found in Iraq Yet. Weapons of Mass Destruction Found in Iraq. not_entailment
    1 A place of sorrow, after Pope John Paul II died, became a place of celebration, as Roman Catholic faithful gathered in downtown Chicago to mark the installation of new Pope Benedict XVI. Pope Benedict XVI is the new leader of the Roman Catholic Church. entailment
    2 Herceptin was already approved to treat the sickest breast cancer patients, and the company said, Monday, it will discuss with federal regulators the possibility of prescribing the drug for more breast cancer patients. Herceptin can be used to treat breast cancer. entailment
    3 Judie Vivian, chief executive at ProMedica, a medical service company that helps sustain the 2-year-old Vietnam Heart Institute in Ho Chi Minh City (formerly Saigon), said that so far about 1,500 children have received treatment. The previous name of Ho Chi Minh City was Saigon. entailment
    4 A man is due in court later charged with the murder 26 years ago of a teenager whose case was the first to be featured on BBC One's Crimewatch. Colette Aram, 16, was walking to her boyfriend's house in Keyworth, Nottinghamshire, on 30 October 1983 when she disappeared. Her body was later found in a field close to her home. Paul Stewart Hutchinson, 50, has been charged with murder and is due before Nottingham magistrates later. Paul Stewart Hutchinson is accused of having stabbed a girl. not_entailment
    """
    logging.info(f'Writing {len(pairs)} pairs to {outfile} in RTE format.')

    with open(outfile, 'w') as file:
        tsv_writer = csv.writer(file, delimiter='\t')
        tsv_writer.writerow(['index', 'sentence1', 'sentence2'])
        for n, (pivot, post) in enumerate(pairs):
            tsv_writer.writerow([n, post.text, pivot.text])



def write_pairs_to_squad_format(pairs, posts, outfile):
    """
    Example of format SQuAD v2:

    {"version": "v2.0",
     "data": [{"title": "Normans", "paragraphs": [

    {"qas": [

    {"question": "In what country is Normandy
    located?", "id": "56ddde6b9a695914005b9628", "answers": [{"text": "France", "answer_start": 159}, {"text": "France",
    "answer_start": 159}, {"text": "France", "answer_start": 159}, {"text": "France", "answer_start": 159}],
    "is_impossible": false},

    {"question": "When were the Normans in Normandy?", "id": "56ddde6b9a695914005b9629",
    "answers": [{"text": "10th and 11th centuries", "answer_start": 94}, {"text": "in the 10th and 11th centuries",
    "answer_start": 87}, {"text": "10th and 11th centuries", "answer_start": 94}, {"text": "10th and 11th centuries",
    "answer_start": 94}], "is_impossible": false},

    {"question": "From which countries did the Norse originate?",
    "id": "56ddde6b9a695914005b962a", "answers": [{"text": "Denmark, Iceland and Norway", "answer_start": 256},
    {"text": "Denmark, Iceland and Norway", "answer_start": 256}, {"text": "Denmark, Iceland and Norway", "answer_start":
    256}, {"text": "Denmark, Iceland and Norway", "answer_start": 256}], "is_impossible": false}, {"question": "Who was
    the Norse leader?", "id": "56ddde6b9a695914005b962b", "answers": [{"text": "Rollo", "answer_start": 308},
    {"text": "Rollo", "answer_start": 308}, {"text": "Rollo", "answer_start": 308}, {"text": "Rollo", "answer_start":
    308}], "is_impossible": false},

    {"question": "What century did the Normans first gain their separate identity?",
    "id": "56ddde6b9a695914005b962c", "answers": [{"text": "10th century", "answer_start": 671}, {"text": "the first half
    of the 10th century", "answer_start": 649}, {"text": "10th", "answer_start": 671}, {"text": "10th", "answer_start":
    671}], "is_impossible": false},

    {"plausible_answers": [{"text": "Normans", "answer_start": 4}], "question": "Who gave
    their name to Normandy in the 1000's and 1100's", "id": "5ad39d53604f3c001a3fe8d1", "answers": [], "is_impossible":
    true},

    {"plausible_answers": [{"text": "Normandy", "answer_start": 137}], "question": "What is France a region of?",
    "id": "5ad39d53604f3c001a3fe8d2", "answers": [], "is_impossible": true},

    ...

    ], "context": "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and
    11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from
    \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear
    fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish
    and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West
    Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th
    century, and it continued to evolve over the succeeding centuries."}
    """

    logging.warning("Writing QA file not yet fully implemented.")
    return

    # logging.info(f'Writing {len(pairs)} pairs to {outfile} in SQUADv2 format.')

    data = []
    result = {"version": f"v0.1.{random.seed()}",
              "data": data}

    items = []
    for n, (question, pivot) in enumerate(pairs):
        # TODO Decide which metadata to save for our analysis, and which for Squad.
        item_info = {'question_id': question['id'],
         'question_text': question['text'],
         'pivot_id': pivot['id'],
         'pivot_text': pivot['text'],
         'user_post_id': pivot['user_post_id'],
         }
        items.append(item_info)

    items_df = pd.DataFrame(items)
    for user_post_id, sub_df in items_df.groupby('user_post_id'):
        # TODO: look up the post data.
        pass
        for item in sub_df.itertuples():
            # TODO: add each item to a squad unit
            pass
        # TODO: Write a finished squad unit
        pass






if __name__ == '__main__':
    main()
