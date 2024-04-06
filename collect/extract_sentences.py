import click
import sys
import json
import spacy
from typing import Iterator, Optional
import re
import logging
import termplotlib
import numpy as np
import itertools

"""
Extract sentences from posts and their context (submission, parents, replies), using either the spacy 
tokenizer (slow!) or regex (fast!). If input is not posts but sentences, will only recompute pivot and question scores. 

Prints each sentence as a json dictionary, storing the text along with various meta-info, like this:

{
    "id": "gl0be9s_2017-2080" <generally postid_startchar-endchar>,
    "text": <the sentence string>,
    "previous": <the preceding sentence text, or null>,
    "next": <the next sentence text, or null>,
    "num_comments": null,
    "upvote_ratio": null,
    "post_id": <id of the post containing the sentence>,
    "snippet_id": <same as above, or with suffix _T in case of submission titles>,
    "from": "reply_comment" <relation of the post from which this sentence came, to the user post>, 
    "from_user": false <whether the post from which this sentence came, was from the same author as the user post>,
    "score": 7,
    "user_post_id": "l69bg5",
    "user_post_author_id": "dxbdv3i", 
    "subreddit_name": "jazzguitar",
    "user_post_created": "2021-01-27 17:13:42"
}

Example:
$ python extract_sentences.py collected/sentences_conspiracy_top_anon.jsonl > collected/sentences_conspiracy.jsonl

Or to update only scores (input file already tokenized):

$ python extract_sentences.py collected/sentences_conspiracy.jsonl > collected/sentences_conspiracy2.jsonl
"""

@click.command(help="Extract sentences from posts and their context. Prints each sentence as a json dictionary, storing "
                    "the text along with various meta-info. If file is already sent-tokenized, only recomputes scores.")
@click.argument("file", type=click.File('r'), default=sys.stdin)
@click.option("--use_spacy", help="Whether to use Spacy (slow); if not, uses simple regex (super fast).", type=bool, required=False, is_flag=True)
def main(file, use_spacy):

    logging.basicConfig(level=logging.INFO)

    nlp = spacy.load('en_core_web_sm') if use_spacy else None

    first_line = next(file)
    first_item = json.loads(first_line)
    already_sentences = 'start' in first_item
    if already_sentences:
        logging.info('Input is already sentence-tokenized; only recomputing scores.')

    # for logging only
    pivot_scores = []
    question_scores = []
    n_lines = 0

    for line in itertools.chain([first_line], file):
        item = json.loads(line)
        n_lines += 1
        sentences = [item] if already_sentences else extract_sentences_from_post(item, nlp)
        for sentence in sentences:
            pivot_score = get_pivot_score(sentence)
            question_score = get_question_score(sentence)
            sentence['pivot_score'] = pivot_score
            sentence['question_score'] = question_score
            pivot_scores.append(pivot_score)
            question_scores.append(question_score)
            print(json.dumps(sentence))

    if already_sentences:
        logging.info(f'Recomputed scores for {len(pivot_scores)} sentences.')
    else:
        logging.info(f'Extracted {len(pivot_scores)} sentences from {n_lines} posts.')

    pivot_scores_nonnull = [s for s in pivot_scores if s > -99]
    question_scores_nonnull = [s for s in question_scores if s > -99]

    counts, bin_edges = np.histogram(pivot_scores_nonnull, bins=8)
    fig = termplotlib.figure()
    fig.hist(counts, bin_edges, force_ascii=False)
    logging.info(f'Pivot scores:\n{fig.get_string()}\nmin={min(pivot_scores_nonnull)}, max={max(pivot_scores_nonnull)},'
                 f'mean={sum(pivot_scores_nonnull) / len(pivot_scores_nonnull):.2f},'
                 f'N={len(pivot_scores_nonnull)} ({100*len(pivot_scores_nonnull) / len(pivot_scores):.2f}%)')

    counts, bin_edges = np.histogram(question_scores_nonnull, bins=8)
    fig = termplotlib.figure()
    fig.hist(counts, bin_edges, force_ascii=False)
    logging.info(f'Question scores:\n{fig.get_string()}\nmin={min(question_scores_nonnull)}, max={max(question_scores)},'
                 f'mean={sum(question_scores_nonnull) / len(question_scores_nonnull):.2f},'
                 f'N={len(question_scores_nonnull)} ({100*len(question_scores_nonnull) / len(question_scores):.2f}%)')


sentence_regex = re.compile(r'([A-Z][^.!?]*[.!?]+)', re.M)


def get_pivot_score(sentence):
    score = 0
    text = sentence['text']

    if text.endswith('?'):
        return -99
    if sentence['from_user']:
        return -99

    sentence_from = sentence['from']
    tokens = text.split()

    if sentence_from in ['root_submission_title']:
        score = 5
    elif sentence_from in ['root_submission_text', 'parent_comment']:
        score = 4
    elif sentence_from in ['reply_comment']:
        score = 3

    # punish personal/subjective things:
    if 'I' in tokens:
        score -= 1
    if 'you' in tokens:
        score -= 0.5

    if text.endswith('!'):
        score -= 1

    # punish too short/too long pivots:
    if len(tokens) < 5:
        score -= 3
    elif len(tokens) < 10:
        score -= 2
    elif len(tokens) < 15:
        score -= 1
    elif len(tokens) > 35:
        score -= 1
    elif len(tokens) > 50:
        score -= 2
    elif len(tokens) > 70:
        score -= 3

    return score


def get_question_score(sentence):
    score = 0

    text = sentence['text']
    tokens = text.split()

    if not text.endswith('?'):
        return -99

    sentence_from = sentence['from']

    if sentence_from in ['user_submission_title']:
        score = 5
    elif sentence_from in ['user_submission_text']:
        score = 4.5
    elif sentence_from in ['user_comment']:
        score = 4
    else:
        if sentence_from in ['root_submission_title']:
            score = 3
        elif sentence_from in ['root_submission_text', 'parent_comment']:
            score = 2.5
        elif sentence_from in ['reply_comment']:
            score = 2

        if sentence['from_user']:
            score += 1

    # punish personal/subjective things:
    if 'I' in tokens:
        score -= .5
    if 'you' in tokens:
        score -= 1

    # punish too short/too long questions
    if len(tokens) < 5:
        score -= 2
    elif len(tokens) < 10:
        score -= 1
    elif len(tokens) > 25:
        score -= 1
    elif len(tokens) > 30:
        score -= 2

    return score


def extract_sentences_from_post(user_post: dict, nlp: Optional[spacy.Language] = None) -> Iterator[dict]:
    post_info_to_save = {
        'user_post_id': user_post['id'],
        'user_post_author_id': user_post['author_id'],
        'subreddit_name': user_post['subreddit_name'],
        'user_post_created': user_post['created'],
    }

    for snippet in iter_context_snippets(user_post):
        snippet_info_to_save = {
            'num_comments': snippet.get('num_comments', None),
            'upvote_ratio': snippet.get('upvote_ratio', None),
            'post_id': snippet['from_id'],
            'snippet_id': snippet['id'],
            'from': snippet['from'],
            'from_user': snippet['from_user'],
            'score': snippet['score'],
        }

        if nlp:
            sentences = [(s.text, (s.start_char, s.end_char)) for s in nlp(snippet['text']).sents]
        else:
            sentences = [(s.group(), s.span()) for s in sentence_regex.finditer(snippet['text'])]
        sentences = [(None, None), *sentences, (None, None)]
        for (previous, _), (text, (start, end)), (next, _) in zip(sentences, sentences[1:], sentences[2:]):
            yield {
                'id': f'{snippet["id"]}_{start}-{end}',
                'start': start,
                'end': end,
                'text': text,
                'previous': previous,
                'next': next,
                **snippet_info_to_save,
                **post_info_to_save,
            }

def iter_context_snippets(user_post: dict) -> Iterator[dict]:
    """
    Iterate over snippets of context: (title and body of) user_post itself, and then its family members.
    Each snippet has keys:
    'from' (user_submission_title/text, user_comment, root_submission_title/text, parent_comment and reply_comment),
    'from_user' (True if snippet is from same user as user_post),
    'text'
    'id'
    """
    # TODO: Refactor this function (OOP?), lots of repetition.

    if user_post['type'] == 'submission':
        # Yield submission title and text
        shared = {
            'score': user_post['score'],
            'num_comments': user_post['num_comments'],
            'upvote_ratio': user_post['upvote_ratio'],
            'from_user': True,
            'from_id': user_post['name'],
        }
        yield {
            'from': 'user_submission_title',
            'text': user_post['title'],
            'id': user_post['name'] + '_T',
            **shared,
             }
        yield {
            'from': 'user_submission_text',
            'text': user_post['selftext'],
            'id': user_post['name'],
            **shared,
        }

    if user_post['type'] == 'comment':
        # Yield comment itself, then its submission title and text, then its parent (unless equal to the submission)
        yield {
            'from': 'comment',
            'from_user': True,
            'text': user_post['body'],
            'id': user_post['id'],
            'from_id': user_post['id'],
            'score': user_post['score'],
        }

        submission = user_post['submission']
        shared = {
            'score': submission['score'],
            'num_comments': submission['num_comments'],
            'upvote_ratio': submission['upvote_ratio'],
            'from_user': submission['author_id'] == user_post['author_id'],
            'from_id': submission['name'],
        }
        yield {
            'from': 'root_submission_title',
            'text': submission['title'],
            'id': submission['name'] + '_T',
            **shared,
        }
        yield {
            'from': 'root_submission_text',
            'text': submission['selftext'],
            'id': submission['name'],
            **shared,
        }

        if (parent := user_post['parent']) and parent['type'] == 'comment':
            yield {
                'from': 'parent_comment',
                'from_user': parent['author_id'] == user_post['author_id'],
                'text': parent['body'],
                'id': parent['id'],
                'from_id': parent['id'],
                'score': parent['score'],
            }

    for reply in user_post['replies']:
        yield {
            'from': 'reply_comment',
            'from_user': reply['author_id'] == user_post['author_id'],
            'text': reply['body'],
            'id': reply['id'],
            'from_id': reply['id'],
            'score': reply['score'],
        }


if __name__ == '__main__':
    main()
