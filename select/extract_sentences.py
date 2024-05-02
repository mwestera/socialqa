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
tokenizer (slow!) or regex (fast!). 

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
    "user_post_id": "blabla",
    "user_post_author_id": "blibli", 
    "subreddit_name": "jazzguitar",
    "user_post_created": "2021-01-27 17:13:42"
}

Example:
$ python extract_sentences.py collected/posts_conspiracy_top_anon.jsonl > collected/sentences_conspiracy.jsonl

"""

# TODO: filter on minimum sentence length    5 < len < 50


@click.command(help="Extract sentences from posts and their context. Prints each sentence as a json dictionary, storing "
                    "the text along with various meta-info.")
@click.argument("file", type=click.File('r'), default=sys.stdin)
@click.option("--use_spacy", help="Whether to use Spacy (slow); if not, uses simple regex (super fast).", type=bool, required=False, is_flag=True)
def main(file, use_spacy):

    logging.basicConfig(level=logging.INFO)

    nlp = spacy.load('en_core_web_sm') if use_spacy else None

    n_sentences = 0
    n_post = 0
    for n_post, line in enumerate(file):
        item = json.loads(line)
        for sentence in extract_sentences_from_post(item, nlp):
            n_sentences += 1
            print(json.dumps(sentence))

    logging.info(f'Extracted {n_sentences} sentences from {n_post} posts.')

    # TODO: This might be a good place to log some sentence-level stats like sentence length and num questions?


sentence_regex = re.compile(r'([A-Z][^.!?]*[.!?]+)', re.M)


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
