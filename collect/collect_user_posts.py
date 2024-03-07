import logging
import click
import json

import os
import sys
import traceback
import dotenv
import time
import datetime

import functools

from praw.models import Comment, Submission, MoreComments
from praw import Reddit, exceptions
import prawcore.exceptions

"""
Collects submissions and comments of a user, newest first, up to Reddit's API limit (1000 each).
Optionally save only submissions/comments from a given subreddit (but the others will add to the 1000 cap).
Posts are stored together with some parents and some replies.

Can take a single username, or a .jsonl file of users, as collected by collect_users.py.

Example:

$ python collect_user_posts.py collected/users_conspiracy.jsonl > collected/posts_conspiracy.jsonl
"""


# TODO: What if internet connection breaks?

@click.command(help="Collect Reddit posts (submissions and comments) from a user (optionally: in a specific subreddit); USERNAME is the name of the user to collect posts from, or a jsonl file containing redditors.")
@click.argument("username", type=str)
@click.option("--n_posts", help="The maximum number of posts", type=int, required=False, default=None)
@click.option("--n_parents", help="The number of parents of posts to download, for context", type=int, required=False, default=None)
@click.option("--subreddit", help="The name of the subreddit to collect posts from", type=str, required=False, default=None)
@click.option("--log", type=click.Path(dir_okay=False, exists=False), default=None)
def main(username, n_posts, n_parents, subreddit, log):
    """
    Queries reddit for posts by the username (or .jsonl-file with users), printing the resulting dictionaries as json.
    """

    dotenv.load_dotenv()

    logger = logging.getLogger('collect_user_posts')
    logging.basicConfig(level=logging.INFO)

    if log:
        handler = logging.FileHandler(log, mode='a')
        handler.setFormatter(logging.Formatter(fmt="%(levelname)s %(asctime)s - %(message)s", datefmt="%Y/%m/%d %H:%M:%S"))
        logger.addHandler(handler)
        handler.setLevel(logging.DEBUG)
        def log_except_hook(*exc_info):
            text = "".join(traceback.format_exception(*exc_info))
            logger.critical("Unhandled exception: %s", text)
        sys.excepthook = log_except_hook

    logger.info(' '.join(sys.argv))
    logger.info(datetime.datetime.now().__str__())

    if username.endswith('.jsonl'):
        usernames = []
        with open(username, 'r') as namesfile:
            for line in namesfile:
                user_dict = json.loads(line)
                usernames.append(user_dict['name'])
    else:
        usernames = [username]

    logger.info(f"Will seek posts of {len(usernames)} usernames.")

    for username in usernames:
        for post in collect_user_posts(username, subreddit, n_posts, n_parents, logger=logger):
            print(json.dumps(post))


def collect_user_posts(username, subreddit=None, n_posts=None, n_parents=None, logger=logging):
    """
    First collects submissions, then posts, yielding dictionaries one at a time.
    """

    reddit = Reddit(
        client_id=os.environ.get('CLIENT_ID'),
        client_secret=os.environ.get('CLIENT_SECRET'),
        user_agent=os.environ.get('USER_AGENT'),
        ratelimit_seconds=600,
    )

    if subreddit:
        subreddit = reddit.subreddit(subreddit)

    user = reddit.redditor(username)

    count = 0

    for n_submission, submission in enumerate(user.submissions.new(limit=n_posts)):

        if (n_submission + 1) % 50 == 0:
            logger.info(f'Submissions found for {username}: {count} (now checking submission {n_submission})')
            logger.info('Taking a brief pause.')
            time.sleep(30)

        if subreddit and subreddit != submission.subreddit:
            continue
        if not submission.selftext or submission.selftext == '[removed]':
            continue

        submission_dict = submission_to_dict(submission, logger=logger)
        if submission_dict:
            yield submission_dict
        else:
            logger.info(f'Couldn\'t turn submission {submission} to dict.')
        count += 1

    count2 = 0

    for n_comment, comment in enumerate(user.comments.new(limit=None)):

        if (n_comment + 1) % 50 == 0:
            logger.info(f'Comments found for {username}: {count2} (now checking comment {n_comment})')
            logger.info('Taking a brief pause.')
            time.sleep(30)

        if subreddit and subreddit != comment.subreddit:
            continue
        if comment.body == '[removed]':
            continue

        comment_dict = comment_to_dict(comment, n_parents=n_parents, logger=logger)
        if comment_dict:
            yield comment_dict
        else:
            logger.info(f'Couldn\'t turn comment {comment} to dict.')

        count2 += 1


def utc_to_str(utc_stamp):
    utc = datetime.datetime.utcfromtimestamp(utc_stamp)
    return utc.strftime('%Y-%m-%d %H:%M:%S')


_comment_attributes_to_save = ['id', 'body', 'link_id', 'parent_id', 'is_submitter', 'score']
_submission_attributes_to_save = ['id', 'name', 'num_comments', 'score', 'selftext', 'title', 'upvote_ratio', 'url']


def just_try(func):
    @functools.wraps(func)
    def inner_func(*args, logger=logging, **kwargs):
        n_attempts = 0
        while n_attempts < 5:
            try:
                result = func(*args, logger=logger, **kwargs)
            except prawcore.exceptions.TooManyRequests as e:
                logger.warning(f'too many requests, sleeping 10 seconds [{e}]')
                n_attempts += 1
                time.sleep(10)
            except prawcore.exceptions.RequestException as e:
                logger.warning(f'no internet connection maybe? waiting 30 s [{e}]')
                time.sleep(30)
            except prawcore.exceptions.ServerError as e:
                logger.warning(f'no internet connection maybe? waiting 60 s [{e}]')
                time.sleep(60)
            except exceptions.ClientException as e:
                logger.warning(f'something wrong with the comment? skipping [{e}]')
                break
            else:
                return result
        return None
    return inner_func


@just_try
def comment_to_dict(comment, n_parents=None, logger=logging, with_replies=True, with_submission=True):
    author_id = None
    n_attempts = 0
    while author_id is None and n_attempts < 3:
        try:
            author_id = getattr(comment.author, 'id', None)
        except prawcore.exceptions.NotFound as e:
            logger.warning(f'author NotFound error (attempt {n_attempts}) [{e}]')
            time.sleep(1)
            n_attempts += 1
        else:
            break
    submission = submission_to_dict(comment.submission, logger=logger, with_replies=False) if with_submission else None
    if with_replies:
        replies = [comment_to_dict(c, 0, logger=logger, with_replies=False, with_submission=False) for c in comment.refresh().replies if not isinstance(c, MoreComments)]
        replies = [x for x in replies if x is not None]
    else:
        replies = None
    parent = None
    if n_parents:
        n_attempts = 0
        if 'parent' in dir(comment):
            while parent is None and n_attempts < 3:
                try:
                    parent = comment.parent()
                except prawcore.exceptions.ServerError as e:
                    logger.warning(f'ServerError (attempt {n_attempts}) [{e}]')
                    time.sleep(1)
                    n_attempts += 1
                else:
                    parent = comment_to_dict(parent, (n_parents - 1), with_replies=False) if isinstance(parent, Comment) else submission if isinstance(parent, Submission) else None
                    break
    comment_dict = {
        'type': 'comment',
        'author_id': author_id,
        'subreddit': comment.subreddit_id,
        'created': utc_to_str(comment.created_utc),
        'submission': submission,
        'replies': replies,
        **{key: getattr(comment, key, None) for key in _comment_attributes_to_save},
        'parent': parent,
    }
    return comment_dict


@just_try
def submission_to_dict(submission, logger=logging, with_replies=True):
    n_attempts = 0
    author_id = None
    while author_id is None and n_attempts < 5:
        try:
            author_id = getattr(submission.author, 'id', None)
        except prawcore.exceptions.NotFound as e:
            logger.warning(f'author NotFound error (attempt {n_attempts}) [{e}]')
            time.sleep(1)
            n_attempts += 1
        else:
            break

    if with_replies:
        replies = [comment_to_dict(c, 0, logger=logger, with_replies=False, with_submission=False) for c in submission.comments if not isinstance(c, MoreComments)]
        replies = [x for x in replies if x is not None]
    else:
        replies = None

    submission_dict = {
        'type': 'submission',
        'subreddit_id': submission.subreddit.id,
        'author_id': author_id,
        'subreddit': submission.subreddit.name,
        'created': utc_to_str(submission.created_utc),
        'parent': None,
        'replies': replies,
        **{key: getattr(submission, key) for key in _submission_attributes_to_save}
    }
    return submission_dict


if __name__ == '__main__':

    main()
