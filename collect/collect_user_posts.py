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
Collects submissions and comments of a user, TOP first, up to Reddit's API limit (1000 each).
Optionally save only submissions/comments from a given subreddit (but the others will add to the 1000 cap).
Posts are stored together with some parents and some replies.

Can take a single username, or a .jsonl file of users, as collected by collect_users.py.

Example:

$ python collect_user_posts.py collected/users_conspiracy.jsonl > collected/posts_conspiracy.jsonl
"""


# TODO: What if internet connection breaks?
# TODO: Make restarting easier: look up last user id and post id in the posts file.

@click.command(help="Collect Reddit posts (submissions and comments) from a user (optionally: in a specific subreddit); USERNAME is the name of the user to collect posts from, or a jsonl file containing redditors.")
@click.argument("username", type=str)   # TODO: Allow reading from stdin?
@click.option("--n_posts", help="The maximum number of posts", type=int, required=False, default=None)
@click.option("--n_parents", help="The number of parents of posts to download, for context", type=int, required=False, default=None)
@click.option("--subreddit", help="The name of the subreddit to collect posts from", type=str, required=False, default=None)
@click.option("--which", help="Whether to collect the 1000 newest or top submissions and comments", type=click.Choice(["new", "top"]), required=False, default=None)
@click.option("--reuse_posts", help="A .jsonl file to reuse already downloaded posts from.", type=click.Path(exists=True), required=False, default=None)
@click.option("--continue_posts", help="A .jsonl file from which to continue downloading posts (typically the same .jsonl file to which the results are piped).", type=click.Path(exists=True), required=False, default=None)
@click.option("--log", type=click.Path(dir_okay=False, exists=False), default=None)
def main(username, n_posts, n_parents, subreddit, which, reuse_posts, continue_posts, log):
    """
    Queries reddit for posts by the username (or .jsonl-file with users), printing the resulting dictionaries as json.
    """
    global skip_until

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

    if reuse_posts:
        with open(reuse_posts, 'r') as file:
            posts_to_reuse = set(json.loads(line)["id"] for line in file)
        logger.info(f"Found {len(posts_to_reuse)} posts to reuse, from {reuse_posts}.")
        # posts_to_reuse = {item["id"]: item for item in items}
    else:
        posts_to_reuse = None

    if continue_posts:
        with open(continue_posts, 'r') as file:
            for line in file:
                if line.strip():
                    last_line = line
        skip_until = json.loads(last_line)
    else:
        skip_until = None

    if username.endswith('.jsonl'):
        usernames = []
        with open(username, 'r') as namesfile:
            for line in namesfile:
                user_dict = json.loads(line)
                name = user_dict['name']
                if skip_until and skip_until['author_id'] == user_dict['id']:
                    usernames = []
                usernames.append(name)
    else:
        usernames = [username]

    logger.info(f"Will seek posts of {len(usernames)} usernames (starting from {usernames[0]}).")

    for username in usernames:
        try:
            for post in collect_user_posts(username, subreddit, n_posts, n_parents, logger=logger, reuse_posts=posts_to_reuse, which=which):
                print(json.dumps(post))
        except prawcore.exceptions.Forbidden as e:
            logging.warning(f'Skipping user {username} altogether: encountered Forbidden: {e}')


def collect_user_posts(username, subreddit=None, n_posts=None, n_parents=None, logger=logging, reuse_posts=None, which=None):
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

    if skip_until is None or skip_until['type'] == 'submission':
        for sub in collect_submissions(user, username, subreddit, which, n_posts, reuse_posts, logger):
            yield sub
    for comment in collect_comments(user, username, subreddit, which, n_posts, reuse_posts, logger, n_parents):
        yield comment


def collect_submissions(user, username, subreddit, which, n_posts, reuse_posts, logger):

    global skip_until

    count = 0
    if which == "top":
        submissions = user.submissions.top(limit=n_posts)
    elif which == "new":
        submissions = user.submissions.new(limit=n_posts)

    for n_submission, submission in enumerate(submissions):

        if (n_submission + 1) % 50 == 0:
            logger.info(f'Submissions found for {username}: {count} (now checking submission {n_submission})')
            logger.info('Taking a brief pause.')
            time.sleep(30)

        if skip_until and skip_until['id'] == submission.id:
            skip_until = None
            continue

        if skip_until:
            continue

        # if reuse_posts and (duplicate := reuse_posts.get(submission.id, False)):
        if reuse_posts and (submission.id in reuse_posts):
            logger.info(f'Reusing existing submission {submission.id}.')
            yield {"type": "submission", "DUPLICATE": submission.id, "id": submission.id, "author_id": user.id}
            count += 1
            # yield duplicate
            continue

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


def collect_comments(user, username, subreddit, which, n_posts, reuse_posts, logger, n_parents):

    global skip_until

    if which == "top":
        comments = user.comments.top(limit=n_posts)
    elif which == "new":
        comments = user.comments.new(limit=n_posts)

    count2 = 0

    for n_comment, comment in enumerate(comments):

        if (n_comment + 1) % 50 == 0:
            logger.info(f'Comments found for {username}: {count2} (now checking comment {n_comment})')
            logger.info('Taking a brief pause.')
            time.sleep(30)

        if skip_until and skip_until['id'] == comment.id:
            skip_until = None
            continue

        if skip_until:
            continue

        # if reuse_posts and (duplicate := reuse_posts.get(comment.id, False)):
        if reuse_posts and (comment.id in reuse_posts):
            logger.info(f'Reusing existing comment {comment.id}.')
            yield {"type": "comment", "DUPLICATE": comment.id, "id": comment.id, "author_id": user.id}
            count2 += 1
            # yield duplicate
            continue

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
            except prawcore.exceptions.Forbidden as e:
                logger.warning(f'comment forbidden? skipping [{e}]')
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
        'subreddit_id': comment.subreddit.name,
        'subreddit_name': comment.subreddit.display_name,
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
        'author_id': author_id,
        'subreddit_id': submission.subreddit.name,
        'subreddit_name': submission.subreddit.display_name,
        'created': utc_to_str(submission.created_utc),
        'parent': None,
        'replies': replies,
        **{key: getattr(submission, key) for key in _submission_attributes_to_save}
    }
    return submission_dict


if __name__ == '__main__':

    main()
