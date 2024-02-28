import logging
import click
import json

import sys
import os
import traceback
import dotenv
import time
import datetime

from prawcore.exceptions import TooManyRequests
from praw import Reddit


"""
Collect usernames (optionally: created before a certain time, with a certain karma)
from a given subreddit.

Uses the Reddit API to browse through the recent comments and posts in a subreddit, printing all users as jsonl.

Example:
$ python collect_usernames.py conspiracy --created_before 2020/01/01 --min_comment_karma 1000 > collected/users_conspiracy.jsonl
"""


# TODO: Allow separate file for keeping track of users, to avoid duplicates across subreddits.


@click.command(help="Collect usernames (optionally: created before a certain time, with a certain karma) from a given subreddit.")
@click.argument("subreddit", type=str)
@click.option("--created_before", help="only accounts created before YYYY/MM/DD", type=str, required=False, default=None)
@click.option("--min_comment_karma", help="only accounts with at least this much comment karma", type=int, required=False, default=None)
@click.option("--n", help="The maximum number of unique usernames", required=False, type=int, default=None)
@click.option("--log", type=click.Path(dir_okay=False, exists=False), default=None)
def main(subreddit, created_before, min_comment_karma, n, log):
    """
    Uses the Reddit API to browse through the recent comments and posts in a subreddit, printing all users as jsonl.
    """

    dotenv.load_dotenv()

    logger = logging.getLogger('collect_users')
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

    if created_before is not None:
        created_before = datetime.datetime.strptime(created_before, "%Y/%m/%d")

    for user in find_users(subreddit, created_before, min_comment_karma, n, logger):
        print(user_to_json(user))


def find_users(subreddit, created_before=None, min_comment_karma=None, n=None, logger=logging):
    """
    Connects to reddit and iterates through new comments, yielding all users that meet the requirements.
    """

    user_ids = set()

    reddit = Reddit(
        client_id=os.environ.get('CLIENT_ID'),
        client_secret=os.environ.get('CLIENT_SECRET'),
        user_agent=os.environ.get('USER_AGENT'),
        ratelimit_seconds=600,
    )

    the_subreddit = reddit.subreddit(subreddit)

    for n_comment, comment in enumerate(the_subreddit.comments(limit=None)):

        if (n_comment + 1) % 100 == 0:
            logger.info(f'Unique usernames collected: {len(user_ids)} (now checking comment {n_comment})')
            logger.info('Taking a quick break.')
            time.sleep(60)

        if n and len(user_ids) >= n:
            break

        try:  # overly cautious? user.created_utc gave me an attributeerror once.
            user = comment.author

            if user is None:
                continue
            if created_before and datetime.datetime.utcfromtimestamp(user.created_utc) > created_before:
                continue
            if min_comment_karma and user.comment_karma < min_comment_karma:
                continue

            if user.id not in user_ids:
                yield user
                user_ids.add(user.id)

        except AttributeError as e:
            logger.info(f'AttributeError: {e}')
            continue
        except TooManyRequests as e:
            logging.warning(f'too many requests, sleeping some seconds [{e}]')
            time.sleep(2)

    logger.info(f'Done: {len(user_ids)} usernames collected.')


_user_attributes_to_save = ['id', 'name', 'comment_karma', 'link_karma']


def user_to_json(user):
    user_dict = {
        'created': utc_to_str(user.created_utc),
        **{key: getattr(user, key) for key in _user_attributes_to_save}
    }
    return json.dumps(user_dict)


def utc_to_str(utc_stamp):
    utc = datetime.datetime.utcfromtimestamp(utc_stamp)
    return utc.strftime('%Y-%m-%d %H:%M:%S')


if __name__ == '__main__':
    main()
