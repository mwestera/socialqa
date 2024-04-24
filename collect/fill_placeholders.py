import click
import sys
import json


"""
To limit reddit API calls, collect_user_posts.py optionally refrains from re-downloading data it has already gotten before.
To save Raspberry Pi memory, these are temporarily stored like 

{"type": "comment", "DUPLICATE": "k04gsuh", "id": "k04gsuh", "author_id": "..."}

This script lets us replace those lines by the actual data for these posts.

Example:

$ cat posts.jsonl | python fill_placeholders.py --reuse_posts older_posts.jsonl > posts_completed.jsonl
"""

@click.command(help="Fill in placeholders resulting from saving reddit api calls and raspi memory.")
@click.argument("posts", type=click.File('r'), default=sys.stdin)
@click.option("--reuse_posts", help="A .jsonl file to reuse already downloaded posts from.", type=click.File('r'), required=True, default=None)
def main(posts, reuse_posts):
    posts_to_reuse = {}
    for line in reuse_posts:
        post = json.loads(line)
        posts_to_reuse[post.get('id') or post.get('DUPLICATE')] = post  # two options, for backwards compatibility.

    for line in posts:
        post = json.loads(line)
        if "DUPLICATE" in post:
            print(json.dumps(posts_to_reuse[post[post.get('id') or post.get('DUPLICATE')]]))
        else:
            print(line.strip())


if __name__ == '__main__':
    main()
