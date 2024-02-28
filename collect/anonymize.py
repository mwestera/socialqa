import re
import click
import sys
import os

"""
Example, to combine all collected posts into a single, anonymized file:

$ cat collected/posts_*.jsonl | python anonymize.py --map "collected/anonymap.csv" > "collected/posts_combined_anon.jsonl"
"""

re_username = re.compile(r'/u/[A-Za-z0-9_-]+')
re_author_id = re.compile(r'"author_id": "([A-Za-z0-9_-]+)"')

@click.command(help="Anonymize posts and store a mapping.")
@click.argument("posts", type=click.File('r'), default=sys.stdin)
@click.option("--map", help="Path to .csv file (existing or will be created)", type=click.Path(file_okay=True, dir_okay=False, writable=True), default=None)
def main(posts, map):
    """
    Given posts file (or stdin) in .jsonl format, replaces all user mentions in the text, and author_ids in the
    metadata, by somewhat anonymous IDs -- simply numbered user_N in order of occurrence.

    If a .csv file --map is given, mapping from author_ids to anonymized IDs is maintained (and reused) there.
    """
    if map and os.path.exists(map):
        with open(map) as file:
            mapping = dict(line.strip().split(',') for line in file)
    else:
        mapping = {}

    n_users = len(mapping)

    def anonymize(match):
        nonlocal n_users
        author_id = match.group(1)
        if anon_id := mapping.get(author_id, False):
            pass
        else:
            anon_id = f'user_{n_users}'
            n_users += 1
            mapping[author_id] = anon_id
            mappingfile.write(f'{author_id},{anon_id}\n')
        return f'"author_id": "{anon_id}"'

    with open(map, 'a') as mappingfile:
        for line in posts:
            line2 = re_username.sub("[user]", line)
            line3 = re_author_id.sub(anonymize, line2)
            line_anonymized = line3
            print(line_anonymized, end='')


if __name__ == '__main__':
    main()