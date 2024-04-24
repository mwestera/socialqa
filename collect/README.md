# Data collection 'pipeline'

Scripts to be run in this order (for the various arguments/options, see each respective file):

1. collect_users.py: to collect a list of recently active users from a subreddit
2. collect_user_posts.py: to collect a list of top posts from a list of users. 
3. (fill_placeholders.py: in case collect_user_posts.py was run with --reuse-posts option, based on an existing .jsonl file) 
4. anonymize.py: to anonymize a list of posts (posters, and users mentioned in posts).
5. clean_posts.py: to slightly clean the resulting post texts.