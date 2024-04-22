# Data collection 'pipeline'

Scripts to be run in this order (for the various arguments/options, see each respective file):

1. collect_users.py: to collect a list of recently active users from a subreddit
2. collect_user_posts.py: to collect a list of top posts from a list of users. 
3. anonymize.py: to anonymize a list of posts (posters, and users mentioned in posts).
4. clean_posts.py: to slightly clean the resulting post texts.
5. extract_sentences.py: to translate each post entry (inc. submission, parent, replies) into a list of sentences (potential questions/pivots).