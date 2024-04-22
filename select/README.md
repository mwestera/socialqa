# Data selection

Starting from the (cleaned, anonymized) posts (see `collect` folder):

1. extract_sentences.py: to translate each post entry (inc. submission, parent, replies) into a list of sentences (potential questions/pivots).
2. make_tasks.py: to combine sentences and the original posts into a QA and entailment task.

Crucial selection criteria are centralized in `scoring_methods.py`.