import subprocess
import os
import subprocess
import json
import json
import subprocess
import os
"""
The pipeline is a series of scripts that are run in sequence. The scripts are run using subprocess.run
and the output of each script is checked for errors. The pipeline is defined in the Main.py script.

The pipeline consists of the following steps:
- Collect users
- Collect posts
- Anonymize the data
- Clean posts
- Cut posts into sentences
- Create scores per sentence
- Create embeddings
- Select pairs
- Run QA
- Run RTE
- Create triplets, and combine scores

The pipeline is defined in the Main.py script. The configuration is loaded from the config.json file.
weights.json is used to store the weights of the scoring method used in the pipeline.
Example:
    $ python Main.py

    or 

    $ python3 Main.py
"""

def run_script(script_name, *args):
    print("running ...", script_name)

    # python3 or just python
    subprocess.run(['python3', script_name, *args], check=True)

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

dir = config['dir']
autocorrect = config['autocorrect']
html = config['html']
post_file_name = config['post_file_name']
n_qa =  config['n_qa'] #Max how many QA items per user."
n_rte = config['n_rte'] #Max how many RTE items per user."

# Files used
post_file = f"{dir}/{post_file_name}.jsonl"

# Files created
sentence_file = f"{dir}/sentences_{post_file_name}.jsonl"
pairs_file_QA = f'{sentence_file}_pairs_qa.tsv'
pairs_file_RTE = f'{sentence_file}_pairs_rte.tsv'
embeddings_file = f"{sentence_file}_embs.csv"
pairs_file_RTE_scores = f"{dir}/{pairs_file_RTE}_scores.tsv"
pairs_file_QA_scores = f"{dir}/{pairs_file_QA}_scores.tsv"
output_file  = f"{post_file_name}_scores_triplets.tsv"

# This has been done already
if not os.path.exists(post_file):
    # Collect users
    run_script('collect_users.py')

    # Collect posts
    run_script('collect_user_posts.py')

    # Anonymize the data
    run_script('anonymize.py')

# Clean posts, and remove sentences not between 5 and 50 tokens
#run_script(f'{dir}/clean_posts.py', post_file)

# Cut posts into sentences
#run_script(f'{dir}/extract_sentences.py',  f'{post_file}', f'{sentence_file}')

# Ceate scores per sentence
#run_script(f'{dir}/classify_sentences.py',  f'{sentence_file}')

# Create embeddings
#run_script(f'{dir}/embed_sentences.py', f'{sentence_file}')  # Writes to sentence_file + "_embs.csv"

# Select pairs
run_script(f'{dir}/make_tasks.py',  f'{sentence_file}', f'{post_file}')

# Run QA
run_script(f'{dir}/calculate_scores_cpu.py {pairs_file_QA} --qa True')

# Run RTE
run_script(f'{dir}/calculate_scores_cpu.py {pairs_file_RTE} --qa False')

# Create triplets, and combine scores
run_script(f'{dir}/create_triplets.py --infile_ent {pairs_file_RTE_scores} --infile_qa {pairs_file_QA_scores} --outfile {output_file}')


