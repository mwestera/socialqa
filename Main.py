import subprocess
import os
import json
import argparse
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
def main(post_file_name):
  print(f"Start running for {post_file_name}")
  # Load configuration
  with open('config.json', 'r') as config_file:
    config = json.load(config_file)

  dir = config['dir']
  autocorrect = config['autocorrect']
  html = config['html']
  n_qa =  config['n_qa'] #Max how many QA items per user."
  n_rte = config['n_rte'] #Max how many RTE items per user."

  # Files used
  post_file = f"{dir}/{post_file_name}"

  # Files created
  sentence_file = f"{dir}/sentences_{post_file_name}.jsonl"
  pairs_file_QA = f'pairs_qa.tsv'
  pairs_file_RTE = f'pairs_rte.tsv'
  embeddings_file = f"{sentence_file}_embs.csv"
  pairs_file_RTE_scores = f"{dir}/{pairs_file_RTE}_scores.tsv"
  pairs_file_QA_scores = f"{dir}/{pairs_file_QA}_scores.tsv"
  output_file  = f"{post_file_name}_scores_triplets.tsv"


  # This has been done already
#  if not os.path.exists(post_file):
    # Collect users
#    run_script('collect_users.py')

    # Collect posts
#    run_script('collect_user_posts.py')

    # Anonymize the data
#    run_script('anonymize.py')

  # Clean posts, and remove sentences not between 5 and 50 tokens
  print("clean posts....")
 # run_script(f'{dir}/clean_posts.py', post_file)

  # Cut posts into sentences
  print("Extract Sentences...") 
#  run_script(f'{dir}/extract_sentences.py',  f'{post_file}', f'{sentence_file}')

  # Create scores per sentence
#  run_script(f'{dir}/classify_sentences.py',  f'{sentence_file}')

  # Create embeddings
#  run_script(f'{dir}/embed_sentences.py', f'{sentence_file}')  # Writes to sentence_file + "_embs.csv"
  # Define the path to your JSONL file

  # Open the JSONL file and read the first 100 lines
  #with open(jsonl_file_path, 'r') as f:
  #    for i, line in enumerate(f):
  #        if i >= 100:
  #            break
  #        first_100_lines.append(json.loads(line))

  #with open(output_jsonl_file_path, 'w') as f:
  #  for line in first_100_lines:
  #      f.write(json.dumps(line) + '\n')

  # Select pairs
  run_script(f'{dir}/make_tasks.py',  'new.jsonl', f'{post_file}.jsonl')

  # Run QA
  run_script(f'{dir}/calculate_scores_gpu.py', f'{pairs_file_QA}', 'qa')

  # Run RTE
  run_script(f'{dir}/calculate_scores_gpu.py',f'{pairs_file_RTE}', 'rte')

  # Create triplets, and combine scores
  run_script(f'{dir}/create_triplets.py', f'--infile_ent {pairs_file_RTE_scores}', f'--infile_qa {pairs_file_QA_scores}', f'--outfile {output_file}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('--post_file', type=str, help='Path to the post file')
    args = parser.parse_args()
    main(args.post_file)
