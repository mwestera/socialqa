import subprocess
import os
import json
import argparse
import logging
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
    $ python Main.py --post_file_name

    or 

    $ python3 Main.py --post_filen_name (posts_conservative_v1)
"""

def run_script(script_name, *args):
    logging.info("running ...", script_name)

    # python3 or just python
    subprocess.run(['python3', script_name, *args], check=True)

def main(post_file_name):
  logging.basicConfig(level=logging.INFO)
  logging.info(f"Start running for {post_file_name}")
  
  # Load configuration
  with open('config.json', 'r') as config_file:
    config = json.load(config_file)

  dir = config['dir']
  autocorrect = config['autocorrect']
  html = config['html']
  n_qa =  config['n_qa'] #Max how many QA items per user."
  n_rte = config['n_rte'] #Max how many RTE items per user."


  # Files created and used in pipeline
  post_file_jsonl = f"{dir}/{post_file_name}.jsonl"
  sentence_file = f"{dir}/sentences_{post_file_name}.jsonl"
  pairs_file_QA = f'pairs_qa_{post_file_name}.tsv'
  pairs_file_RTE = f'pairs_rte_{post_file_name}.tsv'
  embeddings_file = f"{sentence_file}_embs.csv"
  pairs_file_RTE_scores = f"{dir}/{pairs_file_RTE}_scores.tsv"
  pairs_file_QA_scores = f"{dir}/{pairs_file_QA}_scores.tsv"
  similarities_qa_file = f"{post_file_name}_qa_similarities.tsv"  
  similarities_nc_rte_file = f"{post_file_name}_nc_rte_similarities.tsv"  
  similarities_c_rte_file = f"{post_file_name}_c_rte_similarities.tsv"  

  logging.info(f"Post file loaded for analyses {post_file_jsonl}")

  # This has been done already
#  if not os.path.exists(post_file):
    # Collect users
#    run_script('collect_users.py')

    # Collect posts
#    run_script('collect_user_posts.py')

    # Anonymize the data
#    run_script('anonymize.py')

  # Clean posts, and remove sentences not between 5 and 50 tokens
    # Clean posts, and remove sentences not between 5 and 50 tokens
  logging.info("Starting clean posts...")
  run_script(f'{dir}/clean_posts.py', post_file_jsonl)


  # Cut posts into sentences
  logging.info("Starting Extract Sentences...")
  run_script(f'{dir}/extract_sentences.py', f'{post_file_jsonl}', f'{sentence_file}')

  # Create scores per sentence
  logging.info("Starting Create Scores per Sentence...")
    run_script(f'{dir}/classify_sentences.py', f'{sentence_file}')

  # Create embeddings
  logging.info("Starting Create Embeddings...")
  run_script(f'{dir}/embed_sentences.py', f'{sentence_file}', f'{post_file_name}')  # Writes to sentence_file + "_embs.csv"

  # Create  embeddings with context
  logging.info("Starting Collect Contextual Embeddings...")
  run_script(f'{dir}/collect_cont_embeddings.py', f'{sentence_file}', f'{post_file_jsonl}', f'{dir}/{post_file_name}_posts_embeddings.tsv', [1,11,12])

  # Select pairs
  logging.info("Starting Select Pairs...")
  run_script(f'{dir}/make_tasks.py', f'{sentence_file}', f'{post_file_jsonl}', f'{post_file_name}')

  # Run QA
  logging.info("Processing QA pairs")
  run_script(f'{dir}/calculate_scores_gpu.py', f'{pairs_file_QA}', 'qa', f'{similarities_qa_file}', '0.1')

  # Run RTE
  logging.info("Processing RTE pairs")
  run_script(f'{dir}/calculate_scores_gpu.py', f'{pairs_file_RTE}', 'rte', f'{post_file_jsonl}', f'{similarities_nc_rte_file}', f'{similarities_c_rte_file}', '0.1')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('--post_file', type=str, help='Path to the post file')
    args = parser.parse_args()
    main(args.post_file)
