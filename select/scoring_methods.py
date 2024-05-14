import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
"""
{"id": "g8yledd_2136-2181", "start": 2136, "end": 2181, "text": "There was baggies of trail mix, still sealed.",
 "previous": "Bears don't have that much dexterity.", "next": "The mess was *very* human.", "num_comments": null, 
 "upvote_ratio": null, "post_id": "g8yledd", "snippet_id": "g8yledd", "from": "parent_comment",
   "from_user": false, "score": 988,
"context": "Maybe a bear or some other animal ripped through everything and the people whose camp it was were out hiking and hadn't found the mess yet?", "user_post_id": "g948cqa",
 "user_post_author_id": "user_0", "subreddit_name": "AskReddit", 
 "user_post_created": "2020-10-17 16:41:59",
 "subjectivity": 0.937221884727478, "concreteness": 2.585}
"""
def get_embeddings_file():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
        embeddings_file = config['dir'] + "/embeddings.csv"
    return embeddings_file

def get_weights():
    with open('weights.json', 'r') as file:
        weights = json.load(file)
    return weights

def calculate_question_score(sentence: pd.Series, scaler_votes: MinMaxScaler):
    """
    Calculate the score of a question sentence.
    """
    weights = get_weights()['question']
    vote_score = get_vote_score(sentence)
    vote_score_df = pd.DataFrame([vote_score], columns=['total_votes'])
    #vote_score = np.array(vote_score).reshape(-1, 1)  # Reshape data to fit scaler
    scaled_vote_score = scaler_votes.transform(vote_score_df)[0][0]
    score = 0
    text = sentence['text']
    if not text.endswith('?'):
        return None
    score += weights['origin_weight'] * get_origin_score(sentence)
    score += weights['vote_weight'] * scaled_vote_score
    score += weights['length_weight'] * get_length_score(sentence,mean_length = 55,std_dev = 15)  # The standard deviation of the Gaussian function)
    score -= weights['subjectivity_weight'] * sentence['subjectivity']
    score += weights['concreteness_weight'] * sentence['concreteness']
    return score

def calculate_pivot_score(sentence: pd.Series,scaler_votes: MinMaxScaler):
    """
    Calculate the score of a pivot sentence.
    """
    weights = get_weights()['pivot']
    vote_score = get_vote_score(sentence)
    vote_score_df = pd.DataFrame([vote_score], columns=['total_votes'])
    #vote_score = np.array(vote_score).reshape(-1, 1)  # Reshape data to fit scaler
    scaled_vote_score = scaler_votes.transform(vote_score_df)[0][0]
    score = 0
    text = sentence['text']
    if text.endswith('?') or sentence['from_user']:
        return None
    score += weights['origin_weight'] * get_origin_score(sentence)
    score += weights['vote_weight'] * scaled_vote_score
    score += weights['length_weight'] * get_length_score(sentence,mean_length = 55,std_dev = 15)
    score -= weights['subjectivity_weight'] * sentence['subjectivity']
    score += weights['concreteness_weight'] * sentence['concreteness']
    return score

def calculate_post_score(post: pd.Series, scaler_votes: MinMaxScaler):

    # TODO Take into account upvotes etc? / 
    # TODO Also boost posts from the 'origin' subreddit / Done, do this in pairs
    # TODO Aren't the length limits going to cause data sparsity? We might want to cut longer posts up? 
    # / Done, no we take top pairs so we do not remove any pairs, just reorder them
    # TODO Take into account subjectivity and concreteness ratings/ Done
    """
    Calculate the score of a post.
    """
    vote_score = get_vote_score(post)
    vote_score_df = pd.DataFrame([vote_score], columns=['total_votes'])
    #vote_score = np.array(vote_score).reshape(-1, 1)  # Reshape data to fit scaler
    scaled_vote_score = scaler_votes.transform(vote_score_df)[0][0]
    score = 0
    weights = get_weights()['post']
    score += weights['type_weight'] * (1 if post['type'] == 'submission' else 0)
    score += weights['vote_weight'] * scaled_vote_score
    score += weights['length_weight'] * get_length_score(post, mean_length=600, std_dev=150)
    #TODO calculate subjectivity on posts
    #score -= weights['subjectivity_weight'] * post['subjectivity']
    #score += weights['concreteness_weight'] * post['concreteness']
    return score

def calculate_qa_scores(question: pd.Series, pivot: pd.Series, asym: bool = False) -> float:
    """
    How semantically similar or textually related are the two strings?
    Currently computes intersection-over-union.
    If text1 is much shorter than text2, asym=True might be best.
    """
    weights= get_weights()['qa_pairs']
    # Same Subreddit
    score = 0
    score += weights['subreddit_weight'] * get_subreddit_score(question, pivot)
    score += weights['embeddings_weight'] * get_cosine_similarity(question, pivot)
    #score += weights['time_weight'] * get_time_score(question, pivot)
    return score
def calculate_ent_scores(pivot: dict, entailment: dict, asym: bool = False) -> float:
    """
    How semantically similar or textually related are the two strings?
    Currently computes intersection-over-union.
    If text1 is much shorter than text2, asym=True might be best.
    """
    weights= get_weights()['ent_pairs']
    # Same Subreddit
    score = 0
    score += weights['subreddit_weight'] * get_subreddit_score(pivot, entailment)
    score += weights['embeddings_weight'] * get_cosine_similarity(pivot, entailment)
    #score += weights['time_weight'] * get_time_score(pivot, entailment)
    return score

def get_time_score(post1, post2):
    # Time Difference
    time1 = datetime.strptime(post1['user_post_created'], "%Y-%m-%d %H:%M:%S")
    time2 = datetime.strptime(post2['user_post_created'], "%Y-%m-%d %H:%M:%S")
    time_diff = abs((time2 - time1).total_seconds() / 60)

    # Manually scale time_diff to [0, 1] based on a maximum of 1440 minutes (24 hours)
    # Maximum time difference is 30 days in minutes
    max_time_diff = 30 * 24 * 60
    time_diff_score = 1 - (time_diff / max_time_diff)
    score += time_diff_score
    return time_diff_score

def get_embedding(post_id):
    embeddings_file = get_embeddings_file()
    embeddings_df = pd.read_csv(embeddings_file)
    first_column_name = embeddings_df.columns[0]  # Get the name of the first column
    embedding_columns = embeddings_df.columns[1:]

    # Combine all embedding columns into a single one
    embeddings_df['embedding'] = embeddings_df[embedding_columns].values.tolist()
    embedding = embeddings_df.loc[embeddings_df[first_column_name] == post_id, 'embedding'].values[0]
    return embedding


def get_cosine_similarity(post1, post2):
    """
    Calculate the cosine similarity between two embeddings.
    
    Args:
    - post1 (np.ndarray): The first embedding.
    - post2 (np.ndarray): The second embedding.
    
    Returns:
    - float: The cosine similarity between the two embeddings.
    """
    post1_id = post1[1]
    post2_id = post2[1]
    
    # Assuming get_embedding() returns embeddings as NumPy arrays
    embedding1 = tf.constant(get_embedding(post1_id), dtype=tf.float32)
    embedding2 = tf.constant(get_embedding(post2_id), dtype=tf.float32)
    
    # Compute cosine similarity
    similarity = tf.reduce_sum(embedding1[:, tf.newaxis] * embedding2, axis=-1)
    similarity /= tf.norm(embedding1[:, tf.newaxis], axis=-1) * tf.norm(embedding2, axis=-1)
    
    # Extract the similarity value
    similarity = similarity.numpy()[0]  # Convert to NumPy array and extract the value
    
    return similarity

def get_subreddit_score(post1,post2):
    """
    Calculate the subreddit score, if same then 1.
    """

    subreddit_score = 1 if post1[18] == post2[18] else 0
    return subreddit_score

def get_scalers(all_sentences):
    """
    Get the MinMaxScaler objects for the number of votes.
    """
    scaler = MinMaxScaler(feature_range=(0,1))
    # Apply the function to the dataframe
    all_sentences['upvotes'] = all_sentences.apply(lambda row: get_number_of_votes(row['score'], row['upvote_ratio'])[0], axis=1)
    all_sentences['down_votes'] = all_sentences.apply(lambda row: get_number_of_votes(row['score'], row['upvote_ratio'])[1], axis=1)
    all_sentences['total_votes'] = all_sentences['upvotes'] + all_sentences['down_votes']  # Create new feature
    scaled_data = all_sentences[['total_votes']]  # Use new feature for scaling
    scaler.fit(scaled_data)
    return scaler

def get_number_of_votes(score, upvote_ratio):
    """
    Calculate the number of upvotes and downvotes based on score and upvote ratio.
    
    Args:
    - score (int): The score of the post (upvotes - downvotes).
    - upvote_ratio (float): The ratio of upvotes to total votes.
    
    Returns:
    - tuple: A tuple containing the number of upvotes and downvotes (upvotes, downvotes).
             Returns (None, None) if it's impossible to determine absolute values.
    """
    if score is None or upvote_ratio is None or np.isnan(score) or np.isnan(upvote_ratio):
        return (0,0)
    if upvote_ratio == 1:  # All votes are upvotes
        upvotes = score
        downvotes = 0
    elif upvote_ratio == 0:  # Edge case scenario, assuming no votes if ratio is 0
        upvotes = 0
        downvotes = 0
    elif upvote_ratio == 0.5:  # Equal number of upvotes and downvotes, but exact numbers are indeterminate
        return (0,0)  # Cannot determine absolute values from score and upvote_ratio alone
    else:
        # Calculate upvotes using the derived formula
        upvotes = int(round(score / (2 * upvote_ratio - 1)))
        # Calculate downvotes based on the score and upvotes
        downvotes = upvotes - score

    return (upvotes, downvotes)

def get_vote_score(sentence):
    if sentence['upvote_ratio'] is None or np.isnan(sentence['upvote_ratio']):
        return 0
    
    upvotes, downvotes = get_number_of_votes(sentence['score'], sentence['upvote_ratio'])
    votes  = upvotes + downvotes
    return votes

def get_length_score(sentence, mean_length=55, std_dev=15):
    text = sentence['text']
    tokens = text.split()
    length = len(tokens)
    
    # Parameters for the Gaussian function
    
    # Calculate the score using the Gaussian function
    score = np.exp(-((length - mean_length) ** 2) / (2 * std_dev ** 2))
    
    # Normalize the score to the range [-5, 0] (since the original function had this range)
    score = -5 * (1 - score)
    
    return score

def get_origin_score(sentence):
    sentence_from = sentence['from']
    if sentence_from in ['root_submission_title']:
        score = 0.8
    elif sentence_from in ['root_submission_text', 'parent_comment']:
        score = 0.5
    elif sentence_from in ['reply_comment']:
        score = 0.3
    else:
        score = 0
    return score

def filter_QA_pair(pair: tuple) -> bool:
    """
    Boolean indicating whether we should keep this pair.
    Currently only checks that the question precedes the pivot.
    """
    question, pivot = pair
    return question.user_post_created < pivot.user_post_created

def rank_QA_pair(pair: tuple) -> float:
    """
    This should give a high score for good question+pivot pairs.
    Current idea: the question_score should be high, the pivot_score, and they should be related.
    """
    question, pivot = pair
    relatedness = calculate_qa_scores(question, pivot)
    return question.question_score * pivot.pivot_score * relatedness

def filter_RTE_pair(pair: tuple) -> float:
    """
    Boolean indicating whether we should keep this pair.
    Currently only checks that the pivot precedes the post.
    """
    pivot, post = pair
    return pivot.user_post_created < post.created

def rank_RTE_pair(pair) -> float:
    """
    This should give a high score for good pivot+post pairs.
    Current idea: the pivot_score should be high, the post_score, and they should be related.
    """
    pivot, post = pair
    relatedness = calculate_ent_scores(pivot.text, post.text if post.type == 'submission' else post.body, asym=True)
    return pivot.pivot_score * post.post_score * relatedness



