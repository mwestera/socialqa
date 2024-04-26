import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
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
def sensitivity_parameters():
    return {
        # TODO: find double counted features, such as submission have more votes and larger length
        'question': {
            # Weights for the different features, sentences
            'origin_weight': 1,
            'vote_weight': 1,
            'length_weight': 1,
            'subjectivity_weight': 1,
            'concreteness_weight': 0.2,
        },
        'pivot': {
            'origin_weight': 1,
            'vote_weight': 1,
            'length_weight': 1,
            'subjectivity_weight': 1,
            'concreteness_weight': 0.2,
        },
        # Weights for the different features, posts
        'post': {
            'type_weight': 1,
            'length_weight': 1,
            'vote_weight': 1,
            'subjectivity_weight': 1,
            'concreteness_weight': 0.2,
        },
        # Weights for the different features, pairs
        'qa_pairs': {
            'subreddit_weight': 1,
            'iou_weight': 1,
            'time_weight': 1,
        },
        'ent_pairs':{
            'subreddit_weight': 1,
            'iou_weight': 1,
            'time_weight': 1,
        }
    }

def calculate_question_score(sentence: pd.Series, scaler_votes: MinMaxScaler):
    weights = sensitivity_parameters()['question']

    score = 0
    text = sentence['text']
    if not text.endswith('?'):
        return None
    score += weights['origin_weight'] * get_origin_score(sentence)
    score += weights['vote_weight'] * scaler_votes(get_vote_score(sentence))
    score += weights['length_weight'] * get_length_score(sentence,mean_length = 55,std_dev = 15)  # The standard deviation of the Gaussian function)
    score -= weights['subjectivity_weight'] * sentence['subjectivity']
    score += weights['concreteness_weight'] * sentence['concreteness']
    return score

def calculate_pivot_score(sentence: pd.Series,scaler_votes: MinMaxScaler):
    weights = sensitivity_parameters()['pivot']

    score = 0
    text = sentence['text']
    if text.endswith('?') or sentence['from_user']:
        return None
    score += weights['origin_weight'] * get_origin_score(sentence)
    score += weights['subreddit_weight'] * get_origin_score(sentence)
    score += weights['vote_weight'] * scaler_votes(get_vote_score(sentence))
    score += weights['length_weight'] * get_length_score(sentence,mean_length = 55,std_dev = 15)
    score -= weights['subjectivity_weight'] * sentence['subjectivity']
    score += weights['concreteness_weight'] * sentence['concreteness']
    return score

def calculate_post_score(post: pd.Series):

    # TODO Take into account upvotes etc? / 
    # TODO Also boost posts from the 'origin' subreddit / Done, do this in pairs
    # TODO Aren't the length limits going to cause data sparsity? We might want to cut longer posts up? 
    # / Done, no we take top pairs so we do not remove any pairs, just reorder them
    # TODO Take into account subjectivity and concreteness ratings/ Done
    score = 0
    weights = sensitivity_parameters()['post']
    score += weights['type_weight'] * (1 if post['type'] == 'submission' else 0)
    score += weights['vote_weight'] * get_vote_score(post)
    score += weights['length_weight'] * get_length_score(post, mean_length=600, std_dev=150)
    score -= weights['subjectivity_weight'] * post['subjectivity']
    score += weights['concreteness_weight'] * post['concreteness']
    return score

def calculate_qa_scores(question: dict, pivot: dict, asym: bool = False) -> float:
    """
    How semantically similar or textually related are the two strings?
    Currently computes intersection-over-union.
    If text1 is much shorter than text2, asym=True might be best.
    """
    weights= sensitivity_parameters()['qa_pairs']
    # Same Subreddit
    score = 0
    score += weights['subreddit_weight'] * get_subreddit_score(question, pivot)
    score += weights['iou_weight'] * get_iou_score(question, pivot, asym)
    score += weights['time_weight'] * get_time_score(question, pivot)
    return score
def calculate_ent_scores(pivot: dict, entailment: dict, asym: bool = False) -> float:
    """
    How semantically similar or textually related are the two strings?
    Currently computes intersection-over-union.
    If text1 is much shorter than text2, asym=True might be best.
    """
    weights= sensitivity_parameters()['ent_pairs']
    # Same Subreddit
    score = 0
    score += weights['subreddit_weight'] * get_subreddit_score(pivot, entailment)
    score += weights['iou_weight'] * get_iou_score(pivot, entailment, asym)
    score += weights['time_weight'] * get_time_score(pivot, entailment)
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

def get_iou_score(post1,post2, asym):
    text1 = post1['text']
    text2 = post2['text']

    # IOU
    a = set(text1.split())
    b = set(text2.split())
    iou = len(a & b) / len(a if asym else (a | b))
    score = iou
    return score

def get_subreddit_score(post1,post2):
    subreddit_score = 1 if post1['subreddit_name'] == post2['subreddit_name'] else 0
    score += subreddit_score
def get_scalers(all_sentences):
    scaler = MinMaxScaler(0,1)
    scaled_data = all_sentences["votes"].apply(get_number_of_votes).to_frame()
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
    if upvote_ratio == 1:  # All votes are upvotes
        upvotes = score
        downvotes = 0
    elif upvote_ratio == 0:  # Edge case scenario, assuming no votes if ratio is 0
        upvotes = 0
        downvotes = 0
    elif upvote_ratio == 0.5:  # Equal number of upvotes and downvotes, but exact numbers are indeterminate
        return None, None  # Cannot determine absolute values from score and upvote_ratio alone
    else:
        # Calculate upvotes using the derived formula
        upvotes = int(round(score / (2 * upvote_ratio - 1)))
        # Calculate downvotes based on the score and upvotes
        downvotes = upvotes - score

    return (upvotes, downvotes)


def get_vote_score(sentence):
    if sentence['upvote_ratio'] is not None:
        upvotes, downvotes = get_number_of_votes(sentence['score'], sentence['upvote_ratio'])
        votes  = upvotes + downvotes
        score += votes/100
        return votes
    return 0

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



