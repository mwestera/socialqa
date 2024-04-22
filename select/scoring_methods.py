import pandas as pd


def get_pivot_score(sentence: pd.Series):

    # TODO Take into account upvotes etc?
    # TODO Also boost pivots from the 'origin' subreddit
    # TODO Take into account subjectivity and concreteness ratings

    text = sentence['text']

    if text.endswith('?'):
        return None
    if sentence['from_user']:
        return None

    score = 0

    sentence_from = sentence['from']
    tokens = text.split()

    if sentence_from in ['root_submission_title']:
        score = 5
    elif sentence_from in ['root_submission_text', 'parent_comment']:
        score = 4
    elif sentence_from in ['reply_comment']:
        score = 3

    # punish personal/subjective things:
    if 'I' in tokens:
        score -= 1
    if 'you' in tokens:
        score -= 0.5

    if text.endswith('!'):
        score -= 1

    # punish too short/too long pivots:
    if len(tokens) < 5:
        score -= 3
    elif len(tokens) < 10:
        score -= 2
    elif len(tokens) < 15:
        score -= 1
    elif len(tokens) > 35:
        score -= 1
    elif len(tokens) > 50:
        score -= 2
    elif len(tokens) > 70:
        score -= 3

    return score


def get_question_score(sentence: pd.Series):

    # TODO Take into account upvotes etc?
    # TODO Also boost questions from the 'origin' subreddit
    # TODO Take into account subjectivity and concreteness ratings

    text = sentence['text']
    tokens = text.split()

    if not text.endswith('?'):
        return None

    score = 0

    sentence_from = sentence['from']

    if sentence_from in ['user_submission_title']:
        score = 5
    elif sentence_from in ['user_submission_text']:
        score = 4.5
    elif sentence_from in ['user_comment']:
        score = 4
    else:
        if sentence_from in ['root_submission_title']:
            score = 3
        elif sentence_from in ['root_submission_text', 'parent_comment']:
            score = 2.5
        elif sentence_from in ['reply_comment']:
            score = 2

        if sentence['from_user']:
            score += 1

    # punish personal/subjective things:
    if 'I' in tokens:
        score -= .5
    if 'you' in tokens:
        score -= 1

    # punish too short/too long questions
    if len(tokens) < 5:
        score -= 2
    elif len(tokens) < 10:
        score -= 1
    elif len(tokens) > 25:
        score -= 1
    elif len(tokens) > 30:
        score -= 2

    return score


def get_post_score(post: pd.Series):

    # TODO Take into account upvotes etc?
    # TODO Also boost posts from the 'origin' subreddit
    # TODO Aren't the length limits going to cause data sparsity? We might want to cut longer posts up?
    # TODO Take into account subjectivity and concreteness ratings

    if post['type'] == 'submission':
        score = 5
    else:
        score = 4

    text = post['text']

    if len(text) < 40:
        return None
    elif len(text) < 100:
        score -= 1
    elif len(text) > 400:
        score -= 2
    elif len(text) > 800:
        return None


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
    relatedness = compute_relatedness(question.text, pivot.text)
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
    relatedness = compute_relatedness(pivot.text, post.text if post.type == 'submission' else post.body, asym=True)
    return pivot.pivot_score * post.post_score * relatedness


def compute_relatedness(text1: str, text2: str, asym: bool = False) -> float:
    """
    How semantically similar or textually related are the two strings?
    Currently computes intersection-over-union.
    If text1 is much shorter than text2, asym=True might be best.
    """
    # TODO: Choose a more sophisticated relatedness measure?
    a = set(text1.split())
    b = set(text2.split())
    iou = len(a & b) / len(a if asym else (a | b))
    return iou

