import pandas as pd
import numpy as np
import pymc as pm
import random
from matplotlib import pyplot as plt
import seaborn as sns
import arviz as az

import cloudpickle
import functools
import logging


def cached_to_disk(cache):

    def decorator(func):

        logger = logging.getLogger('pymc')
        @functools.wraps(func)
        def inner(*args, **kwargs):
            if cache:
                try:
                    with open(cache, 'rb') as buff:
                        data = cloudpickle.load(buff)
                    logger.info(f'Loaded cached results from {cache}')
                    return data
                except FileNotFoundError:
                    pass
            data = func(*args, **kwargs)
            with open(cache, 'wb') as buff:
                cloudpickle.dump(data, buff)
                logger.info(f'Saved results to cache: {cache}')
            return data

        return inner

    return decorator



def load_real_data(path_qa, path_rte):

    rte = pd.read_csv(path_qa, sep='\t')
    qa = pd.read_csv(path_rte, sep='\t')

    logging.warning('User ids not yet included in real data.')
    users = 'ABCDEFGHIJKLMN'

    rte['user_id'] = [random.choice(users) for _ in range(len(rte))]
    qa['user_id'] = [random.choice(users) for _ in range(len(qa))]
    rte['after_pivot'] = [random.choice([True, False]) for _ in range(len(rte))]

    return rte, qa


def load_toy_data(n_qa_items, n_rte_items, n_users, n_pivots, effect_size):
    # np.random.seed(42)

    user_ids = list(range(n_users))
    pivot_ids = list(range(n_pivots))

    qa_pivots = np.random.choice(pivot_ids, size=n_qa_items)

    qa_baseline_scores_per_pivot = pd.Series(dict(zip(pivot_ids, np.random.uniform(0, .8, size=n_pivots))))

    qa = pd.DataFrame({
        'user_id': np.random.choice(user_ids, size=n_qa_items),
        'pivot_id': qa_pivots,
        'score': qa_baseline_scores_per_pivot[qa_pivots] * np.random.uniform(0, 1, size=n_qa_items),
    })

    qa_scores = qa.groupby('pivot_id').agg({'score': 'mean'})

    rte_pivots = np.random.choice(pivot_ids, size=n_rte_items)
    after_pivot = np.random.choice([0, 1], size=n_rte_items)
    rte = pd.DataFrame({
        'user_id': np.random.choice(user_ids, size=n_rte_items),
        'pivot_id': rte_pivots,
        'after_pivot': after_pivot.astype(bool),
        'score': np.minimum(1, np.random.uniform(0, 1, size=n_rte_items) + (after_pivot * effect_size * qa_scores.iloc[rte_pivots].values.squeeze())),
    })

    return qa, rte


def prob_at_least_one(probs):
    """
    Given np array with (independent) event probabilities, computes the probability that at least one event
    occurred (i.e., 1 minus the probability that NONE occurred).
    """
    return 1 - (1 - probs).prod()


def combine_qa_and_rte_data(qa, rte):
    """
    Creates a single dataframe containing information about entailment rate (rte_count, rte_total, rte_rate),
    and answerhood probability (qa_prob).
    """
    rte_counts = rte.groupby(['user_id', 'pivot_id', 'after_pivot']).agg({'score': ['sum', 'count', 'mean']}).reset_index()
    rte_counts.columns = ['user_id', 'pivot_id', 'after_pivot', 'rte_count', 'rte_total', 'rte_rate']

    qa_probs = qa.groupby(['user_id', 'pivot_id']).agg({'score': prob_at_least_one}).reset_index()
    qa_probs.columns = ['user_id', 'pivot_id', 'qa_prob']

    combined = pd.merge(rte_counts, qa_probs, on=['user_id', 'pivot_id'], how='left')
    # This merge can result in nan-values (e.g., pivots with only an rta_rate before, not after;
    # pivots with no qa score);
    # It also (correctly) duplicates the qa-rates for the before and after rows (if present).

    combined_filtered = (combined
                         .groupby(['user_id', 'pivot_id'])
                         .filter(lambda x: len(x['after_pivot'].unique()) == 2)
                         .dropna())

    return combined_filtered


def init_mixed_effects_Poisson_model(data):
    """
    Bayesian mixed-effects Poisson regression model
    """
    with pm.Model() as mixed_effects_model:

        # intercepts:
        intercept = pm.Normal('intercept', mu=0, sigma=10)
        pivot_intercept = pm.Normal('pivot_intercept', mu=0, sigma=1, shape=len(np.unique(data['pivot_id'])))
        intercept_term = intercept + pivot_intercept[data['pivot_id']]

        # effects:
        fixed_effect = pm.Normal('fixed_effect', mu=0, sigma=1)
        random_effect_user = pm.Normal('random_effect_user', mu=0, sigma=1, shape=len(np.unique(data['user_id'])))

        # random_effect_pivot = pm.Normal('random_effect_pivot', mu=0, sigma=1, shape=len(np.unique(data['pivot_id'])))
        #    <-- this won't work, since there is only one datum per pivot.

        effect = (fixed_effect + random_effect_user[data['user_id']]) * data['qa_prob']

        offset = np.log(data['rte_total']) # because we predict rates, not counts
        linear_predictor = intercept_term + effect * data['after_pivot'] + offset

        rte_rate_mu = pm.math.exp(linear_predictor)
        likelihood = pm.Poisson('likelihood', mu=rte_rate_mu, observed=data['rte_count'])

    return mixed_effects_model, ['intercept', 'fixed_effect']


@cached_to_disk('poisson.cache')
def do_sampling(model, **kwargs):
    with model:
        trace = pm.sample(**kwargs)
        posterior = pm.sample_posterior_predictive(trace=trace)
        # linear_prior = pm.sample_prior_predictive()   # to inspect if the prior even makes sense?
    return trace


def plot_data(data):
    plt.figure(figsize=(12, 8))
    plt.title('Data: entailment rate before and after pivot, depending on question probability')
    ax = sns.regplot(data.loc[~data['after_pivot']], x='qa_prob', y='rte_rate', scatter=True, label='before pivot')
    sns.regplot(data.loc[data['after_pivot']], x='qa_prob', y='rte_rate', scatter=True, ax=ax, label='after pivot')
    plt.legend(title='For user posts that occur:')
    plt.xlabel('Probability that pivot answers at least one prior question')
    plt.ylabel('Rate of user posts entailing the pivot')
    # plt.tight_layout()


def plot_posterior(df, trace):
    """
    Plot the posterior predictions as well as a posterior trend line.
    Based on https://www.pymc.io/projects/examples/en/latest/time_series/Air_passengers-Prophet_with_Bayesian_workflow.html#part-1-linear-trend .
    """

    mean_pivot_intercept = trace.posterior['pivot_intercept'].mean(axis=-1).values
    mean_random_effect_user = trace.posterior['random_effect_user'].mean(axis=-1).values

    # Compute the expected rates for after_pivot = 0 and after_pivot = 1
    rows = []
    for qa_prob in np.linspace(0, 1, 100):
        predictions_before = np.exp(trace.posterior['intercept'] + mean_pivot_intercept)
        for r in predictions_before.values.flatten():
            rows.append({'qa_prob': qa_prob, 'after_pivot': 'before pivot', 'rte_rate': r})

        predictions_after = np.exp(trace.posterior['intercept'] + mean_pivot_intercept +
                                   (trace.posterior['fixed_effect'] + mean_random_effect_user) * qa_prob)
        for r in predictions_after.values.flatten():
            rows.append({'qa_prob': qa_prob, 'after_pivot': 'after pivot', 'rte_rate': r})
    df = pd.DataFrame(rows)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='qa_prob', y='rte_rate', hue='after_pivot', errorbar='sd')
    plt.xlabel('Probability that pivot answers at least one prior question')
    plt.ylabel('Expected rate of user posts entailing the pivot')
    plt.title('Model prediction: entailment rate before and after pivot, depending on question probability')
    plt.legend(title='For user posts that occur:')
    plt.show()


def main():

    ########
    # TODO: Make these config/command line args

    DRAWS = 2000
    TUNE = 1000
    CHAINS = 4

    TOY = True
    N_QA = 1000
    N_RTE = 5000
    N_PIVOTS = 20
    N_USERS = 5
    EFFECT_SIZE = .5

    PATH_QA = '../data/pairs_qa_scores.tsv'
    PATH_RTE = '../data/pairs_rte_scores.tsv'

    ##############

    if TOY:
        qa, rte = load_toy_data(N_QA, N_RTE, N_USERS, N_PIVOTS, EFFECT_SIZE)
    else:
        qa, rte = load_real_data(PATH_QA, PATH_RTE)

    data = combine_qa_and_rte_data(qa, rte)

    plot_data(data)
    plt.show()

    # for pymc, ids should be integers:
    data['user_id'] = pd.Categorical(data['user_id']).codes
    data['pivot_id'] = pd.Categorical(data['pivot_id']).codes

    model, relevant_var_names = init_mixed_effects_Poisson_model(data)
    trace = do_sampling(model, draws=DRAWS, tune=TUNE, chains=CHAINS)

    azsummary = az.summary(trace, round_to=2, var_names=relevant_var_names)
    print(azsummary.to_string())

    azplot = az.plot_trace(trace, combined=True, var_names=relevant_var_names)
    plt.show()

    posterior_plot = plot_posterior(data, trace)
    plt.show()



if __name__ == '__main__':
    main()