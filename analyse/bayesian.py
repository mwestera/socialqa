import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import pymc as pm
import arviz as az

import logging
import functools
import cloudpickle

"""
Rough first version.

Currently simulates user's post histories and tries to fit Bayesian models to it.
So far only a simple linear model (e.g., user's posts' entailment scores 
gradually increase as time passes); a more sophisticated curiosity-based model
is a work in progress.

Outputs a number of plots (plt.show(), not saved to disk) and prints some summary statistics.

With current settings takes around 3 minutes to run, and raises some warnings.

Beware of model caching: if you want to fit a new model, delete the cached model file.

Some used and/or potentially useful resources:
- https://www.pymc.io/projects/examples/en/latest/time_series/Air_passengers-Prophet_with_Bayesian_workflow.html
- https://www.pymc.io/projects/examples/en/latest/time_series/AR.html
- https://www.pymc.io/projects/examples/en/latest/time_series/longitudinal_models.html
- https://www.pymc.io/projects/examples/en/latest/causal_inference/interrupted_time_series.html
- https://www.bayesrulesbook.com/chapter-9
- https://areding.github.io/6420-pymc/unit10/Unit10-sunspots.html
"""

def main():
    global do_sampling      # hmmmmmm

    # TODO make command line args
    DRAWS, TUNE, CHAINS = 500, 500, 4
    N_USERS, LENGTH, EFFECT = 10, 1000, .4
    CACHE_FILE = 'cache.pkl' # Caching only works when data remains exactly the same
    SEED = 12345

    if CACHE_FILE:
        do_sampling = cached_to_disk(do_sampling, CACHE_FILE)

    logging.getLogger('pymc').info(f'Random seed: {SEED}')
    rng = np.random.default_rng(SEED)

    df = simulate_multiuser_post_histories(n_users=N_USERS, length=LENGTH, effectsize=EFFECT, rand=rng)
    plot_post_histories(df)
    plt.show()

    df['time'] /= df['time'].max()  # Not sure if this normalization seemed to help with model fitting?

    model = build_linear_model(df)
    # model = build_autoregressive_model(df)    # TODO Work in progress

    with model:
        results = do_sampling(model, draws=DRAWS, tune=TUNE, chains=CHAINS)

    azsummary = az.summary(results['trace'], round_to=2, var_names=['α', 'β', 'rho', 'tau', 'exposure'], filter_vars='like')
    print(azsummary.to_string())

    azplot = az.plot_trace(results['trace'], combined=True, var_names=['α', 'β', 'rho', 'tau', 'exposure'], filter_vars='like')
    plt.show()

    posterior_plot = plot_posterior(df, results['trace'], results['posterior'])
    plt.show()



def cached_to_disk(func, cache):
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



def do_sampling(model, **kwargs):
    with model:
        trace = pm.sample(**kwargs)
        posterior = pm.sample_posterior_predictive(trace=trace)
        # linear_prior = pm.sample_prior_predictive()   # to inspect if the prior even makes sense?
    data = {'trace': trace, 'posterior': posterior}
    return data


def simulate_post_history(length,
                          effectsize,
                          prop_question_posts=0.2,
                          prop_own_posts=0.1,
                          prop_other_posts=0.4,
                          rand=None):
    """
    Creates a dataframe with three time series of length 'size', representing the scores of question_posts with
    which the user has enganged, user's own_posts, and other_posts with
    which the user has engaged.

    Assuming a hypothetical piece of information A, the score of a question_post represents the
    degree to which it is answered by A, for self_posts and other_posts it represents the degree
    to which it entails A.

    A trend is simulated with 'effect size': self_posts' scores will become higher the more the user is exposed to high-scoring
    question_posts and high-scoring other_posts.
    """
    rand = rand or np.random.default_rng()

    question_posts = rand.binomial(1, prop_question_posts, size=length).astype(float)
    own_posts = rand.binomial(1, prop_own_posts, size=length).astype(float)
    other_posts = rand.binomial(1, prop_other_posts, size=length).astype(float)

    question_posts[question_posts == 0] = np.nan
    own_posts[own_posts == 0] = np.nan
    other_posts[other_posts == 0] = np.nan

    # This gamma distribution results in most scores between 0.1-0.6ish.
    engaged_posts_entail_A = other_posts * rand.gamma(5, .05, size=length).clip(0, 1)
    own_posts_entail_A = own_posts * rand.gamma(5, .05, size=length).clip(0, 1)
    question_posts_answered_by_A = question_posts * rand.gamma(5, .05, size=length).clip(0, 1)

    own_posts_entail_A = (own_posts_entail_A * (1 + np.nancumsum(question_posts_answered_by_A * engaged_posts_entail_A) * effectsize)).clip(0, 1)

    df = pd.DataFrame(columns=['time', 'questions', 'own_entailing', 'engaged_entailing'],
                      data=zip(range(length),
                               question_posts_answered_by_A,
                               own_posts_entail_A,
                               engaged_posts_entail_A)
                      )

    return df


def simulate_multiuser_post_histories(n_users, length, effectsize, rand=None):
    """
    Simulate 'post histories' for multiple users, returning a single dataframe.
    """
    dfs = []
    for n in range(n_users):
        df = simulate_post_history(length, effectsize, rand=rand)
        df['user'] = f'user_{n}'
        dfs.append(df)
    final_df = pd.concat(dfs).reset_index(drop=True)
    return final_df


def plot_post_histories(df):
    """
    Create scatterplot of scores of question_posts, own_posts and other_posts,
    with a trend line for the self_posts' scores.
    """
    plt.figure(figsize=(18, 36))
    fig, ax = plt.subplots(nrows=3, ncols=1)

    plt.title('Synthetic data')
    sns.scatterplot(df, x='time', y='own_entailing', hue='user', alpha=.4, ax=ax[0], size=2)
    sns.scatterplot(df, x='time', y='questions', hue='user', alpha=.4, ax=ax[1], size=2)
    sns.scatterplot(df, x='time', y='engaged_entailing', hue='user', alpha=.4, ax=ax[2], size=2)

    for (user, subdf), hue in zip(df.groupby('user'), sns.color_palette()):
        sns.regplot(subdf, x='time', y='own_entailing', scatter=False, color=hue, ax=ax[0])
    ax[0].get_legend().remove()
    ax[1].get_legend().remove()
    ax[2].get_legend().remove()
    plt.tight_layout()


def build_linear_model(df):
    """
    Simple time-only model: own_entailing = α + β * time
    """
    with pm.Model(check_bounds=False) as model:
        α = pm.Normal("α", mu=0, sigma=0.05)
        β1 = pm.Normal("β1", mu=0, sigma=0.05)
        σ = pm.HalfNormal("σ", sigma=0.05)

        trend = pm.Deterministic("trend", α + β1 * df['time'])
        likelihood = pm.Normal("likelihood", mu=trend, sigma=σ, observed=df['own_entailing'])

    return model


def build_autoregressive_model(df):
    """
    More sophisticated model that attempts to model 'curiosity' as a latent, auto-regressive variable.
    Based on https://www.pymc.io/projects/examples/en/latest/time_series/AR.html
    """
    with pm.Model(check_bounds=False) as model:
        # Or consider using a discrete switch point from https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_overview.html#case-study-2-coal-mining-disasters
        # How to handle missing values in input:  https://areding.github.io/6420-pymc/unit8/Unit8-missrats.html

        α = pm.Normal("α", mu=0, sigma=0.05)
        β1 = pm.Normal("β1", mu=0, sigma=0.05)
        σ = pm.HalfNormal("σ", sigma=0.05)

        α_ar = pm.Normal("α_ar", mu=1.0, sigma=1.0)
        β_ar = pm.Normal("β_ar", mu=0, sigma=5)
        rho = pm.Deterministic("rho", α_ar + β_ar * df['engaged_entailing'])
        tau = pm.Exponential("tau", lam=0.5)
        exposure = pm.AR("exposure", rho=rho, constant=True, steps=1, tau=tau, init_dist=pm.Normal.dist(0, 10))

        β2 = pm.Normal("β2", mu=0, sigma=0.5)
        trend = pm.Deterministic("trend", α + β1 * df['time'] + β2 * exposure)
        likelihood = pm.Normal("likelihood", mu=trend, sigma=σ, observed=df['own_entailing'])

    return model


def plot_posterior(df, trace, posterior):
    """
    Plot the posterior predictions as well as a posterior trend line.
    Based on https://www.pymc.io/projects/examples/en/latest/time_series/Air_passengers-Prophet_with_Bayesian_workflow.html#part-1-linear-trend .
    """
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax[0].plot(
        df["time"],
        az.extract(posterior, group="posterior_predictive", num_samples=100)["likelihood"],
        color="blue",
        alpha=0.01,
    )
    df.plot.scatter(x="time", y="own_entailing", color="k", ax=ax[0])
    ax[0].set_title("Posterior predictive")
    ax[1].plot(
        df["time"],
        az.extract(trace, group="posterior", num_samples=100)["trend"],
        color="blue",
        alpha=0.01,
    )
    df.plot.scatter(x="time", y="own_entailing", color="k", ax=ax[1])
    ax[1].set_title("Posterior trend lines")

    return fig


if __name__ == '__main__':
    main()