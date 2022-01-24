from pdb import set_trace as T

from collections import defaultdict
import numpy as np

import openskill


def rank(policy_ids, scores):
    '''Compute policy rankings from per-agent scores'''
    agents = defaultdict(list)
    for policy_id, score in zip(policy_ids, scores):
        agents[policy_id].append(score + 1e-8*np.random.normal())

    # Double argsort returns ranks
    return np.argsort(np.argsort(
        [-np.mean(vals) for policy, vals in 
        sorted(agents.items())])).tolist()


class OpenSkillRating:
    '''OpenSkill Rating wrapper for estimating relative policy skill

    Provides a simple method for updating skill estimates from raw
    per-agent scores as are typically returned by the environment.'''
    def __init__(self, agents, anchor, mu=1000, sigma=100/3, anchor_mu=1500):
        '''
        Args:
            agents: List of agent classes to rank
            anchor: Baseline policy name to anchor to mu
            mu: Anchor point for the baseline policy (cannot be exactly 0)
            sigma: 68/95/99.7 win rate against 1/2/3 sigma lower SR'''

        if __debug__:
            err = 'Agents must be ordered (e.g. list, not set)'
            assert type(agents) != set, err

        self.ratings = {agent:
                openskill.Rating(mu=mu, sigma=sigma)
                for agent in agents}

        self.mu        = mu
        self.sigma     = sigma
        self.anchor    = anchor
        self.anchor_mu = anchor_mu

        self.anchor_baseline()

    def update(self, ranks=None, policy_ids=None, scores=None):
        '''Updates internal skill rating estimates for each policy

        You should call this function once per simulated environment
        Provide either ranks OR policy_ids and scores

        Args:
            ranks: List of ranks in the same order as agents
            policy_ids: List of policy IDs for each agent episode
            scores: List of scores for each agent episode

        Returns:
            Dictionary of ratings keyed by agent names'''

        if __debug__:
            err = 'Specify either ranks or policy_ids and scores'
            assert ranks is None  != (policy_ids is None and scores is None), err

        if ranks is None:
            ranks = rank(policy_ids, scores)

        teams = [[e] for e in list(self.ratings.values())]
        ratings = openskill.rate(teams, rank=ranks)
        ratings = [openskill.create_rating(team[0]) for team in ratings]
        for agent, rating in zip(self.ratings, ratings):
            self.ratings[agent] = rating

        self.anchor_baseline()

        return self.ratings

    def anchor_baseline(self):
        '''Resets the anchor point policy to mu SR'''
        for agent, rating in self.ratings.items():
            rating.sigma = self.sigma
            if agent == self.anchor:
                rating.mu    = self.anchor_mu
                rating.sigma = self.sigma
