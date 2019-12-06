from random import random
import itertools
import math
import numpy as np
from scipy.stats import norm
import random

class TinderMDP():

    
    h = 500
    attr_range = 10
    num_likes = 100
    p_fl = 0.2
    p_fdl = 0.15 
    observations = ['matched', 'unmatched']
    actions = ['like', 'dislike', 'changeprofile']
    candidate_mean, candidate_std = 5, 3
        
    attr_change_range = 2

def space(x, n): return [i for i in itertools.product(x, repeat = n)]
# [ my_attr, candidate_attr, horizon_remaining, likes_remaining ]
def states(mdp):
    return [i for i in itertools.product(range(1, mdp.attr_range + 1), range(1, mdp.attr_range + 1), range(1, mdp.h + 1), range(1, mdp.num_likes + 1))]

def observation(mdp, a, o, s):
    if(a == 'dislike' or a == 'changeprofile'):
        return o == 'unmatched'

    # prob of getting a match when liked
    prob = 1 / 3 * (math.log(s[0]**2 / s[1], 10) + 1)
    if(o == 'matched'):
        return prob
    else:
        return 1 - prob
    
    
def get_cont_prob(attr):
    if attr not in [1, 10]: 
        return (norm.cdf((attr + 0.5 - mdp.candidate_mean) / mdp.candidate_std) \
                - norm.cdf((attr - 0.5 - mdp.candidate_mean) / mdp.candidate_std) )
    elif attr == 1:
        return norm.cdf((1.5 - mdp.candidate_mean) / mdp.candidate_std) 
    else:
        return 1 - norm.cdf((9.5 - mdp.candidate_mean) / mdp.candidate_std) 
    
def transition(mdp, s, a, s_next):
    prob = 1
    
    if s_next[2] != s[2] - 1:
        return 0
    
    if s_next[3] != (s[3] - (a == 'like')):
        return 0

    # candidate attraction probabilities via normal distrib
    prob *= get_cont_prob(s_next[1])
    
    # changes profile
    if (a != 'changeprofile'):
        prob *= s[0] == s_next[0]
    else:
        if abs(s_next[0] - s[0]) <= mdp.attr_change_range:
            possibility_size = 1 # number of possible values to change to including self
            possibility_size += min(mdp.attr_change_range, s[0] - 1)
            possibility_size += min(mdp.attr_change_range, mdp.attr_range - s[0])
                        
            prob *= 1 / possibility_size
        else: 
            prob *= 0
    return prob

def immediate_reward(mdp, s, a, o):
    result = 0
    
    beta = mdp.num_likes / mdp.h
    like_cost = beta * (s[2] - s[3]) / s[3]

    if(a == 'like'):
        result -= like_cost
    elif(a == 'changeprofile'):
        gamma = mdp.h / 10
        result -= (gamma / s[2] - 1)
    if(o == 'unmatched'):
        return result

    # if matched, consider matched result with candidate
    result += 1.1 * s[1] / (math.log(s[0], 10) + 1)
    return result



    
def initbelief(mdp):
    b = [1 / mdp.attr_range] * mdp.attr_range
    return b

def updatebelief(mdp, b, a, o, remaining_h, remaining_l):
    newB = []
    for i in range(1, mdp.attr_range + 1):
        sums = 0
        for j in range(1, mdp.attr_range + 1):  
            s_next = [i, j, remaining_h - 1, remaining_l - (a == 'like')]
            o_prob = observation(mdp, a, o, s_next)
            for next_attr in itertools.product(range(1, mdp.attr_range + 1), repeat=2):
                s = [next_attr[0], next_attr[1], remaining_h, remaining_l]
                sums += get_cont_prob(j) * transition(mdp, s, a, s_next) * b[s[0]-1]
        newB.append(o_prob * sums)
        
    newB = [float(i) / sum(newB) for i in newB] # normalize
    return newB


def alphavector(mdp, a, remaining_h, remaining_l, reward):
    alpha = []
    for my_attr in range(1, mdp.attr_range+1):
        reward_for_attr = 0
        for candidate_attr in range(1, mdp.attr_range+1):
            s = [my_attr, candidate_attr, remaining_h, remaining_l]
            for o in mdp.observations:
                reward_for_attr += get_cont_prob(candidate_attr) * observation(mdp, a, o, s) * reward(mdp, s, a, o)
        alpha.append(reward_for_attr)
    return alpha

def testGivenResult:
    
    
    
    
    
if __name__ == "__main__":
    
    mdp = TinderMDP()
    b = initbelief(mdp)
    # b = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    
    print(mdp.actions)
    print("initial belief")
    print(b)
    
#     alpha = alphavector(mdp, 'like', 400, 100, immediate_reward)
#     print("like, horizon 400", alpha)
#     print(sum(i * j for i, j in zip(alpha, b)))

#     alpha = alphavector(mdp, 'changeprofile', 400, 100, immediate_reward)
#     #alpha = alphavector(mdp, 'dislike', 500, 100)
#     print("changeprofile, horizon 400" , alpha)
    #print(sum(i * j for i, j in zip(alpha, b)))
    
    #b = updatebelief(mdp, b, 'like', 'unmatched', 500, 100)
    #print("after updating belief with like: ", b)
    
    newb = updatebelief(mdp, b, 'changeprofile', 'unmatched', 499, 99)
    print("after updating belief with changeprofile: ")
    print(newb)

