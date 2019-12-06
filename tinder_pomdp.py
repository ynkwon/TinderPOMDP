
from random import random
import itertools
import math
import numpy as np
from scipy.stats import norm
import random

class TinderMDP():

    
    h = 100
    attr_range = 10
    num_likes = 20
    p_fl = 0.2
    p_fdl = 0.15 
    observations = ['matched', 'unmatched']
    actions = ['like', 'dislike', 'changeprofile']
    candidate_mean, candidate_std = 5, 3
    state = []
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
        prob *= (s[0] == s_next[0])
    else:
        if abs(s_next[0] - s[0]) <= mdp.attr_change_range:
            possibility_size = 1 # number of possible values to change to including self
            possibility_size += min(mdp.attr_change_range, s[0] - 1)
            possibility_size += min(mdp.attr_change_range, mdp.attr_range - s[0])
                        
            prob *= 1 / possibility_size
        else: 
            prob *= 0
    return prob


def get_next_states(mdp, my_attr, a, remaining_h, remaining_l):
    nextstates = []
    nextmystates = [my_attr]
    if a == 'changeprofile':
        nextmystates = range(max(1, my_attr - mdp.attr_change_range), min(mdp.attr_range, my_attr + mdp.attr_change_range) + 1)
    for i in itertools.product(nextmystates, range(1, mdp.attr_range + 1)):
        nextstates.append([i[0], i[1], remaining_h-1, remaining_l - (a == 'like')])
    return nextstates

def recursive_reward(mdp, s, a, o, level=1):
    
    reward = immediate_reward(mdp, s, a, o)
    
    if level==0:
        return reward
    
    for a_new in mdp.actions:
        for s_new in get_next_states(mdp, s[0], a_new, s[2], s[3]):
            t = transition(mdp, s, a_new, s_new)
            if (t == 0):
                continue
            for o_new in mdp.observations:
                reward += t * observation(mdp, a_new, o_new, s_new) * recursive_reward(mdp, s_new, a_new, o_new, level - 1)
    
    return reward
    
    

def immediate_reward(mdp, s, a, o):
    result = 0
    
    beta = mdp.num_likes / mdp.h
    #like_cost = 

    if(a == 'like'):
        result -= beta * (s[2] - s[3]) / s[3]
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
    
    #print("b: ", b, "a: ", a, "o: ", o, "Updated B: ", newB)
        
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

def generate_candidates(horizon):
    s = np.random.normal(5.5, 2, horizon)
    for i, n in enumerate(s):
        if n < 1:
            s[i] = 1
        elif n > 10:
            s[i] = 10
        else:
            s[i] = int(n)
    return s

def mynew_attr(attr):
    lower = attr-2 if attr>2 else 1
    higher = attr+2 if attr<9 else 10
    
    return random.randint(lower, higher)

def simulate_MDP(true_myattr, policy, b, reward):
    mdp = TinderMDP()
    total_reward = 0
    total_matches = 0
    data = generate_candidates(mdp.h)
    mdp.state = [true_myattr, data[0], mdp.h, mdp.num_likes]
    count = 0
    
    while mdp.state[2] > 0 and mdp.state[3] > 0:
        print("Simulating horizon ", mdp.state[2])
        a, r = policy(mdp, b, mdp.state[2], mdp.state[3], reward)
        
        o = "matched" if random.random() < observation(mdp, a, "matched", mdp.state) else "unmatched"
        if o == "matched":
            total_matches += 1
        this_reward = reward(mdp,mdp.state, a, o)
        print("action chosen: ", a, " reward: ", this_reward)
        total_reward += reward(mdp,mdp.state, a, o)
        if (count >= mdp.h-1):
            break
        
        b = updatebelief(mdp, b, a, o, mdp.state[2], mdp.state[3])
        nextcand_attr = data[count + 1]
        print("current state: ", mdp.state)
        if a == "like":
            mdp.state = [mdp.state[0], nextcand_attr, mdp.state[2] - 1, mdp.state[3] - 1]
        elif a == "dislike":
            mdp.state = [mdp.state[0], nextcand_attr, mdp.state[2] - 1, mdp.state[3]]
        else:
            mdp.state = [mynew_attr(mdp.state[0]), nextcand_attr, mdp.state[2] - 1, mdp.state[3]]
        count += 1
    return total_reward, total_matches
                
def choose_action_random(mdp, b, remaining_h, remaining_l, reward):
    if remaining_l == 0:
        return 'dislike'
    choice = random.randint(0, 2)
    return mdp.actions[choice],

def expected_utility(alpha, b):
    return sum(i * j for i, j in zip(alpha, b))
    
def choose_max_utility_action(mdp, b, remaining_h, remaining_l, reward):
    exp_utilities = []
    for a in mdp.actions:
        alpha = alphavector(mdp, a, remaining_h, remaining_l, reward)
        exp_utilities.append(expected_utility(alpha, b))
    return mdp.actions[np.argmax(exp_utilities)], np.max(exp_utilities)

def greedy_action(mdp, b, remaining_h, remaining_l, reward):
    candidate_attr = mdp.state[1]
    if candidate_attr > 7:
        return 'like', -1
    elif candidate_attr < 4:
        return 'dislike', -1
    else:
        return 'changeprofile', -1
          
def choose_recursive_action(mdp, b, remaining_h, remaining_l, reward, depth=1, gamma=0.9):
    if depth == 0 or remaining_h == 1 or remaining_l == 1:
        a, r = choose_max_utility_action(mdp, b, remaining_h, remaining_l, reward)
        return a, r
    a_max, u_max = -1, float("-inf")
    for a in mdp.actions:
        alpha = alphavector(mdp, a, remaining_h, remaining_l, reward)
        u = expected_utility(alpha, b)
        for o in mdp.observations:
            if(o == "matched" and a != "liked"):
                continue 
            b_new = updatebelief(mdp, b, a, o, remaining_h, remaining_l)
            a_new, u_new = choose_recursive_action(mdp, b_new, remaining_h - 1, remaining_l - (a == 'like'), reward, depth=depth-1)
            
            for i in range(len(b)):
                nextstates = get_next_states(mdp, i + 1, a_new, remaining_h, remaining_l)
                for state_new in nextstates:
                    u += gamma * b[i] * observation(mdp, a, o, state_new) * u_new
        if u > u_max:
            a_max, u_max = a, u
    return a_max, u_max
        
    
if __name__ == "__main__":
    
    mdp = TinderMDP()
    true_myattr = 7
    
    #b
    uniformB = initbelief(mdp)
    
    
    total_reward, total_matches = simulate_MDP(true_myattr, choose_recursive_action, uniformB, immediate_reward)
    
    print("myattr: ", true_myattr, "action: max_utility, b: ", uniformB, " total_reward: ", total_reward, " total_matches: ", total_matches)
    
    #print(mdp.actions)
    #print("initial belief")
    #print(b)
    
    # alpha = alphavector(mdp, 'like', 400, 100, recursive_reward)
#     print("like, horizon 400", alpha)
#     print(sum(i * j for i, j in zip(alpha, b)))

#     alpha = alphavector(mdp, 'changeprofile', 400, 100, recursive_reward)
#     #alpha = alphavector(mdp, 'dislike', 500, 100)
#     print("changeprofile, horizon 400" , alpha)
    #print(sum(i * j for i, j in zip(alpha, b)))
    
    #b = updatebelief(mdp, b, 'like', 'unmatched', 500, 100)
    #print("after updating belief with like: ", b)
    # newb = updatebelief(mdp, b, 'like', 'matched', 499, 99)
    # print("after updating belief with changeprofile: ")
    # print(newb)

                                

                                
                                
                                
