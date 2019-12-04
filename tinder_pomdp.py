import numpy as np
import matplotlib.pyplot as plt

class TinderMDP():

    states = range(1, 11)
    h = 500
    p_fl = 0.2
    p_fdl = 0.15 
    observations = ['matched', 'unmatched']
    actions = ['like', 'dislike', 'changeprofile']
    candidate_mean, candidate_std = 5, 3
   
            
        
    def transition(self):
    
    def reward(self, s, a, o, candidate_attr):
        result = 0
        
        if(a == 'like'):
            result -= 1
        elif(a == 'dislike'):
            result -= 0.2
     
        if(o == 'unmatched'):
            return result
        
        # if matched, consider matched result with candidate
        result += (candidate_attr - s)
        
    
    def observation(self, s, o):
        
        return
    
    def updatebelief(self, b, a, o):
        newbelief = np.zeros(10)
        self.observation(o, s, a)
        return newbelief
    def 


