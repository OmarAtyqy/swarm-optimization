from WhaleAgent import WhaleAgent
import numpy as np
from tqdm import tqdm


# class to model a whale pod
class WhalePod:

    def __init__(self, targetFunction, podSize=1000, b=1):
        self.targetFunction = targetFunction
        self.podSize = podSize

        # constant that defines the shape of the logaritmic spiral
        self.b = b

        # variables to store the best position and fitness found by this pod
        self.bestPosition = None
        self.bestFitness = None

        # initialize the whales
        self.whales = [WhaleAgent(targetFunction, self) for _ in range(podSize)]
    
    # method to get the best whale
    # for a minimization problem, the best whale is the one with the lowest fitness
    def get_best_whale(self):
        return min(self.whales, key=lambda whale: whale.fitness)
    
    # method to get a random whale from the pod
    def get_random_whale(self):
        return np.random.choice(self.whales)
    
    # method to run the search
    def run(self, index, verbose=False):

        description = f"Pod {index+1}"
        with tqdm(total=self.targetFunction.maxIter, disable=not verbose, position=index, desc=description) as pbar:
            for iter in range(self.targetFunction.maxIter):

                # for each whale in the pod
                # decide on the course of action
                for whale in self.whales:
                    whale.decide(iter)

                pbar.update(1)
        
        # at the end of the search, get the best whale
        bestWhale = self.get_best_whale()

        # store the best position and fitness found by this pod
        self.bestPosition = bestWhale.position
        self.bestFitness = bestWhale.fitness