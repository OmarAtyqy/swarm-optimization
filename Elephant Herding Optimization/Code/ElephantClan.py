import numpy as np
from ElephantAgent import ElephantAgent
from tqdm import tqdm


# class to simulate the herding behavior of a clan of elephants
class ElephantClan:

    # constructor
    def __init__(self, targetFunction, alpha, beta, N_elephants=1000):

        # clan parameters
        self.alpha = alpha
        self.beta = beta
        self.targetFunction = targetFunction

        # generate the elephants with random positions
        self.elephants = [ElephantAgent(targetFunction) for _ in range(N_elephants)]

        # determine the initial matriarch of this clan
        self.matriarch = self.get_matriarch()

        # variables to store the best position and fitness found by this clan
        self.bestPosition = None
        self.bestFitness = None
    
    # method to get the matriarch of this clan
    # the matriarch is the elephant with the lowest fitness
    def get_matriarch(self):
        return min(self.elephants, key=lambda elephant: elephant.fitness)
    
    # method to get the adult of this clan
    # the adult is the elephant with the highest fitness
    def get_adult(self):
        return max(self.elephants, key=lambda elephant: elephant.fitness)
    
    # method to get the center of the clan
    def get_center(self):
        return np.mean([elephant.position for elephant in self.elephants if not elephant.isAdult], axis=0)

    # method to update the position of the elephants in the clan
    def update_positions(self):
        for elephant in self.elephants:
            
            # check if the elephant is not an adult
            if not elephant.isAdult:

                # check if the elephant is the matriarch
                # if yes, update according to the center of the clan
                if elephant.isMatriarch:
                    new_position = self.beta * self.get_center()
                    elephant.update_position(new_position)
                
                # otherwise, update according to the position of the matriarch
                else:
                    new_position = elephant.position + self.alpha * np.random.uniform() * (self.matriarch.position - elephant.position)
                    elephant.update_position(new_position)
    
    # method to run the EHO algorithm for a given number of iterations for this clan
    def run(self, index, verbose=False):

        description = f"Clan {index+1}"
        with tqdm(total=self.targetFunction.maxIter, disable=not verbose, position=index, desc=description) as pbar:
            # if verbose is True, show a progress bar
            for _ in range(self.targetFunction.maxIter):
                
                # update the positions of the elephants
                self.update_positions()

                # change the matriarch
                self.matriarch.isMatriarch = False
                self.matriarch = self.get_matriarch()

                # get the new adult
                self.adult = self.get_adult()

                # excile the adult from the clan 
                self.adult.isAdult = True
                self.adult.position = self.targetFunction.lowerBound + np.random.uniform() * (self.targetFunction.upperBound - self.targetFunction.lowerBound + 1)

                pbar.update(1)
        
        # update the best position and fitness found by this clan
        self.bestPosition = self.matriarch.position
        self.bestFitness = self.matriarch.fitness