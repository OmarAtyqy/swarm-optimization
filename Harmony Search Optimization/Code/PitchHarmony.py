import numpy as np
from PitchAgent import PitchAgent
from tqdm import tqdm


# class to handle the harmony agent
class PitchHarmony:

    def __init__(self, targetFunction, memSize=75, p=0.5, epsilon=0.01):

        # set the target function and the pitch memory size
        self.targetFunction = targetFunction
        self.memSize = memSize

        # set the pitch adjustment rate and the epislon value
        self.p = p
        self.epsilon = epsilon

        # start by populating the memory with random agents spread randomly across the search space
        self.memory = []
        for _ in range(memSize):
            position = np.random.uniform(targetFunction.lowerBound, targetFunction.upperBound, size=(targetFunction.d,))
            self.memory.append(PitchAgent(position, targetFunction))

        
        # get the worst agent in the memory
        self.worstAgent = self.get_worst_pitch()
    
    # method to get the worst agent in the memory
    # since we're dealing with a minimization problem, the worst agent is the one with the highest fitness
    def get_worst_pitch(self):
        return max(self.memory, key=lambda agent: agent.fitness)
    
    # method to get a new pitch by randomly combining elements from pitches in the memory
    def get_new_pitch(self):
        position = np.zeros(self.targetFunction.d)
        for i in range(self.targetFunction.d):
            # randomly choose an agent from the memory and get its corresponding ith component
            agent = np.random.choice(self.memory)
            component = agent.position[i]

            # randomly decide between slighly adjusting the component or changing it randomly
            if np.random.uniform() < self.p:
                component += np.random.uniform(-self.epsilon, self.epsilon)
            else:
                component = np.random.uniform(self.targetFunction.lowerBound[i], self.targetFunction.upperBound[i])

            # set the ith component of the new pitch
            position[i] = component
        
        # create a new agent with the new pitch
        return PitchAgent(position, self.targetFunction)

    # method to get the best agent in the memory
    # since we're dealing with a minimization problem, the best agent is the one with the lowest fitness
    def get_best_pitch(self):
        return min(self.memory, key=lambda agent: agent.fitness)
    
    # method to update the memory with a new pitch
    # if the new pitch performs better than the current worst agent, replace the worst agent with the new pitch
    def update_memory(self, newPitch):
        if newPitch.fitness < self.worstAgent.fitness:
            self.memory.remove(self.worstAgent)
            self.memory.append(newPitch)
            
            # recompute the worst agent
            self.worstAgent = self.get_worst_pitch()
    
    # method to run the harmony search algorithm
    def run(self, index, verbose=False):

        description = f"Harmony {index+1}"
        with tqdm(total=self.targetFunction.maxIter, disable=not verbose, position=index, desc=description) as pbar:
            for iter in range(self.targetFunction.maxIter):

                # get a new pitch
                newPitch = self.get_new_pitch()

                # update the memory
                self.update_memory(newPitch)

                pbar.update(1)
        
        # at the end of the search, get the best whale
        bestPitch = self.get_best_pitch()

        # store the best position and fitness found by this pod
        self.bestPosition = bestPitch.position
        self.bestFitness = bestPitch.fitness