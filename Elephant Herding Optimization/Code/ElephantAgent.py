import numpy as np


# agent class to represent an elephant in the EHO algorithm
class ElephantAgent:

    # class constuctor
    def __init__(self, targetFunction):
        self.targetFunction = targetFunction
        self.position = np.random.uniform(targetFunction.lowerBound, targetFunction.upperBound, size=(targetFunction.d, ))

        self.isMatriarch = False
        self.isAdult = False

        # calculate the fitness of the agent
        self.fitness = self.targetFunction.function(self.position)
    
    # method to update the position of the agent
    def update_position(self, newPosition):
        self.position = newPosition
        self.fitness = self.targetFunction.function(self.position)