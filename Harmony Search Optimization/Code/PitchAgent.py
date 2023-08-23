import numpy as np


# class to represent a Pitch Agent
class PitchAgent:

    def __init__(self, position, targetFunction) -> None:
        
        self.targetFunction = targetFunction

        # upon creation, generate a random point in the search space
        self.position = position

        # evaluate the agent's position
        self.fitness = self.evaluate()

        # set the isWorse property to false
        self.isWorse = False
    
    # function to evaluate the agent's position
    def evaluate(self):
        return self.targetFunction.function(self.position)