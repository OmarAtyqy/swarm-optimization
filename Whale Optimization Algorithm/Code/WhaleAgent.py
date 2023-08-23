import numpy as np


# class for the whale agent
class WhaleAgent:

    def __init__(self, targetFunction, pod):
        self.targetFunction = targetFunction
        self.position = np.random.uniform(targetFunction.lowerBound, targetFunction.upperBound, size=(targetFunction.d, ))

        # the pod the agent belongs to
        self.pod = pod

        self.r = np.random.uniform(0, 1)

        # calculate fitness
        self.fitness = self.evaluate()
    
    # evaluate the fitness of the agent
    def evaluate(self):
        return self.targetFunction.function(self.position)

    # def update position
    def update_position(self, newPosition):
        self.position = newPosition
        self.keep_in_bounds()
        self.fitness = self.evaluate()

    # method to update the position of the agent using the spiral mechanism
    def encircle(self, A, C):
        bestWhale = self.pod.get_best_whale()
        if bestWhale != self:
            D = np.abs(C * bestWhale.position - self.position)
            newPosition = bestWhale.position - A * D
            self.update_position(newPosition)
    
    # method to update the position of the agent using the searching mechanism
    def search(self, A, C):
        # keep choosing whales at random until you find one that is not the current whale
        targetWhale = self.pod.get_random_whale()
        while targetWhale == self:
            targetWhale = self.pod.get_random_whale()

        D = np.abs(C * targetWhale.position - self.position)
        newPosition = targetWhale.position - A * D
        self.update_position(newPosition)
    
    # method to update the position of the agent using the attacking mechanism
    def attack(self, l):
        bestWhale = self.pod.get_best_whale()
        if bestWhale != self:
            newPosition = np.abs(bestWhale.position - self.position) * np.exp(self.pod.b * l) * np.cos(2 * np.pi * l) + bestWhale.position
            self.update_position(newPosition)

    # check if whale is out of bounds
    # if it is, then update its position
    def keep_in_bounds(self):
        for i in range(self.targetFunction.d):
            if self.position[i] < self.targetFunction.lowerBound[i]:
                self.position[i] = self.targetFunction.lowerBound[i]
            elif self.position[i] > self.targetFunction.upperBound[i]:
                self.position[i] = self.targetFunction.upperBound[i]
    
    # method to decide on the course of action
    def decide(self, iter):
        # choose the parameters
        a = 2 * (1 - iter/self.targetFunction.maxIter)
        p = np.random.uniform()
        C = 2 * self.r
        A = a * (2 * self.r - 1)
        l = np.random.uniform(-1, 1)

        # if p is less than 0.5 and |A| < 1, then update then choose the encircling mechanism
        if p < 0.5 and np.abs(A) < 1:
            self.encircle(A, C)
        
        # if p is less than 0.5 and |A| >= 1, then update the best whale position using the searching mechanism
        elif p < 0.5 and np.abs(A) >= 1:
            self.search(A, C)
        
        # if p is greater than 0.5, then update the best whale position using the attacking mechanism
        else:
            self.attack(l)