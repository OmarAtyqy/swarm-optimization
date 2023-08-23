# helper class to hold the target function and its info
class TargetFunction:

    def __init__(self, function, lowerBound, upperBound, d, maxIter=1000):
        self.function = function
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.d = d
        self.maxIter = maxIter