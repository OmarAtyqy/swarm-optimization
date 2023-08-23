from PitchHarmony import PitchHarmony
import numpy as np
import threading


# wrapper class for the Harmony Search algorithm
class PitchOptimizer:

    def __init__(self, targetFunction, N_harmonies=10, memSize=100):
        
        # generate the harmonies
        self.N_harmonies = N_harmonies
        self.harmonies = [PitchHarmony(targetFunction,
                                       memSize= memSize,
                                       p=np.random.uniform(0.3, 0.7),
                                       epsilon=np.random.uniform(0.001, 0.01)) for _ in range(N_harmonies)]

        # variables to store the best position and fitness found by this algorithm
        self.bestPosition = None
        self.bestFitness = None

        # variables to store the best parameters found by this algorithm
        self.bestEpsilon= None
        self.best_p= None
        self.bestMemSize= None
        
        # lock for multithreading
        self.lock = threading.Lock()
    
    # fing the optimum value of the target function
    # this implementation runs the algorithm sequentially
    def find_optimum(self, verbose=False):
            
            # run the algorithm for each harmony
            for i in range(self.N_harmonies):
                self.harmonies[i].run(verbose=verbose, index=i)
            
            # get the best harmony
            bestHarmony = min(self.harmonies, key=lambda harmony: harmony.bestFitness)

            # get the best position and fitness found by any harmony
            self.bestPosition = bestHarmony.bestPosition
            self.bestFitness = bestHarmony.bestFitness
    
            # get the best parameters found by any harmony
            self.bestEpsilon = bestHarmony.epsilon
            self.best_p = bestHarmony.p
            self.bestMemSize = bestHarmony.memSize
    
    # method to run a single harmony
    def run_harmony(self, index, verbose):
        harmony = self.harmonies[index]
        harmony.run(index, verbose)
    
    # find the optimum value of the target function
    # this implementation uses threads to run the algorithm in parallel
    # each harmony is run in a separate thread
    def find_optimum_threaded(self, verbose=False):
        # create a list of threads
        threads = []

        # create a thread for each harmony
        for i in range(self.N_harmonies):
            thread = threading.Thread(target=self.run_harmony, args=(i, verbose))
            threads.append(thread)
            thread.start()

        # wait for all threads to finish
        for thread in threads:
            thread.join()

        # get the best position and fitness found by any harmony
        with self.lock:
            # get the best harmony
            bestHarmony = min(self.harmonies, key=lambda harmony: harmony.bestFitness)

            # get the best position and fitness found by any harmony
            self.bestPosition = bestHarmony.bestPosition
            self.bestFitness = bestHarmony.bestFitness
    
            # get the best parameters found by any harmony
            self.bestEpsilon = bestHarmony.epsilon
            self.best_p = bestHarmony.p
            self.bestMemSize = bestHarmony.memSize
    
    # function to print the best position and fitness found by this algorithm
    def print_best(self):
        print("Best position:", self.bestPosition)
        print("Best fitness:", self.bestFitness)
        print("Best epsilon:", self.bestEpsilon)
        print("Best p:", self.best_p)
        print("Best memory size:", self.bestMemSize)
    
    # function to return the best position and fitness found by this algorithm
    def get_best(self, verbose=False):
        if verbose:
            self.print_best()
        return self.bestPosition, self.bestFitness