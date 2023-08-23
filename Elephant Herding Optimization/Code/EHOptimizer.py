from ElephantClan import ElephantClan
import numpy as np
import threading

# class for the EHOptimizer algorithm
class EHOptimizer:

    def __init__(self, targetFunction, N_clans=10, N_elephants=100):
        
        # generate the clans
        self.N_clans = N_clans
        self.clans = [ElephantClan(targetFunction,
                                alpha=np.random.uniform(0, 1),
                                beta=np.random.uniform(0, 1),
                                N_elephants=N_elephants) for _ in range(N_clans)]

        # variables to store the best position and fitness found by this algorithm
        self.bestPosition = None
        self.bestFitness = None

        # variables to store the best parameters found by this algorithm
        self.bestAlpha = None
        self.bestBeta = None

        # lock for multithreading
        self.lock = threading.Lock()

    
    # fing the optimum value of the target function
    # this implementation runs the algorithm sequentially
    def find_optimum(self, verbose=False):
            
            # run the algorithm for each clan
            for i in range(self.N_clans):
                self.clans[i].run(verbose=verbose, index=i)

            # get the best clan
            bestClan = min(self.clans, key=lambda clan: clan.bestFitness)
            
            # get the best position and fitness found by any clan
            self.bestPosition = bestClan.bestPosition
            self.bestFitness = bestClan.bestFitness
    
            # get the best parameters found by any clan
            self.bestAlpha = bestClan.alpha
            self.bestBeta = bestClan.beta
        
        
    # find the optimum value of the target function
    # this implementation uses threads to run the algorithm in parallel
    # each clan is run in a separate thread
    def find_optimum_threaded(self, verbose=False):
            
        # create a list of threads
        threads = []
        
        # create a thread for each clan
        for i in range(self.N_clans):
            thread = threading.Thread(target=self.run_clan, args=(i, verbose))
            threads.append(thread)
            thread.start()
            
        # wait for all threads to finish
        for thread in threads:
            thread.join()
        
        with self.lock:
            # get the best clan
            bestClan = min(self.clans, key=lambda clan: clan.bestFitness)
            
            # get the best position and fitness found by any clan
            self.bestPosition = bestClan.bestPosition
            self.bestFitness = bestClan.bestFitness
    
            # get the best parameters found by any clan
            self.bestAlpha = bestClan.alpha
            self.bestBeta = bestClan.beta
    
    # function to print the best position and fitness found by this algorithm
    def print_best(self):
        print('Best position:', self.bestPosition)
        print('Best fitness:', self.bestFitness)
        print('Best alpha:', self.bestAlpha)
        print('Best beta:', self.bestBeta)
    
    # function to return the best position and fitness found by this algorithm
    def get_best(self, verbose=False):
        if verbose:
            self.print_best()
        return self.bestPosition, self.bestFitness

    # method to run a single clan
    def run_clan(self, index, verbose):
        clan = self.clans[index]
        clan.run(index, verbose)