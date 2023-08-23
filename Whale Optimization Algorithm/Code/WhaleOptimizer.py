from WhalePod import WhalePod
import numpy as np
import threading

# class wrapper for the Whale Optimization Algorithm
class WhaleOptimizer:

    def __init__(self, targetFunction, N_pods=10, N_whales=100):
        
        # generate the pods
        self.N_pods = N_pods
        self.pods = [WhalePod(targetFunction,
                              podSize=N_whales,
                              b=np.random.randint(1, N_pods)) for _ in range(N_pods)]

        # variables to store the best position and fitness found by this algorithm
        self.bestPosition = None
        self.bestFitness = None

        # variables to store the best parameters found by this algorithm
        self.best_b = None
        
        # lock for multithreading
        self.lock = threading.Lock()
    
    # fing the optimum value of the target function
    # this implementation runs the algorithm sequentially
    def find_optimum(self, verbose=False):
            
            # run the algorithm for each pod
            for i in range(self.N_pods):
                self.pods[i].run(verbose=verbose, index=i)

            # get the best pod
            bestPod = min(self.pods, key=lambda pod: pod.bestFitness)
            
            # get the best position and fitness found by any pod
            self.bestPosition = bestPod.bestPosition
            self.bestFitness = bestPod.bestFitness
    
            # get the best parameters found by any pod
            self.best_b = bestPod.b
    
    # find the optimum value of the target function
    # this implementation uses threads to run the algorithm in parallel
    # each pod is run in a separate thread
    def find_optimum_threaded(self, verbose=False):
        # create a list of threads
        threads = []

        # create a thread for each pod
        for i in range(self.N_pods):
            thread = threading.Thread(target=self.run_pod, args=(i, verbose))
            threads.append(thread)
            thread.start()

        # wait for all threads to finish
        for thread in threads:
            thread.join()

        # get the best position and fitness found by any pod
        with self.lock:
            # get the best pod
            bestPod = min(self.pods, key=lambda pod: pod.bestFitness)
            
            # get the best position and fitness found by any pod
            self.bestPosition = bestPod.bestPosition
            self.bestFitness = bestPod.bestFitness
    
            # get the best parameters found by any pod
            self.best_b = bestPod.b
    
    # function to print the best position and fitness found by this algorithm
    def print_best(self):
        print('Best position:', self.bestPosition)
        print('Best fitness:', self.bestFitness)
        print('Best b:', self.best_b)
    
    # function to return the best position and fitness found by this algorithm
    def get_best(self, verbose=False):
        if verbose:
            self.print_best()
        return self.bestPosition, self.bestFitness
    
    # method to run a single pod
    def run_pod(self, index, verbose):
        pod = self.pods[index]
        pod.run(index, verbose)