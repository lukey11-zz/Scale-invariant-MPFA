#!/usr/bin/env python

import argos_util
import subprocess
import csv
import tempfile
import os
import numpy as np
import time
import argparse
import errno
import copy
from lxml import etree
import logging
import pdb

# http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class ArgosRunException(Exception):
    pass


class iAntGA(object):
    def __init__(self, pop_size=50, gens=20, elites=1,
                 mut_rate=0.1, system="linux", terminateFlag=0):
				
        self.system = system
        self.pop_size = pop_size
        self.gens = gens
        self.elites = elites
        self.mut_rate = mut_rate
        self.current_gen = 0
        # Initialize population
        self.population_data=[]
        self.population = []
        self.prev_population = []
        self.system = system
        self.fitness = np.zeros(pop_size)
        self.starttime = int(time.time())
        self.terminateFlag =terminateFlag 
        self.not_evolved_idx = [-1]*self.pop_size #check whether a population is from previous generation and is not modified
        self.not_evolved_count = [0]*self.pop_size
        self.prev_not_evolved_count = [0]*self.pop_size 
        self.prev_fitness = np.zeros(pop_size) 
        for _ in xrange(pop_size):
            #self.population.append({"X": np.random.uniform(0, 4), "Y": np.random.uniform(0, 1), "Z": np.random.uniform(0, 4)})
            self.population.append({"X": np.random.uniform(0, 4), "Y": np.random.uniform(0, 1)})
        dirstring = str(self.starttime) + "_e_" + str(elites) + "_p_" + str(pop_size)
        self.save_dir = os.path.join("gapy_saves", dirstring)
        mkdir_p(self.save_dir)
        logging.basicConfig(filename=os.path.join(self.save_dir,'iAntGA.log'),level=logging.DEBUG)

    def test_fitness(self, pop, datas): #p: parameters
        fitness = 0.0
        for [d, r, f] in datas:
            #fitness = fitness + (pop["Z"] + pop["X"]*np.log2(d) -np.log2(f) - (pop["Y"]*np.log2(r)))**2
            fitness += (pop["X"]*np.log2(d) -np.log2(f) - (pop["Y"]*np.log2(r)))**2
        return fitness/len(datas)

    def run_ga(self, datas):
        while self.current_gen <=self.gens and self.terminateFlag == 0:
            self.run_generation(datas)

    def run_generation(self, datas):
        logging.info("Starting generation: " + str(self.current_gen))
        self.fitness = np.zeros(pop_size) #reset it
        for i, pop in enumerate(self.population):
            print "Gen: "+str(self.current_gen)+'; Population: '+str(i+1)
            
            if self.not_evolved_idx[i] == -1 or self.not_evolved_count[i] > 3:
                self.not_evolved_count[i] =0;    
                self.fitness[i] = self.test_fitness(pop, datas)
            else:
		self.fitness[i] = self.prev_fitness[self.not_evolved_idx[i]] 
	        logging.info("partial fitness = %f", self.prev_fitness[self.not_evolved_idx[i]])
        
        # sort fitness and population
        #fitpop = sorted(zip(self.fitness, self.population, self.not_evolved_count), reverse=True)
        fitpop = sorted(zip(self.fitness, self.population, self.not_evolved_count), reverse=False)  
        self.fitness, self.population, self.not_evolved_count = map(list, zip(*fitpop))
        
        self.save_population()
        
        self.prev_population = copy.deepcopy(self.population)
        self.prev_fitness = copy.deepcopy(self.fitness) 
	self.prev_not_evolved_count = copy.deepcopy(self.not_evolved_count) 

        self.not_evolved_idx=[] 
        self.not_evolved_count = []
        self.population = []
        self.check_termination() 
        self.population_data=[] 
        # Add elites
        for i in xrange(self.elites):
            # reverse order from sort
            self.population.append(self.prev_population[i])
            self.not_evolved_idx.append(i) 
            self.not_evolved_count.append(self.prev_not_evolved_count[i] + 1)

        # Now do crossover and mutation until population is full

        num_newOffSpring = self.pop_size - self.elites
        count = 0
        for i in xrange(num_newOffSpring):
            if count == num_newOffSpring: break
            p1c = np.random.choice(len(self.prev_population), 2)
            p2c = np.random.choice(len(self.prev_population), 2)
            if p1c[0] <= p1c[1]:
                parent1 = self.prev_population[p1c[0]]
                idx1 = p1c[0]
            else: 
                parent1 = self.prev_population[p1c[1]]
                idx1 = p1c[1]
                
            if p2c[0] <= p2c[1]:
                parent2 = self.prev_population[p2c[0]]
                idx2 = p2c[0]
            else:
                parent2 = self.prev_population[p2c[1]]
                idx2 = p2c[1]
            
            if parent1 != parent2: #crossover
                
                children = argos_util.uniform_crossover(parent1, parent2, 0.5, self.system)
            else:
                children = [copy.deepcopy(parent1), copy.deepcopy(parent2)]
            for child in children:
                argos_util.mutate_parameters(child, self.mut_rate)
                self.population.append(child)
                if parent1 == child:
                    self.not_evolved_idx.append(idx1)
                    self.not_evolved_count.append(self.prev_not_evolved_count[idx1] + 1)
	       	elif parent2 == child:
                    self.not_evolved_idx.append(idx2) 
                    self.not_evolved_count.append(self.prev_not_evolved_count[idx2] + 1)
                else:
                    self.not_evolved_idx.append(-1)
                    self.not_evolved_count.append(0)
            count += 2
            while count > num_newOffSpring:
                del self.population[-1]
                del self.not_evolved_idx[-1]
                del self.not_evolved_count[-1]
                count -=1
        self.current_gen += 1

    def check_termination(self):
        upperBounds = [4.0, 4.0]
        fitness_convergence_rate = 0.97
        diversity_rate=0.03
        data_keys= self.population_data[0].keys()
        data_keys.sort()
        complete_data =[]
        for data in self.population_data:
            complete_data.append([float(data[key]) for key in data_keys])
        npdata = np.array(complete_data)

        #Fitness convergence and population diversity
        means = npdata.mean(axis=0)
        stds = np.delete(npdata.std(axis=0), [2])
        normalized_stds = stds/upperBounds
        
        current_fitness_rate = npdata[0,2]/means[2]
        current_diversity_rate = normalized_stds.max()
        if npdata[0,2]<1.0 and current_diversity_rate<=diversity_rate and current_fitness_rate>= fitness_convergence_rate:
            self.terminateFlag = 1
            print "Convergent ..."
            print 
        elif current_diversity_rate>diversity_rate and current_fitness_rate<fitness_convergence_rate:
            print 'Fitness is not convergent ...best fitness', npdata[0,2]
            print 'Fitness rate is '+str(current_fitness_rate)
            print 'Deviation is '+str(current_diversity_rate)
        elif current_diversity_rate > diversity_rate:
            print  'Best fitness = ', npdata[0,2] 
            print 'Fitness rate is '+str(current_fitness_rate)
            print 'population diversity is high ...'
            print 'The curent standard deviation is '+str(current_diversity_rate)+', which is greater than '+str(diversity_rate)+' ...'
        else:
            print 'Fitness is not convergent ...', npdata[0,2], str(current_fitness_rate)
            
            print 'The current rate of mean of fitness is '+str(current_fitness_rate)+', which is less than '+str(fitness_convergence_rate)+' ...'

    def save_population(self):
        save_dir = self.save_dir
        mkdir_p(save_dir)
        filename = "gen_%d.gapy" % self.current_gen
        for f, p in zip(self.fitness, self.population):
            p["fitness"] = f
            self.population_data.append(p)
            
        data_keys = p.keys()
        
        data_keys.sort()
        
        with open(os.path.join(save_dir, filename), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data_keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.population_data) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GA for argos')
    parser.add_argument('-f', '--file', action='store', dest='xml_file')
    parser.add_argument('-s', '--system', action='store', dest='system')
    parser.add_argument('-r', '--robots', action='store', dest='robots', type=int)
    parser.add_argument('-m', '--mut_rate', action='store', dest='mut_rate', type=float)
    parser.add_argument('-e', '--elites', action='store', dest='elites', type=int)
    parser.add_argument('-g', '--gens', action='store', dest='gens', type=int)
    parser.add_argument('-p', '--pop_size', action='store', dest='pop_size', type=int)
    parser.add_argument('-t', '--time', action='store', dest='time', type=int)
    parser.add_argument('-k', '--tests_per_gen', action='store', dest='tests_per_gen', type=int)
    parser.add_argument('-o', '--terminateFlag', action='store', dest='terminateFlag', type=int)
    pop_size = 40
    gens = 100
    elites = 1
    mut_rate = 0.05
    
    system = "linux"
    tests_per_gen= 10
    terminateFlag = 0
    
    print "pop_size ="+ str(pop_size)
    print "gens="+str(gens)
    print "elites="+ str(elites)
    print "mut_rate="+str(mut_rate)
    
    datas = [[1,6,16.2], [2, 12, 16.2], [3, 16, 16.31], [4, 20, 16.56], [6, 22, 16.27], [8, 26, 16.56], [10, 27, 16.42], [12, 28, 16.2]]
    ga = iAntGA(pop_size=pop_size, gens=gens, elites=elites, mut_rate=mut_rate, system=system, terminateFlag = terminateFlag)
    start = time.time()
    ga.run_ga(datas)
    stop = time.time()
    print 'It runs '+str((stop-start)/3600.0)+ ' hours...'
