'''

    Example 1 on using WindFLO library

  Purpose:
      
      This example demonstrate how to use the Python WindFLO API for serial optimization 
      of wind farm layout. It uses WindFLO to analyze each layout configuration
      and a the particle swarm optimization (PSO) algorithm from the pyswarm package
      to perform the optimization.  The layout is optimized for maximum power generation
      and incorporates constraints on the minimum allowable clearance between turbines. 
  
      IMPORTANT: The PSO minimizes the fitness function, therefore the negative of the
                 farm power is returned by the EvaluateFarm function
  
  Licensing:
  
    This code is distributed under the Apache License 2.0 
    
  Author:
      Sohail R. Reddy
      sredd001@fiu.edu
      
'''


import sys
# Append the path to the API directory where WindFLO.py is
sys.path.append('API/')
# Append the path to the Optimizers directory where pso.py is
sys.path.append('Optimizers/')


import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg

from pso import pso
from WindFLO import WindFLO
import cma

###############################################################################
#    WindFLO Settings and Params
nTurbines = 25            # Number of turbines
libPath = 'release/'        # Path to the shared library libWindFLO.so
inputFile = 'Examples/Example1/WindFLO.dat'    # Input file to read
turbineFile = 'Examples/Example1/V90-3MW.dat'    # Turbine parameters
terrainfile = 'Examples/Example1/terrain.dat'    # Terrain file
diameter = 90.0            # Diameter to compute clearance constraint

windFLO = WindFLO(inputFile = inputFile, nTurbines = nTurbines, libDir = libPath, 
          turbineFile = turbineFile,
          terrainfile = terrainfile)
windFLO.terrainmodel = 'IDW'    # change the default terrain model from RBF to IDW




# Function to evaluate the farm's performance
def EvaluateFarm(x):

    k = 0
    for i in range(0, nTurbines):
        for j in range(0, 2):
               # unroll the variable vector 'x' and assign it to turbine positions
            windFLO.turbines[i].position[j] = x[k]
            k = k + 1
    # Run WindFLO analysis
    windFLO.run(clean = True)    
    
    # Return the farm power or any other farm output
      # NOTE: The negative value is returns since PSO minimizes the fitness value
    return -windFLO.farmPower



# Compute the minimum clearance between turbines in farm must be greater 
# than turbine diameter. The constraint to satisfy is g(x) >= 0
def ComputeClearance(x):

    position = np.zeros((nTurbines,2))
    k = 0
    for i in range(0, windFLO.nTurbines):
        for j in range(0, 2):
               # unroll the variable vector 'x' and assign it to turbine positions        
            position[i,j] = x[k]
            k = k + 1

    minClearence = 10000000
    for i in range(0, nTurbines):
        for j in range(i+1, nTurbines):
            # get the minimum clearance between turbines in the farm
            minClearence = min(minClearence, linalg.norm( position[i,0:2] - position[j,0:2] ))
            
    # if minClearence < diameter, then g(x) < 0 and the constraint is violated
    # if minClearence >= 0 then g(x) >= 0 and the constraint is satisfied            
    return (minClearence - diameter ) # g(x) = minClearence - diameter



# Main function
if __name__ == "__main__":

    # Two variable per turbines (its x and y coordinates)
    lbound = np.zeros(nTurbines*2)    #lower bounds of x and y of turbines
    ubound = np.ones(nTurbines*2)*2000    #upper bounds of x and y of turbines

    def transform_to_problem_dim(x):
        return lbound + x*(ubound - lbound)

    seed = 20
    maxfevals = 5000
    np.random.seed(seed)
    es = cma.CMAEvolutionStrategy(np.random.random(nTurbines*2), 0.33, inopts={'bounds': [0, 1],'seed':seed,'maxiter':1e9, 'maxfevals':maxfevals})
    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [EvaluateFarm(transform_to_problem_dim(x)) for x in solutions])
        es.logger.add()  # write data to disc to be plotted
        print("---------")
        print("Funcion objetivo: ", es.result.fbest)
        # print("Mejor turbina so far: ", transformar_turb_params(es.result.xbest, blade_number))
        print("Evaluaciones funcion objetivo: ", es.result.evaluations)
        # print("Tiempo: ", sw.get_time())
        print("---------")
        es.disp()
    es.result_pretty()


    xBest = transform_to_problem_dim(es.result.xbest)
    bestPower = es.result.fbest

    windFLO.run(clean = True, resFile = 'WindFLO.res')
    
    # Plot the optimum configuration    
    fig = plt.figure(figsize=(8,5), edgecolor = 'gray', linewidth = 2)
    ax = windFLO.plotWindFLO2D(fig, plotVariable = 'P', scale = 1.0e-3, title = 'P [kW]')
    windFLO.annotatePlot(ax)
    plt.savefig("result_fig.pdf")
    
    # save the optimum to a file
    np.savetxt('optimum.dat', xBest)

