# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
from os import listdir
import reproduccion_ulutas as rp
import time


# %%
mypath = 'data/instances/A16_A32'
datasets = [f for f in listdir(mypath)] #dataset reading


# %%
POPULATION_SIZE = 1000   # indicate the initial size of antibodies population
# N is a random % of antibodies with highest affinities selected to be cloned
R = 1 # % of best cloned antibodies to the pool of antibodies
B = 0.1 # worst % of the whole population (RECEPTOR EDITING)

MAX_ITERATIONS = 10000
MAX_ITERATIONS_WITHOUT_IMPROVEMENT = 1000
number_of_runs = 10
log = False # Boolean: print logs at the execution of the algorithm.


# %%
#reads data and translates it into part-machine matrix
problem_names, grouping_efficacies, number_of_iterations, elapsed_time = [],[],[],[]
path = mypath + '/'
for dataset in datasets:
    data = path + dataset
    runs = 0
    while runs < number_of_runs: #number of runs for each instance
        matrix, m, p,number_of_operations, columns, rows = rp.part_machine_incidence_matrix(data)
        if log == True: print("\nPart-machine matrix\n",matrix)
        #GENERATION OF INITIAL POPULATION
        antibodies = rp.generation_initial_population(p = p, m= m, population_size = POPULATION_SIZE)
        if log == True: print("\nInitial antibodies\n",antibodies)
        iterations, best_solution, improvement_counter = 0, 0, 0
        start = time.time() #start time for each problem

        while iterations <= MAX_ITERATIONS:
            if log == True: print("\n Iteration number",iterations+1)
            #identifies cells in the antibodies
            total_cells = rp.cell_identification(antibodies = antibodies)
            if log == True: print("\n Cells\n", total_cells)
            #translates numbers into parts and machines
            total_machines, total_parts = rp.decode_cells(total_cells = total_cells, rows= rows, columns = columns, p = p, log = log) 

            #calculates part-machine matrix for each antibody
            antibody_matrices = rp.create_machine_part_matrix(matrix = matrix, 
                                                            antibodies = antibodies, 
                                                            total_machines = total_machines, 
                                                            total_parts = total_parts)
            #evaluates efficacie of each antibody
            efficacies, affinities, voids, exceptions = rp.evaluate_antibodies(antibody_matrices = antibody_matrices, 
                                                                                total_cells = total_cells,
                                                                                number_of_operations = number_of_operations)
            if log == True: print("\n Efficacies",efficacies)
            if log == True: print("\n Affinities",affinities)

            # SELECTS N% OF ANTIBODIES WITH HIGHES ANTIBODIES (SELECTION ROULETTE WHEEL)
            N = np.random.rand(1) # N is a random number each iteration
            if log: print("N% =",N[0])
            cloned_antibodies, positions_antibodies_selected = rp.antibodies_selection(antibodies=antibodies, N=N[0], affinities = affinities)
            if log == True: print("\n Cloned antibodies\n",cloned_antibodies)
            if log == True: print("\n Cloned antibodies positions (pool)\n",positions_antibodies_selected)
            # MUTATION
            rp.mutate_cloned_antibodies(cloned_antibodies = cloned_antibodies, log= log)
            if log == True: print("\n Mutation of cloned antibodies\n",cloned_antibodies)
            #CLONES: identifies cells in the antibody
            cloned_total_cells = rp.cell_identification(cloned_antibodies)
            #CLONES: translates numbers into parts and machines
            cloned_total_machines, cloned_total_parts = rp.decode_cells(total_cells = cloned_total_cells,rows= rows, columns = columns, p = p, log = log)
            #CLONES: calculates part-machine matrix for each antibody
            cloned_antibody_matrices = rp.create_machine_part_matrix(matrix = matrix,
                                                                        antibodies = cloned_antibodies, 
                                                                        total_machines = cloned_total_machines, 
                                                                        total_parts = cloned_total_parts)                                                          
            #CLONES evaluates efficacie of each antibody
            cloned_efficacies, cloned_affinities, cloned_voids, cloned_exceptions = rp.evaluate_antibodies(cloned_antibody_matrices,
                                                                                                            cloned_total_cells,
                                                                                                            number_of_operations)
            if log == True: print("\n Cloned Efficacies",cloned_efficacies)
            if log == True: print("\n Cloned Affinities",cloned_affinities)
            # Add R% of best cloned antibodies to the pool of antibodies
            antibodies, efficacies, affinities = rp.select_best_cloned_antibodies_v2(antibodies = antibodies,
                                                cloned_antibodies = cloned_antibodies,
                                                efficacies = efficacies, 
                                                cloned_efficacies = cloned_efficacies,
                                                affinities = affinities, 
                                                cloned_affinities = cloned_affinities,
                                                R = R,
                                                positions_antibodies_selected = positions_antibodies_selected, 
                                                log = log)
            if log == True: print("\n New antibodies pool \n",antibodies)                                                                        
            # RECEPTOR EDITING: Remove worst members of the antibodies pool
            antibodies, efficacies, affinities, amount_antibodies_erased = rp.receptor_editing(antibodies_pool = antibodies, 
                                                                                        efficacies = efficacies, 
                                                                                        affinities=affinities, 
                                                                                        B = B, 
                                                                                        log = log)
            if log == True: print("\n Antibodies pool after removal \n", antibodies)
            #GENERATION OF NEW RANDOM ANTIBODIES
            number_new_random_antibodies = amount_antibodies_erased
            new_random_antibodies = rp.generation_initial_population(p = p, 
                                                                m = m, 
                                                                population_size = number_new_random_antibodies)
            if log == True: print("\n New random antibodies \n",new_random_antibodies)                                                        
            antibodies = np.concatenate((antibodies, new_random_antibodies), axis = 0)
            if log == True: print("\n Antibodies pool after adding new random antibodies \n",antibodies)
            
            #TERMINATION CRITERIA AND SELECTION OF BEST SOLUTION
            iteration_best_solution = np.amax(efficacies) #max value efficacy for this iteration
            index_best_solution = np.argmax(efficacies)
            best_antibody = antibodies[index_best_solution]
            if log: print("\n Best solution at the moment is {} (antibody {}, index {})".format(iteration_best_solution,best_antibody,index_best_solution))
            if iteration_best_solution <= best_solution:
                improvement_counter = improvement_counter + 1
                if improvement_counter >= MAX_ITERATIONS_WITHOUT_IMPROVEMENT:
                    if log: print("Maximun number of iterations without improvement reached: {}".format(MAX_ITERATIONS_WITHOUT_IMPROVEMENT))
                    if log: print("Best solution obtained is {}".format(best_solution))
                    break
            else:
                best_solution = iteration_best_solution
                improvement_counter = 0
            #update iteration counter
            iterations = iterations + 1
        runs = runs + 1
        end = time.time()
        #saving results:
        problem_names.append(dataset)
        grouping_efficacies.append(iteration_best_solution)
        number_of_iterations.append(iterations)
        elapsed_time.append(end-start)
        print(dataset, "\tEficcacy: ",iteration_best_solution, "\tNumber of iterations:", iterations, "Elapsed time", end-start)
        
results = {"Problem": problem_names,
            "Efficacy": grouping_efficacies,
            "Iterations": number_of_iterations,
            "Elapsed Time": elapsed_time}
results = pd.DataFrame(results)
results.to_csv('results1.4.csv')


