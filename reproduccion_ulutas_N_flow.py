# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

import numpy as np
import pandas as pd
import heapq

# %%
def evaluate_intercell_transport(antibody_matrices,total_cells,rows):   
    intercell_transports, sum_intercell_transport = [], []
    for i,matrix in enumerate(antibody_matrices): #recorremos todas las matrices de incidencia
        intercell_transport = []
        for part in rows: # recorremos todas las partes para cada matriz
            part_incidence = matrix.loc[part]
            part_incidence_1 = part_incidence[part_incidence == 1]
            cont = 0 # contador para el número de celdas por las que pasa un trabajo
            for cell in total_cells[i]: # recorremos, para cada matriz y parte, las celdas de máquinas
                if any(machine in cell for machine in part_incidence_1.index): cont = cont + 1
            intercell_transport.append(cont)
        intercell_transports.append(intercell_transport)
        sum_intercell_transport.append(sum(intercell_transport))
    return intercell_transports,sum_intercell_transport

# %%
def evaluate_antibodies(antibody_matrices, total_cells, number_of_operations): #m and p should be added as parameters
    # calculation of exceptions, voids and possible penalties of the configuration
    exceptions, voids, penalties = [],[],[]
    for i in range(0,len(total_cells)):
        void, exception, penalty = 0,0,0
        cell_machine_flag, cell_flag_part = 0,0
        for cell in total_cells[i]:
            machines, parts = [], []
            machine_flag, part_flag = 0,0 #flag variables for calculation of penalties
            for mp in cell:
                if mp[0] == 'M': 
                    machines.append(mp)
                    machine_flag = 1
                if mp[0] == 'P': 
                    parts.append(mp)
                    part_flag = 1
            if machine_flag == 0: cell_machine_flag = 1
            if part_flag == 0: cell_flag_part = 1
            for machine in machines:
                for part in antibody_matrices[i].index:
                    if part in parts and antibody_matrices[i][machine][part] == 0: 
                        void = void+1
                    if part not in parts and antibody_matrices[i][machine][part] == 1:
                        exception = exception+1
        voids.append(void)
        exceptions.append(exception)
        penalty = 0.5 * (cell_machine_flag + cell_flag_part)
        penalties.append(penalty)
    # calculation of efficacy thanks to the previous calculation of exceptions, voids and penalties.
    efficacies = []
    for i in range(0,len(total_cells)):
        exceptions_ratio = exceptions[i]/number_of_operations
        voids_ratio = voids[i]/number_of_operations
        efficacy= (1 - exceptions_ratio) / (1 + voids_ratio)
        efficacies.append(efficacy - efficacy * penalties[i])
    return efficacies, voids, exceptions


#%%
#en este caso, como queremos MINIMIZAR sum_intercell_transport hay que tener en cuenta que es distinto a MAXIMIZAR efficacies
def antibodies_selection_v3(antibodies, N, sum_intercell_transport):
    positions_antibodies_selected = []
    if not np.count_nonzero(sum_intercell_transport): #if all sum_intercell_transport are zero, we assign all antibodies the same probability to be selected
        sel_probabilities = [1 for i in range(0,len(sum_intercell_transport))]
    else: 
        sel_probabilities = []
    sum_intercell_transport = pd.Series(sum_intercell_transport).sort_values()
    total_sum_intercell_transport = np.sum(sum_intercell_transport)
    sel_probabilities.append(sum_intercell_transport[0]/total_sum_intercell_transport)
    # print("valor inicial",sel_probabilities[0])
    for i in range(1,len(sum_intercell_transport)):
        sel_probabilities.append(sel_probabilities[i-1] + sum_intercell_transport[i]/np.sum(sum_intercell_transport))
    # print(sel_probabilities)
    for i in range(0,len(sel_probabilities)):
        if N < sel_probabilities[i]: 
            positions_antibodies_selected.append(sum_intercell_transport.index[i])
    antibodies_selected = antibodies[(positions_antibodies_selected)]        
    return antibodies_selected, np.array(positions_antibodies_selected)

#%%
def select_best_cloned_antibodies_v2(antibodies, 
                                    cloned_antibodies, 
                                    efficacies, 
                                    cloned_efficacies,
                                    sum_intercell_transport,
                                    cloned_sum_intercell_transport, 
                                    R, 
                                    positions_antibodies_selected,
                                    log):
    #if antibody efficacy improves after mutation, we keep the mutated antibody, otherwise we dismiss it.
    pool_positions_antibodies_improved, position_antibodies_improved = [], []
    cloned_efficacies = np.array(cloned_efficacies)
    cloned_sum_intercell_transport = np.array(cloned_sum_intercell_transport)
    for i in range(0,len(cloned_antibodies)):
        if cloned_efficacies[i] < efficacies[(positions_antibodies_selected[i])]:
            if log: print("\nclon {} presents worse efficacy while mutating".format(cloned_antibodies[i]))
            if log: print("Mutated clon efficacy",cloned_efficacies[i])
            if log: print("Non-mutated clon efficacy",efficacies[(positions_antibodies_selected[i])])
            if log: print("Keep original clon {}".format(antibodies[(positions_antibodies_selected[i])]))
        else:
            pool_positions_antibodies_improved.append(positions_antibodies_selected[i])
            position_antibodies_improved.append(i)
    if log: print("\nPositions (clone-pool) of antibodies that improved while mutating", position_antibodies_improved)
        
    cloned_antibodies = cloned_antibodies[(position_antibodies_improved)]
    cloned_efficacies = cloned_efficacies[(position_antibodies_improved)]
    cloned_sum_intercell_transport = cloned_sum_intercell_transport[(position_antibodies_improved)]
    amount_selected_antibodies = round(len(pool_positions_antibodies_improved)*R)

    #2nd part of the function: select R% of the best (efficacy) cloned antibodies
    if log: print("\n{} antibodies were selected and updated in the pool".format(amount_selected_antibodies))
    positions = [i #positions of best R% of selected antibodies
        for x, i
        in heapq.nlargest(
            amount_selected_antibodies,
            ((x, i) for i, x in enumerate(cloned_efficacies)))]
    # print(positions)
    if log: print("Positions (pool) of updated antibodies",positions_antibodies_selected[(positions)])
    for i in range(0,amount_selected_antibodies):
        antibodies[(pool_positions_antibodies_improved[(positions[i])])] = cloned_antibodies[(positions[i])]
        efficacies[(pool_positions_antibodies_improved[(positions[i])])] = cloned_efficacies[(positions[i])]
        sum_intercell_transport[(pool_positions_antibodies_improved[(positions[i])])] = cloned_sum_intercell_transport[(positions[i])]
    return antibodies, efficacies, sum_intercell_transport

#%%
#AQUI MODIFICAMOS PARA QUE SE ELIMINEN LOS QUE TIENEN MAYOR TRANSPORTE INTERCELULAR (Nlargest)
def receptor_editing(antibodies_pool,efficacies,sum_intercell_transport, B, log):
    amount_antibodies = round(len(efficacies)*B)
    if log: print("\n{} antibodies were deleted".format(amount_antibodies))
    positions = [i
        for x, i
        in heapq.nlargest(
            amount_antibodies,
            ((x, i) for i, x in enumerate(sum_intercell_transport)))]
    # print(positions)
    antibodies_pool = np.delete(antibodies_pool, positions, axis=0)
    efficacies = np.delete(efficacies, positions, axis=0)
    sum_intercell_transport = np.delete(sum_intercell_transport, positions, axis=0)
    return antibodies_pool, efficacies, sum_intercell_transport, amount_antibodies