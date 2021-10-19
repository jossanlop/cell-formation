# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import heapq

# %%
# m: number of machines
# p: number of parts
# population_size : number of antibody of the initial population
# seed: seed for reproducible results
def generation_initial_population(p, m, population_size):
    MaxCell = min(p,m) #calculation of max number of cells
    number_of_zeros = MaxCell - 1 #number of zeros in each antibody
    antibodies = np.empty(shape=(population_size, p+m+number_of_zeros), dtype=int)
    antibody = np.append(np.array([*range(1,p+m+1)]), np.zeros(number_of_zeros,dtype=int))
    for i in range(0,population_size):
        # np.random.seed(seed) 
        np.random.shuffle(antibody) #random positions in the array
        antibodies[i] = antibody
    return antibodies

# %%
def part_machine_incidence_matrix(data):    
    f = open(data,'r')
    lines = [line.split('\n') for line in f]
    f.close()

    m,p = [int(num) for num in lines[8][0].split(' ')] #m: number of machines, p: number of parts

    machines=[] # matrix with representation of data
    for i in range(9,9+m):
        machine_line = []
        j = 0 #contador para saltar primera columna correspondiente a la máquina y no a la incidencia
        for operation in lines[i][0].split(' '):
            if operation != '' and j>0: machine_line.append(int(operation))
            j = j +1
        machines.append(machine_line)
    columns, rows = ['M'+str(i) for i in range(1,m+1)], ['P'+str(i) for i in range(1,p+1)]
    m_p_matrix = pd.DataFrame(columns= columns, index= rows)
    ones_zeros = [] #ones and zeros vector
    number_of_operations = 0 #number of 'ones' in the part-machine matrix
    for i in range(0,len(machines)):
        aux = []
        for j in range(1,p+1):
            if j in machines[i]: 
                aux.append(1)
                number_of_operations = number_of_operations +1
            else: aux.append(0)
        ones_zeros.append(aux) 
    for i in range(0,len(columns)):
        m_p_matrix[columns[i]] = ones_zeros[i]

    return m_p_matrix, m, p, number_of_operations, columns, rows


# %%
def cell_identification(antibodies):
    total_cells = []
    for antibodie in antibodies:
        flag = 1 #bandera que indica si el num anterior es un cero
        cells, cell = [],[]
        for num in antibodie:
            if (num == 0): 
                if flag == 0:
                    cells.append(cell)
                    cell = []
                flag = 1
            else: 
                cell.append(num)
                flag = 0
                if num == antibodie[len(antibodie)-1]:
                    cells.append(cell)
        total_cells.append(cells)
    # print("list of all cells for all antibodies",total_cells)
    # print("cells for antibody 1",total_cells[0])
    return total_cells

# %%
def decode_cells(total_cells, rows, columns, p, log):
    total_machines, total_parts = [], []
    for antibodie in total_cells:
        # print("antibodie",antibodie)
        machines = []
        parts = []
        decoded_antibodie = antibodie
        for i in range(0,len(antibodie)):
            # print(antibodie[i])
            for j in range(0,len(antibodie[i])):
                if antibodie[i][j] <= p: 
                    parts.append(rows[antibodie[i][j]-1])
                    decoded_antibodie[i][j] = rows[antibodie[i][j]-1]
                else: 
                    machines.append(columns[antibodie[i][j]-p-1])
                    decoded_antibodie[i][j] = columns[antibodie[i][j]-p-1]
            antibodie = decoded_antibodie
        total_machines.append(machines)
        total_parts.append(parts)
        if log: print("decoded antibodies",antibodie)

    return total_machines, total_parts
    # print(total_machines)
    # print(total_parts)

# %%
def create_machine_part_matrix(matrix, antibodies, total_machines, total_parts):
    antibody_matrices = []
    for i in range(0,len(antibodies)):
        antibodie_matrix = matrix.loc[:,total_machines[i]]
        antibodie_matrix = antibodie_matrix.loc[total_parts[i]]
        antibody_matrices.append(antibodie_matrix)
    return antibody_matrices


# %%
def evaluate_antibodies(antibody_matrices, total_cells, number_of_operations): #m and p should be added as parameters
    exceptions, voids, penalties = [],[],[]
    for i in range(0,len(total_cells)):
        # print("\n",total_cells[i])
        # print(antibody_matrices[i])
        void, exception, penalty = 0,0,0
        cell_machine_flag, cell_flag_part = 0,0
        for cell in total_cells[i]:
            # print(cell)
            machines, parts = [], []
            machine_flag, part_flag = 0,0 #flag variables for calculation of penalties
            for mp in cell:
                if mp[0] == 'M': 
                    machines.append(mp)
                    machine_flag = 1
                    # print(machines)
                if mp[0] == 'P': 
                    parts.append(mp)
                    part_flag = 1
                    # print(parts)
            if machine_flag == 0: cell_machine_flag = 1
            if part_flag == 0: cell_flag_part = 1
            # print(i, cell_machine_flag, cell_flag_part)
            for machine in machines:
                for part in antibody_matrices[i].index:
                    if part in parts and antibody_matrices[i][machine][part] == 0: 
                        void = void+1
                        # print("void",machine, part)
                    if part not in parts and antibody_matrices[i][machine][part] == 1:
                        exception = exception+1
                        # print("exception",machine, part)
            # print(void, exception)
            # print("\n")
        voids.append(void)
        exceptions.append(exception)
        penalty = 0.5 * (cell_machine_flag + cell_flag_part)
        # print(penalty)
        penalties.append(penalty)
    # print("voids",voids)
    # print("exceptions",exceptions)
    efficacies, affinities = [], []
    # matrix_dimension = m*p
    for i in range(0,len(total_cells)):
        exceptions_ratio = exceptions[i]/number_of_operations
        voids_ratio = voids[i]/number_of_operations
        efficacy= (1-exceptions_ratio)/(1+voids_ratio)
        # efficacies.append(efficacy)
        efficacies.append(efficacy - efficacy * penalties[i])
        affinities.append(efficacy - efficacy * penalties[i])
    return efficacies, affinities, voids, exceptions


# %%
def antibodies_selection(antibodies, N, affinities):
    population_pool = len(antibodies)
    # special treatment if all affinities are 0:
    if np.count_nonzero(affinities): #if there are non-zero values, we calculate normally de selection probabilities
        sel_probabilities = affinities/np.sum(affinities) #selection probabilities
    else: #if all affinity values are zero, then we set equal probabilities to the whole population.
        sel_probabilities = np.ones(len(affinities))
        # print(sel_probabilities)
        sel_probabilities = sel_probabilities/np.sum(sel_probabilities)
        # print(sel_probabilities)

    size = round(population_pool*N)
    nonzero_affinities = np.count_nonzero(sel_probabilities)
    if nonzero_affinities < size: # if there are more antibodies to be selected than affinities different from zero
        # print("MORE antibodies to be selected than affinities different from zero")
        # print(sel_probabilities)
        # print("number of antibodies to be selected",type(nonzero_affinities))
        # print(nonzero_affinities)
        positions_antibodies_selected = np.random.choice(population_pool, size=int(nonzero_affinities), replace=False, p = sel_probabilities)
        sel_probabilities[:] = 1
        sel_probabilities[positions_antibodies_selected] = 0
        sel_probabilities = sel_probabilities/np.sum(sel_probabilities)
        # print("new probabilities",sel_probabilities)
        add_number_of_antibodies = size - nonzero_affinities
        add_positions_antibodies_selected = np.random.choice(population_pool, size=int(add_number_of_antibodies), replace=False, p = sel_probabilities)
        positions_antibodies_selected = np.concatenate((positions_antibodies_selected, add_positions_antibodies_selected), axis = 0)
        antibodies_selected = antibodies[positions_antibodies_selected.tolist()]
        # return print("¡¡ERROR!! en la selección inicial de anticuerpos. INSUFICIENTES ANTICUERPOS CON AFINIDAD > 1")
        return antibodies_selected, positions_antibodies_selected
    else:
        # print("LESS antibodies to be selected than affinities different from zero")
        # print(sel_probabilities)
        # print(size)
        # print(type(size))
        positions_antibodies_selected = np.random.choice(population_pool, size=int(size), replace=False, p = sel_probabilities)
        antibodies_selected = antibodies[positions_antibodies_selected.tolist()]
        return antibodies_selected, positions_antibodies_selected

# %%
def antibodies_selection_v2(antibodies, N, affinities):
    positions_antibodies_selected = []
    if np.count_nonzero(affinities):
        sel_probabilities = affinities/np.sum(affinities) #selection probabilities
    else:
        sel_probabilities = np.ones(len(affinities))
        sel_probabilities = sel_probabilities/np.sum(sel_probabilities)
    # print(sel_probabilities)
    for i in range(0,len(sel_probabilities)):
        if N < sel_probabilities[i]: 
                positions_antibodies_selected.append(i)
    antibodies_selected = antibodies[(positions_antibodies_selected)]        
    return antibodies_selected, positions_antibodies_selected


# %%
def antibodies_selection_v3(antibodies, N, affinities):
    # print(affinities)
    positions_antibodies_selected = []
    if not np.count_nonzero(affinities): 
        # sel_probabilities = np.ones(len(affinities)) #solventamos el caso en el que todas las afinidades sean 0
        sel_probabilities = [1 for i in range(0,len(affinities))]
    else: 
        sel_probabilities = []
    affinities = pd.Series(affinities).sort_values()
    # print(affinities)
    total_affinity = np.sum(affinities)
    # print("tota_affinity", total_affinity)
    sel_probabilities.append(affinities[0]/total_affinity)
    # print("valor inicial",sel_probabilities[0])
    for i in range(1,len(affinities)):
        sel_probabilities.append(sel_probabilities[i-1] + affinities[i]/np.sum(affinities))
    # print(sel_probabilities)
    for i in range(0,len(sel_probabilities)):
        if N < sel_probabilities[i]: 
            positions_antibodies_selected.append(affinities.index[i])
    antibodies_selected = antibodies[(positions_antibodies_selected)]        
    return antibodies_selected, np.array(positions_antibodies_selected)

# %%
def mutate_cloned_antibodies(cloned_antibodies, log):
    for antibodie in cloned_antibodies:
        # print(antibodie)
        positions = np.random.choice(len(antibodie),size=2,replace=False)
        # print(positions)
        antibodie[positions[0]], antibodie[positions[1]] = antibodie[positions[1]], antibodie[positions[0]]
    if log: return cloned_antibodies
    else: return

# %%
def select_best_cloned_antibodies(antibodies, cloned_antibodies, efficacies, 
                                    cloned_efficacies,
                                    affinities,
                                    cloned_affinities, 
                                    R, 
                                    positions_antibodies_selected,
                                    log):
    #if antibody efficacy improves after mutation, we keep the mutated antibody, otherwise we dismiss it.
    for i in range(0,len(cloned_antibodies)):
        if cloned_efficacies[i] < efficacies[(positions_antibodies_selected[i])]:
            if log: print("\nclon {} presenta peor eficacia al ser mutado".format(cloned_antibodies[i]))
            if log: print("eficacia clon mutado",cloned_efficacies[i])
            if log: print("eficacia clon sin mutar",efficacies[(positions_antibodies_selected[i])])
            cloned_antibodies[i] = antibodies[(positions_antibodies_selected[i])]
            cloned_efficacies[i] = efficacies[(positions_antibodies_selected[i])]
            cloned_affinities[i] = affinities[(positions_antibodies_selected[i])]
            # print("Conservamos el clon {} original".format(cloned_antibodies[i]))

    # second part of the function: select R% of the best (efficacy) cloned antibodies
    amount_selected_antibodies = round(len(cloned_efficacies)*R)
    if log: print("\n{} antibodies were selected and updated in the pool".format(amount_selected_antibodies))
    positions = [i #positions of best R% of selected antibodies
        for x, i
        in heapq.nlargest(
            amount_selected_antibodies,
            ((x, i) for i, x in enumerate(cloned_efficacies)))]
    # print(positions)
    if log: print("Positions in the pool of updated antibodies",positions_antibodies_selected[(positions)])
    for i in range(0,amount_selected_antibodies):
        antibodies[(positions_antibodies_selected[(positions[i])])] = cloned_antibodies[(positions[i])]
        efficacies[(positions_antibodies_selected[(positions[i])])] = cloned_efficacies[(positions[i])]
        affinities[(positions_antibodies_selected[(positions[i])])] = cloned_affinities[(positions[i])]
    return antibodies, efficacies, affinities


# %%
# #10% de los que mejoran al mutar
def select_best_cloned_antibodies_v2(antibodies, cloned_antibodies, efficacies, 
                                    cloned_efficacies,
                                    affinities,
                                    cloned_affinities, 
                                    R, 
                                    positions_antibodies_selected,
                                    log):
    #if antibody efficacy improves after mutation, we keep the mutated antibody, otherwise we dismiss it.
    pool_positions_antibodies_improved, position_antibodies_improved = [], []
    cloned_efficacies = np.array(cloned_efficacies)
    cloned_affinities = np.array(cloned_affinities)
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
    cloned_affinities = cloned_affinities[(position_antibodies_improved)]
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
        affinities[(pool_positions_antibodies_improved[(positions[i])])] = cloned_affinities[(positions[i])]
    return antibodies, efficacies, affinities


# %%
def receptor_editing(antibodies_pool,efficacies,affinities, B, log):
    amount_antibodies = round(len(efficacies)*B)
    if log: print("\n{} antibodies were deleted".format(amount_antibodies))
    positions = [i
        for x, i
        in heapq.nsmallest(
            amount_antibodies,
            ((x, i) for i, x in enumerate(efficacies)))]
    # print(positions)
    antibodies_pool = np.delete(antibodies_pool, positions, axis=0)
    efficacies = np.delete(efficacies, positions, axis=0)
    affinities = np.delete(affinities, positions, axis=0)
    return antibodies_pool, efficacies, affinities, amount_antibodies
