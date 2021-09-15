# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import heapq
import random

# %% [markdown]
# # Generation of initial population of random antibodies

# %%
# m: number of machines
# p: number of parts
# population_size : number of antibody of the initial population
# seed: seed for reproducible results
def generation_initial_population(p, m, population_size, seed):
    MaxCell = min(p,m) #calculation of max number of cells
    number_of_zeros = MaxCell - 1 #number of zeros in each antibody
    antibodies = np.empty(shape=(population_size, p+m+number_of_zeros), dtype=int)
    antibody = np.append(np.array([*range(1,p+m+1)]), np.zeros(number_of_zeros,dtype=int))
    for i in range(0,population_size):
        np.random.seed(seed) 
        np.random.shuffle(antibody) #random positions in the array
        antibodies[i] = antibody
    return antibodies


# %%
antibodies = generation_initial_population(p = 7, m = 5, population_size = 3, seed = 1995)
antibodies

# %% [markdown]
# # Evaluate all existing antibodies and compute their affinities
# %% [markdown]
# Lectura de los datos del problema + Traducción del formato del dataset a matriz trabajo-estación

# %%
def part_machine_incidence_matrix(data):    
    f = open(data,'r')
    lines = [line.split('\n') for line in f]
    f.close()

    m,p = [int(num) for num in lines[8][0].split(' ')] #m: number of machines, p: number of parts

    machines=[[m,p]]
    for i in range(9,9+m):
        #machines.append([int(num) for num in lines[i][0].split(' ')])
        machines.append([int(lines[i][0].split(' ')[j]) for j in range(1,m)])

    columns, rows = ['M'+str(i) for i in range(1,m+1)], ['P'+str(i) for i in range(1,p+1)]
    m_p_matrix = pd.DataFrame(columns= columns, index= rows)

    ones_zeros = []
    for i in range(1,len(machines)):
        aux = []
        for j in range(1,p+1):
            if j in machines[i]: aux.append(1)
            else: aux.append(0)
        ones_zeros.append(aux)

    for i in range(0,len(columns)):
        m_p_matrix[columns[i]] = ones_zeros[i]

    return m_p_matrix, m, p, columns, rows


# %%
matrix, m, p, columns, rows = part_machine_incidence_matrix('data/instances/testset_a/5x7_Waghodekar_Sahu(1984)[Problem-2].txt')
matrix

# %% [markdown]
# ## Decodificación de anticuerpos
# 1er Paso: Separación de celdas.

# %%
def cell_identification(antibodies):
    total_cells = []
    for antibodie in antibodies:
        print("antibodie", antibodie)
        flag = 0 #bandera que indica si el num anterior es un cero
        cells, cell = [],[]
        i = 0
        for num in antibodie:
            if (num == 0): 
                if flag == 0: 
                    i=i+1
                    cells.append(cell)
                    # print(cell)
                    cell = []
                flag = 1
            else: 
                cell.append(num)
                # print(num)
                flag = 0
                if num == antibodie[len(antibodie)-1]:
                    cells.append(cell)
        total_cells.append(cells)
        # print("cells", cells)
    # print("list of all cells for all antibodies",total_cells)
    # print("cells for antibody 1",total_cells[0])
    return total_cells


# %%
total_cells = cell_identification(antibodies = antibodies)
total_cells

# %% [markdown]
# reorganizar filas y columnas de la matriz maquinas-trabajos en función de lo descrito por el anticuerpo

# %%
def decode_cells(total_cells):
    total_machines, total_parts = [], []
    for antibodie in total_cells:
        print("antibodie",antibodie)
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
        print("decoded",antibodie)

    return total_machines, total_parts
    # print(total_machines)
    # print(total_parts)


# %%
total_machines, total_parts = decode_cells(total_cells=total_cells)

# %% [markdown]
# Representación de la matriz: usamos total_machines y total_parts, donde hemos colocado en orden las máquinas y los trabajos respectivamente.
# 

# %%
def create_machine_part_matrix(matrix, antibodies, total_machines, total_parts):
    antibody_matrices = []
    for i in range(0,len(antibodies)):
        antibodie_matrix = matrix.loc[:,total_machines[i]]
        antibodie_matrix = antibodie_matrix.loc[total_parts[i]]
        antibody_matrices.append(antibodie_matrix)
    return antibody_matrices


antibody_matrices = create_machine_part_matrix(matrix=matrix,
                                                antibodies=antibodies, 
                                                total_machines=total_machines, 
                                                total_parts=total_parts)
antibody_matrices


# %%
def evaluate_antibodies(antibody_matrices, total_cells): #m and p should be added as parameters
    exceptions, voids = [],[]
    for i in range(0,len(total_cells)):
        # print("\n",total_cells[i])
        # print(antibody_matrices[i])
        void, exception = 0,0
        for cell in total_cells[i]:
            # print(cell)
            machines, parts = [], []
            for mp in cell:
                if mp[0] == 'M': 
                    machines.append(mp)
                    # print(machines)
                if mp[0] == 'P': 
                    parts.append(mp)
                    # print(parts)
                # else: print('error')
            
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
    # print("voids",voids)
    # print("exceptions",exceptions)
    efficacies = []
    matrix_dimension = m*p
    for i in range(0,len(total_cells)):
        exceptions_ratio = exceptions[i]/matrix_dimension
        voids_ratio = voids[i]/matrix_dimension
        efficacy= (1-exceptions_ratio)/(1+voids_ratio)
        efficacies.append(efficacy)
    return efficacies, voids, exceptions


# %%
efficacies, voids, exceptions = evaluate_antibodies(antibody_matrices = antibody_matrices, total_cells=total_cells)
efficacies


# %%
def calculate_efficacies(total_cells, m, p):
    efficacies = []
    matrix_dimension = m*p
    for i in range(0,len(total_cells)):
        exceptions_ratio = exceptions[i]/matrix_dimension
        voids_ratio = voids[i]/matrix_dimension
        efficacy= (1-exceptions_ratio)/(1+voids_ratio)
        efficacies.append(efficacy)
    return efficacies 


# %%
efficacies = calculate_efficacies(total_cells = total_cells, m = m, p = p)
efficacies

# %% [markdown]
# # Select N% of antibodies with highest affinities & Clone selected antibodies
# %% [markdown]
# con el parámetro p de probabilities de np.random.choice podemos pasar un vector de probabilidades para el sampleo

# %%
def antibodies_selection(antibodies, N):
    positions_antibodies_selected = np.random.choice(len(antibodies), size=(round(len(antibodies)*N)), replace=False)
    antibodies_selected = antibodies[positions_antibodies_selected.tolist()]
    return antibodies_selected, positions_antibodies_selected

cloned_antibodies, positions_antibodies_selected = antibodies_selection(antibodies=antibodies, N=0.66)

print(positions_antibodies_selected)
cloned_antibodies

# %% [markdown]
# # Mutation operator
# %% [markdown]
# Aqui estoy comentiendo un fallo y es que realizo la mutación en todos los casos.
# Se debe comparar la efficacy del anticuerpo antes y después de la mutación y quedarse con el mejor.
# %% [markdown]
# ## Maturate cloned antibodies

# %%
def mutate_cloned_antibodies(cloned_antibodies):
    for antibodie in cloned_antibodies:
        print(antibodie)
        positions = np.random.choice(len(antibodie),size=2,replace=False)
        print(positions)
        antibodie[positions[0]], antibodie[positions[1]] = antibodie[positions[1]], antibodie[positions[0]]
    return cloned_antibodies


# %%
mutate_cloned_antibodies(cloned_antibodies = cloned_antibodies)

# %% [markdown]
# ## Evaluate cloned antibodies

# %%
cloned_total_cells = cell_identification(cloned_antibodies)
cloned_total_cells


# %%
cloned_total_machines, cloned_total_parts = decode_cells(cloned_total_cells)


# %%
cloned_antibody_matrices = create_machine_part_matrix(matrix = matrix,
                                                        antibodies=cloned_antibodies, 
                                                        total_machines = cloned_total_machines, 
                                                        total_parts= cloned_total_parts)
cloned_antibody_matrices


# %%
cloned_efficacies, cloned_voids, cloned_exceptions = evaluate_antibodies(cloned_antibody_matrices, cloned_total_cells)
cloned_efficacies

# %% [markdown]
# ## Add R% of best cloned antibodies to the pool of antibodies
# %% [markdown]
# select the R% of best cloned antibodies and adds them to the pool of antibodies

# %%
def select_best_cloned_antibodies(antibodies, cloned_antibodies, efficacies, cloned_efficacies, R):
    updated_efficacies = 0
    amount_antibodies = round(len(cloned_efficacies)*R)
    print("{} antibodies were added to the antibodies pool".format(amount_antibodies))
    positions = [i
        for x, i
        in heapq.nlargest(
            amount_antibodies,
            ((x, i) for i, x in enumerate(cloned_efficacies)))]
    updated_antibodies_pool = np.concatenate((antibodies, cloned_antibodies[(positions)]), axis = 0)
    cloned_efficacies = np.array(cloned_efficacies)
    updated_efficacies = efficacies + cloned_efficacies[(positions)].tolist()
    return updated_antibodies_pool, updated_efficacies


# %%
antibodies_pool, updated_efficacies = select_best_cloned_antibodies(antibodies = antibodies,
                                                                    cloned_antibodies = cloned_antibodies,
                                                                    efficacies = efficacies, cloned_efficacies = cloned_efficacies, R = 0.5)
print(updated_efficacies)
antibodies_pool

# %% [markdown]
# # Remove worst members of the antibodies pool (RECEPTOR EDITING)

# %%
def receptor_editing(antibodies_pool,updated_efficacies, B):
    amount_antibodies = round(len(updated_efficacies)*B)
    print("Se han borrado {} anticuerpos".format(amount_antibodies))
    positions = [i
        for x, i
        in heapq.nsmallest(
            amount_antibodies,
            ((x, i) for i, x in enumerate(updated_efficacies)))]
    print(positions)
    antibodies_pool = np.delete(antibodies_pool, positions, axis=0)
    return antibodies_pool


# %%
antibodies_pool = receptor_editing(antibodies_pool = antibodies_pool, updated_efficacies = updated_efficacies, B = 0.5)
antibodies_pool

# %% [markdown]
# # New Random antibodies into the population
# %% [markdown]
# # Stopping criteria

# %%


# %% [markdown]
# 

