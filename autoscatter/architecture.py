import numpy as np
import jax
import jax.numpy as jnp
import tqdm
from itertools import product

from autoscatter.constraints import Constraint_coupling_zero, Constraint_coupling_phase_zero

# def matrix1_subgraph_to_matrix2(matrix1, matrix2):
#     return jnp.sum((matrix2 - matrix1) < 0) == 0

# matrix1_subgraph_to_matrix2 = jax.jit(matrix1_subgraph_to_matrix2)

NO_COUPLING = 0
DETUNING = 1
COUPLING_WITHOUT_PHASE = 1
COUPLING_WITH_PHASE = 2

def find_min_number_of_pumps(triu_matrix):
    '''
    Identifies the minimum number of parametric pumps required to realise this graph
    To do so, we label all modes. If two modes have the same label, they are operated at the same frequency, otherwise they are not.
    To find the minimum number of pumps we iterate over all possible labeling combinations. 
    '''

    def check_redundancy(labels):
        # returns False for combinations, like [0,0,2,3,4] as they are identical with [0,0,1,2,3]
        unique_labels = np.unique(labels)
        if np.max(unique_labels) >= len(unique_labels):
            return False

        # returns False for combinations like [0,0,0,2,1] as they are identical with [0,0,0,1,2]
        for idx in range(1, len(labels)):
            if labels[idx] > np.max(labels[:idx])+1:
                return False

        return True
    
    upper_triu_adjacency_matrix = triu_to_upper_triu_adjacency_matrix(triu_matrix)
    num_modes = upper_triu_adjacency_matrix.shape[0]
    
    counted_pumps = []
    possible_labels = np.arange(num_modes)

    possible_labels = []
    for idx in range(num_modes):
        possible_labels.append(np.arange(idx+1))

    labels_results = []
    for labels in product(*possible_labels):
        labels = np.array(labels)
        if check_redundancy(labels):
            result = count_pumps(labels, upper_triu_adjacency_matrix=upper_triu_adjacency_matrix)
            if result is not None:
                counted_pumps.append(result)
                labels_results.append(labels)

    counted_pumps = np.array(counted_pumps)
    labels_results = np.array(labels_results)

    num_min_pumps = np.min(counted_pumps)
    idxs_min = counted_pumps == num_min_pumps

    return num_min_pumps, labels_results[idxs_min]

def triu_to_upper_triu_adjacency_matrix(triu_matrix):
    '''
    transform a graph in list form to its adjacency matrix, where the lower triangle is set to zero
    the full adjacency matrix would be provided by the function triu_to_adjacency_matrix
    '''
    size_upper_triangle_matrix = len(triu_matrix)
    num_modes = int((-1 + np.sqrt(1+4*2*size_upper_triangle_matrix))//2)
    idxs_upper_triangle = np.array(np.triu_indices(num_modes))
    adjacency_matrix = np.zeros([num_modes, num_modes], dtype='int8')
    for idxs, value in zip(idxs_upper_triangle.T, triu_matrix):
        adjacency_matrix[idxs[0], idxs[1]] = value
    return adjacency_matrix

def count_pumps(labels, triu_matrix=None, upper_triu_adjacency_matrix=None):
    '''
    Counts the number of parametric pumps given a certain graph and labels

    labels: list of integers equalling the number of modes. If two modes have the same label (the same integer), the are operated at the same frequency. Modes with different labels are operated at different frequencies
    triu_matrix: defines the graph as a list containing the elements of the upper triangular matrix of the adjacency matrix
    upper_triu_adjacency_matrix: adjacency matrix, where the lower triangle is set to zero. Passing an actual adjacency matrix will result in a miscounting!

    If both triu_matrix and upper_triu_adjacency_matrix are passed, the function will use triu_matrix
    '''
    if triu_matrix is not None:
        upper_triu_adjacency_matrix = triu_to_upper_triu_adjacency_matrix(triu_matrix)

    # if a coupling is either a complex-valued beamsplitter interaction or a squeezing interaction the labels have to be different between the modes connected by the correspond edge
    # this loop tests if this is the case, if not the function returns None as the labeling is not valid
    idxs1, idxs2 = np.where(upper_triu_adjacency_matrix==COUPLING_WITH_PHASE)
    for idx1, idx2 in zip(idxs1, idxs2):
        if labels[idx1] == labels[idx2]:
            return None
    
    # Now we count the pumps. We sum over all edges. If an edge connects two modes with a different label, a pump is required, otherwise not.
    num_pumps = 0
    idxs1, idxs2 = np.where(upper_triu_adjacency_matrix!=NO_COUPLING)
    for idx1, idx2 in zip(idxs1, idxs2):
        if labels[idx1] != labels[idx2]:
            num_pumps += 1

    return num_pumps

def adjacency_to_triu_matrix(adjacency_matrix):
    num_modes = len(adjacency_matrix)
    triu_indices = np.triu_indices(num_modes)
    return adjacency_matrix[triu_indices]

def triu_to_adjacency_matrix(triu_matrix):
    size_upper_triangle_matrix = len(triu_matrix)
    num_modes = int((-1 + np.sqrt(1+4*2*size_upper_triangle_matrix))//2)
    idxs_upper_triangle = np.array(np.triu_indices(num_modes))
    adjacency_matrix = np.zeros([num_modes, num_modes], dtype='int8')
    for idxs, value in zip(idxs_upper_triangle.T, triu_matrix):
        adjacency_matrix[idxs[0], idxs[1]] = adjacency_matrix[idxs[1], idxs[0]] = value
    return adjacency_matrix

def characterize_architectures(list_of_architectures):
    
    list_of_detunings = []
    list_of_real_valued_couplings = []
    num_complex_couplings_and_squeezings = []
    minimal_number_of_pumps = []
    for arch_idx in tqdm.trange(len(list_of_architectures)):
        arch = list_of_architectures[arch_idx]
        num_detunings, num_passive, num_active = characterize_architecture(arch)
        list_of_detunings.append(num_detunings)
        list_of_real_valued_couplings.append(num_passive)
        num_complex_couplings_and_squeezings.append(num_active)
        min_number, _ = find_min_number_of_pumps(arch)
        minimal_number_of_pumps.append(min_number)
    info_dict = {
        'num_detunings': np.asarray(list_of_detunings),
        'num_real_couplings': np.asarray(list_of_real_valued_couplings),
        'num_complex_couplings_and_squeezings': np.asarray(num_complex_couplings_and_squeezings),
        'num_couplings': np.asarray(list_of_real_valued_couplings)+np.asarray(num_complex_couplings_and_squeezings),
        'minimal_number_of_pumps': np.asarray(minimal_number_of_pumps)
    }
    return info_dict

def characterize_architecture(arch):
    size_upper_triangle_matrix = len(arch)
    num_modes = int((-1 + np.sqrt(1+4*2*size_upper_triangle_matrix))//2)

    idxs_upper_triangle = np.array(np.triu_indices(num_modes))

    num_detunings = 0
    num_real_valued_couplings = 0
    num_complex_valued_couplings_and_squeezing = 0
    for idx1, idx2, val_coupling in zip(idxs_upper_triangle[0], idxs_upper_triangle[1], arch):
        if val_coupling != NO_COUPLING:
            if idx1 == idx2:
                num_detunings += 1
            elif val_coupling == COUPLING_WITHOUT_PHASE:
                num_real_valued_couplings += 1
            else:
                num_complex_valued_couplings_and_squeezing += 1
    return num_detunings, num_real_valued_couplings, num_complex_valued_couplings_and_squeezing

def translate_graph_to_conditions(graph):
    num_modes = graph.shape[0]
    conditions = []
    for idx in range(num_modes):
        if graph[idx,idx] == NO_COUPLING:
            conditions.append(Constraint_coupling_zero(idx,idx))
        elif graph[idx,idx] == DETUNING:
            pass
        else:
            raise NotImplementedError()
        
    for idx2 in range(num_modes):
        for idx1 in range(idx2):
            if graph[idx1,idx2] == NO_COUPLING:
                conditions.append(Constraint_coupling_zero(idx1,idx2))
            elif graph[idx1,idx2] == COUPLING_WITHOUT_PHASE:
                conditions.append(Constraint_coupling_phase_zero(idx1,idx2))
            elif graph[idx1,idx2] == COUPLING_WITH_PHASE:
                pass
            else:
                raise NotImplementedError()
    
    return conditions

def translate_upper_triangle_coupling_matrix_to_conditions(coupling_matrix_upper_triangle):
    num_couplings = len(coupling_matrix_upper_triangle)
    num_modes = int((-1 + np.sqrt(1+4*2*num_couplings))//2)

    idxs_upper_triangle = np.array(np.triu_indices(num_modes))
    conditions = []
    for idx1, idx2, val_coupling in zip(idxs_upper_triangle[0], idxs_upper_triangle[1], coupling_matrix_upper_triangle):
        if idx1 == idx2:
            if val_coupling == NO_COUPLING:
                conditions.append(Constraint_coupling_zero(idx1,idx2))
            elif val_coupling == DETUNING:
                pass
            else:
                raise NotImplementedError()
        else:
            if val_coupling == NO_COUPLING:
                conditions.append(Constraint_coupling_zero(idx1,idx2))
            elif val_coupling == COUPLING_WITHOUT_PHASE:
                conditions.append(Constraint_coupling_phase_zero(idx1,idx2))
            elif val_coupling == COUPLING_WITH_PHASE:
                pass
            else:
                raise NotImplementedError()
    
    return conditions

def translate_conditions_to_upper_triangle_coupling_matrix(conditions, num_modes):
    coupling_matrix_upper_triangle = np.zeros((num_modes**2 + num_modes)//2)

    idx_counter = 0
    for idx1, idx2 in np.array(np.triu_indices(num_modes)).T:
        
        if Constraint_coupling_zero(idx1, idx2) in conditions:
            coupling_matrix_upper_triangle[idx_counter] = NO_COUPLING
        elif idx1 != idx2 and Constraint_coupling_phase_zero(idx1, idx2) in conditions:
            coupling_matrix_upper_triangle[idx_counter] = COUPLING_WITHOUT_PHASE
        else:
            if idx1 != idx2:
                coupling_matrix_upper_triangle[idx_counter] = COUPLING_WITH_PHASE
            else:
                coupling_matrix_upper_triangle[idx_counter] = DETUNING
        idx_counter += 1

    return coupling_matrix_upper_triangle


def fill_coupling_matrix(num_modes, detunings, couplings_without_phase, couplings_with_phase):
    detunings_array = np.asarray(detunings)
    couplings_with_phase_array = np.asarray(couplings_with_phase)
    couplings_without_phase_array = np.asarray(couplings_without_phase)

    if len(couplings_with_phase_array) > 0:
        idxs1 = couplings_with_phase_array[:,0]
        idxs2 = couplings_with_phase_array[:,1]
        couplings_with_phase_idxs = (np.concatenate((idxs1, idxs2)), np.concatenate((idxs2, idxs1)))
    else:
        couplings_with_phase_idxs = ([],[])

    if len(couplings_without_phase_array) > 0:
        idxs1 = couplings_without_phase_array[:,0]
        idxs2 = couplings_without_phase_array[:,1]
        couplings_without_phase_idxs = (np.concatenate((idxs1, idxs2)), np.concatenate((idxs2, idxs1)))
    else:
        couplings_without_phase_idxs = ([],[])

    if len(detunings_array) > 0:
        detunings_idxs = (detunings_array, detunings_array)
    else:
        detunings_idxs = ([],[])

    coupling_matrix = np.zeros([num_modes, num_modes], dtype='int')

    coupling_matrix[detunings_idxs] = DETUNING
    coupling_matrix[couplings_with_phase_idxs] = COUPLING_WITH_PHASE
    coupling_matrix[couplings_without_phase_idxs] = COUPLING_WITHOUT_PHASE

    return jnp.array(coupling_matrix)

class Architecture():
    def __init__(self, num_modes=None, detunings=[], couplings_without_phase=[], couplings_with_phase=[], permutation_rules=None, coupling_matrix=None):
        
        if coupling_matrix is not None:
            self.coupling_matrix = coupling_matrix
            self.num_modes = coupling_matrix.shape[0]
        else:
            self.num_modes = num_modes
            self.detunings = detunings
            self.couplings_with_phase = couplings_with_phase
            self.couplings_without_phase = couplings_without_phase
            self.coupling_matrix = fill_coupling_matrix(num_modes, detunings, couplings_without_phase, couplings_with_phase)
        
        self.permutation_rules = permutation_rules
        if permutation_rules is not None:
            raise NotImplementedError()
        
    def is_subgraph_to(self, arch):
        # checks if self (or one of its isomorphic versions) is a subgraph of arch
        # return matrix1_subgraph_to_matrix2(self.coupling_matrix, coupling_matrix)
        return np.sum((self.coupling_matrix - self.coupling_matrix) < 0) == 0


def check_if_subgraph(coupling_matrices, potential_subgraphs):
    if len(coupling_matrices) == 0 or len(potential_subgraphs) == 0:
        return False

    if len(coupling_matrices.shape) > 2 and len(potential_subgraphs.shape) > 2:
        raise NotImplementedError()
    
    return np.any(np.sum((coupling_matrices - potential_subgraphs) < 0, (-1,-2)) == 0)

def check_if_subgraph_upper_triangle(coupling_matrices_upper_triangle, potential_subgraphs_upper_triangle):
    if len(coupling_matrices_upper_triangle) == 0 or len(potential_subgraphs_upper_triangle) == 0:
        return False

    if len(coupling_matrices_upper_triangle.shape) > 1 and len(potential_subgraphs_upper_triangle.shape) > 1:
        raise NotImplementedError()
    
    return np.any(np.sum((coupling_matrices_upper_triangle - potential_subgraphs_upper_triangle) < 0, (-1,)) == 0)

def calc_number_of_possibilities(mode_types, phase_constraints_for_squeezing=False):
    num_modes = len(mode_types)
    num_beamsplitter_couplings = 0
    num_squeezing_couplings = 0
    for idx2 in range(num_modes):
        for idx1 in range(idx2):
            if mode_types[idx1] == mode_types[idx2]:
                num_beamsplitter_couplings += 1
            else:
                num_squeezing_couplings += 1

    return 2**num_modes * 3**num_beamsplitter_couplings * (2+phase_constraints_for_squeezing)**num_squeezing_couplings

    # def __str__(self):
    #     return self.prepare_string_output(False)
    
    # def prepare_string_output(self, extended=False):
    #     if extended:
    #         to_print = self.extended_sets_of_constraints[0]
    #     else:
    #         to_print = self.all_sets_of_constraints[0]
        
    #     output_string = 'Contains the following constraints:\n'
    #     for c in to_print:
    #         output_string += c.__str__() + '\n'
    #     return output_string
    
    # def print(self, extended=False):
    #     print(self.prepare_string_output(extended))

    # def return_characteristics(self):
    #     combo = self.all_sets_of_constraints[0]
    #     num_detuning_constraints = 0
    #     num_coupling_constraints = 0
    #     num_coupling_phase_constraints = 0
    #     for c in combo:
    #         if type(c) == Constraint_coupling_zero:
    #             if c.idxs[0] == c.idxs[1]:
    #                 num_detuning_constraints += 1
    #             else:
    #                 num_coupling_constraints += 1
    #         elif type(c) == Constraint_coupling_phase_zero:
    #             num_coupling_phase_constraints += 1
    #         else:
    #             return NotImplementedError()
        
    #     N = self.num_modes

    #     info = {
    #         'num_detuning_constraints': num_detuning_constraints,
    #         'num_coupling_constraints': num_coupling_constraints,
    #         'num_coupling_phase_constraints': num_coupling_phase_constraints,
    #         'num_constraints': num_detuning_constraints+num_coupling_constraints+num_coupling_phase_constraints,
    #         'num_detunings': N - num_detuning_constraints,
    #         'num_couplings_including_detunings': (N**2-N)//2 + N - num_detuning_constraints - num_coupling_constraints,
    #         'num_couplings_excluding_detunings': (N**2-N)//2 - num_coupling_constraints,
    #     }
        
    #     return info
