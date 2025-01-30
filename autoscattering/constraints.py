import jax.numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
from collections import Counter
from tqdm import trange
import numpy as np

def plot_list_of_graphs(
        list_of_graphs, node_colors, positions=None,
        size_per_column=2.5, size_per_row=2.5, architectures_per_row=5, **kwargs
        ):
    
    num_columns = max(len(list_of_graphs)%architectures_per_row, architectures_per_row)
    num_rows = (len(list_of_graphs)-1)//architectures_per_row + 1
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(size_per_column*num_columns,size_per_row*num_rows))
    for ax in axes.flatten():
        ax.axis('off')
    for idx, combo in enumerate(list_of_graphs):
        plot_graph(triu_matrix=combo, node_colors=node_colors, positions=positions, ax=axes.flatten()[idx], **kwargs)

class Base_Constraint():
    def __init__(self):
        pass

    def __call__(self, S, coupling_matrix, kappa_int_matrix, mode_tyes):
        raise NotImplementedError()
    
    def __equ__(self, other_object):
        raise NotImplementedError()

class Coupling_Constraint(Base_Constraint):
    def __init__(self, idx1, idx2):
        if idx1 <= idx2:   
            self.idxs = [idx1, idx2]
        else:
            self.idxs = [idx2, idx1]
        
    def __eq__(self, other_object):
        if type(self) == type(other_object):
            if set(self.idxs) == set(other_object.idxs):
                return True
            else:
                return False
        else:
            return False
    
class Constraint_coupling_zero(Coupling_Constraint):
    def __call__(self, S, coupling_matrix, kappa_int_matrix, mode_tyes):
        idx1, idx2 = self.idxs
        element = coupling_matrix[idx1, idx2]

        if idx1 == idx2:
            return jnp.array([jnp.real(element)])
        else:
            return jnp.array([jnp.abs(element)])

    def __str__(self):
        return 'Coupling between %i and %i is set to 0'%(self.idxs[0], self.idxs[1])
    
    def __hash__(self):
        return hash(('Constraint_coupling_zero', self.idxs[0], self.idxs[1]))
        
class Constraint_coupling_phase_zero(Coupling_Constraint):
    def __init__(self, idx1, idx2):
        if idx1 == idx2:
            raise Exception('Constraint_coupling_phase_zero does only work if idx1 and idx2 are different')
        super().__init__(idx1, idx2)

    def __call__(self, S, coupling_matrix, kappa_int_matrix, mode_tyes):
        idx1, idx2 = self.idxs
        element = coupling_matrix[idx1, idx2]
        return jnp.array([jnp.imag(element)])
    
    def __str__(self):
        return 'Coupling phase between %i and %i is set to 0'%(self.idxs[0], self.idxs[1])
    
    def __hash__(self):
        return hash(('Constraint_coupling_zero', self.idxs[0], self.idxs[1]))
    
class Equal_Coupling_Rates(Base_Constraint):
    def __init__(self, list_equal_couplings):
        self.list_equal_couplings = list_equal_couplings
    
    def __call__(self, S, coupling_matrix, kappa_int_matrix, mode_tyes):
        idx0_ref = self.list_equal_couplings[0][0]
        idx1_ref = self.list_equal_couplings[0][1]
        deviation = 0
        for idx in range(1, len(self.list_equal_couplings)):
            idx0, idx1 = self.list_equal_couplings[idx]
            deviation += jnp.abs(jnp.abs(coupling_matrix[idx0,idx1]) - jnp.abs(coupling_matrix[idx0_ref,idx1_ref]))**2

        return deviation

class MinimalAddedInputNoise(Base_Constraint):
    def __call__(self, scattering_matrix, coupling_matrix, kappa_int_matrix, mode_types):
        '''
        calculates the difference between the number of added input photons and the quantum limit

        input arguments:
        scattering matrix: full scattering matrix for the current parameter set, this also includes the scattering from and to the auxiliary modes
        coupling matrix: dimensionless coupling matrix (sigma_z @ H in our equations) for the current parameter set, this constraint does not make any use of the coupling matrix
        kappa_int_matrix: diagonal matrix with the dimensionless intrinsic loss rates on the diagonal
        mode_types: whether a mode is part of subset M_1 or M_2 (True for M_1, False for M_2), see Appendix F for more details
        '''

        # calculate the linear response to fluctuations entering from the intrinsic loss channels (\mathcal{N} in the equations above)
        noise_matrix = (scattering_matrix - jnp.eye(scattering_matrix.shape[0])) @ jnp.complex_(jnp.sqrt(kappa_int_matrix))

        # calculate number of added photons at the input port, already considers that the target scattering matrix will be enforced
        total_noise = 1/2 * (jnp.sum(jnp.abs(scattering_matrix[0,1:])**2) + jnp.sum(jnp.abs(noise_matrix[0,:])**2))
        quantum_limit = 1/2
        return total_noise - quantum_limit

class MinimalAddedOutputNoise(Base_Constraint):
    def __init__(self, Gval):
        '''
        Gval: target gain value
        '''
        self.Gval = Gval

    def __call__(self, scattering_matrix, coupling_matrix, kappa_int_matrix, mode_types):
        '''
        calculates the difference between the number of added output photons and the quantum limit

        input arguments:
        scattering matrix: full scattering matrix for the current parameter set, this also includes the scattering from and to the auxiliary modes
        coupling matrix: dimensionless coupling matrix (sigma_z @ H in our equations) for the current parameter set, this constraint does not make any use of the coupling matrix
        kappa_int_matrix: diagonal matrix with the dimensionless intrinsic loss rates on the diagonal
        mode_types: list of boolean values. These values determine whether a mode is part of subset M_1 or M_2 (True for M_1, False for M_2), see Appendix F for more details
        '''

        # calculate the linear response to fluctuations entering from the intrinsic loss channels (\mathcal{N} in the equations above)
        noise_matrix = (scattering_matrix - jnp.eye(scattering_matrix.shape[0])) @ jnp.complex_(jnp.sqrt(kappa_int_matrix))

        # calculate number of added photons at the output port, already considers that the target scattering matrix will be enforced
        total_noise = 1/2 * (jnp.sum(jnp.abs(scattering_matrix[1,2:])**2) + jnp.sum(jnp.abs(noise_matrix[1,:])**2))

        # input_output_part_of_same_set=True means that both modes are part of the same set M_1 or M_2
        # This slightly influences the quantum limit for the output noise
        input_output_part_of_same_set = mode_types[0]==mode_types[1]
        if input_output_part_of_same_set:
            quantum_limit = (self.Gval - 1)/2
        else:
            quantum_limit = (self.Gval + 1)/2

        return total_noise - quantum_limit



def check_overlapping_constraints(list_of_constraints):
    for idx2 in range(len(list_of_constraints)):
        for idx1 in range(idx2):
            constraint1 = list_of_constraints[idx1]
            constraint2 = list_of_constraints[idx2]
            if type(constraint1) == type(constraint2):
                if constraint1 == constraint2:
                    raise Exception('constraint %i overlaps with constraint %i'%(idx1, idx2))
            elif set([type(constraint1), type(constraint2)]) == set([Constraint_coupling_zero, Constraint_coupling_phase_zero]):
                if set(constraint1.idxs) == set(constraint2.idxs):
                    raise Exception('Coupling and its phase are both set to zero, check constraints %i and %i'%(idx1, idx2))
                
def setup_constraints(couplings_set_to_zero, coupling_phases_set_to_zero):
    constraints = []
    for idx1, idx2 in couplings_set_to_zero:
        constraints.append(Constraint_coupling_zero(idx1, idx2))
    for idx1, idx2 in coupling_phases_set_to_zero:
        constraints.append(Constraint_coupling_phase_zero(idx1, idx2))
    return constraints

EDGETYPEACTIVE = 'active'
EDGETYPEPASSIVE = 'passive'
EDGETYPEDETUNING = 'detuning'
EDGETYPESQUEEZING = 'squeezing'
EDGETYPENONE = None

def return_edge_type(combos, mode_types, idx1, idx2):
    for c in cleanup_list_of_constraints(list(combos)):
        if set([idx1, idx2]) == set(c.idxs):
            if type(c)==Constraint_coupling_zero:
                return EDGETYPENONE
            elif type(c) == Constraint_coupling_phase_zero:
                return EDGETYPEPASSIVE
    
    if idx1 == idx2:
        return EDGETYPEDETUNING
    elif mode_types[idx1] == mode_types[idx2]:
        return EDGETYPEACTIVE
    else:
        return EDGETYPESQUEEZING

def plot_graph(combination=None, triu_matrix=None, node_colors=None, mode_types='no_squeezing', color_detuning='black', color_passive='black', color_active='green', color_squeezing='blue', positions=None, ax=None, edge_width=2):
    from autoscattering.architecture import translate_upper_triangle_coupling_matrix_to_conditions
    if combination is None:
        combination = translate_upper_triangle_coupling_matrix_to_conditions(triu_matrix)
    num_modes = len(node_colors)

    if mode_types == 'no_squeezing':
        mode_types = np.ones(num_modes, dtype='bool')

    # node_labels = {}
    # for idx in range(num_modes):
    #     if mode_types[idx]:
    #         node_labels[idx] = ''
    #     else:
    #         node_labels[idx] = '$\dag$'


    G = nx.Graph()
    G.add_nodes_from([(idx, {"color": node_colors[idx]}) for idx in range(num_modes)])
    for idx2 in range(num_modes):
        for idx1 in range(idx2+1):
            edge_type = return_edge_type(combination, mode_types, idx1, idx2)
            if edge_type is not EDGETYPENONE:
                if edge_type is EDGETYPEACTIVE:
                    color = color_active
                if edge_type is EDGETYPEPASSIVE:
                    color = color_passive
                if edge_type is EDGETYPEDETUNING:
                    color = color_detuning
                if edge_type is EDGETYPESQUEEZING:
                    color = color_squeezing
                G.add_edge(idx1, idx2, color=color)
    nx.draw(
        G,
        positions,
        node_color=nx.get_node_attributes(G, 'color').values(),
        # labels=node_labels, #nx.get_node_attributes(G, 'label'),
        edge_color=nx.get_edge_attributes(G, 'color').values(),
        width=[edge_width for u,v in G.edges],
        ax=ax
    )


def cleanup_list_of_constraints(combo):
    combo_cleaned = combo.copy()
    for idx2 in range(len(combo)):
        for idx1 in range(idx2):
            constraint1 = combo[idx1]
            constraint2 = combo[idx2]
            if set([type(constraint1), type(constraint2)]) == set([Constraint_coupling_zero, Constraint_coupling_phase_zero]):
                if set(constraint1.idxs) == set(constraint2.idxs):
                    if type(constraint1) == Constraint_coupling_phase_zero:
                        combo_cleaned.remove(constraint1)
                    if type(constraint2) == Constraint_coupling_phase_zero:
                        combo_cleaned.remove(constraint2)

    return combo_cleaned

def add_phase_zero_constraints(all_combinations):
    extended_combinations = []
    for combo in all_combinations:
        combo_out = list(combo).copy()
        for c in list(combo):
            if type(c) == Constraint_coupling_zero:
                if c.idxs[0] != c.idxs[1]:
                    combo_out.append(Constraint_coupling_phase_zero(c.idxs[0], c.idxs[1]))
        extended_combinations.append(set(combo_out))
    return extended_combinations

def clean_and_reject_uncomplete_combinations(all_combinations):
    extended_combos = add_phase_zero_constraints(all_combinations) 
    sorted_combos = extended_combos.copy()

    for idx2 in trange(len(extended_combos)):
        for idx1 in range(idx2):
            combo1 = extended_combos[idx1]
            combo2 = extended_combos[idx2]
            if combo1 in sorted_combos and combo2 in sorted_combos:
                if combo1.issubset(combo2):
                    sorted_combos.remove(combo1)
                if combo2.issubset(combo1):
                    sorted_combos.remove(combo2)
    # return sorted_combos
    return [set(cleanup_list_of_constraints(list(combo))) for combo in sorted_combos]

def characterize_combinations(combinations):
    num_coupling_conditions = []
    num_coupling_phase_conditions = []
    for combo in combinations:
        counters = Counter(map(type, combo))
        num_coupling_conditions.append(counters[Constraint_coupling_zero])
        num_coupling_phase_conditions.append(counters[Constraint_coupling_phase_zero])
    
    num_coupling_conditions = np.array(num_coupling_conditions)
    num_coupling_phase_conditions = np.array(num_coupling_phase_conditions)

    info = {
        'combinations': combinations,
        'num_constraints': num_coupling_conditions+num_coupling_phase_conditions,
        'num_coupling_constraints': num_coupling_conditions,
        'num_coupling_phase_constraints': num_coupling_phase_conditions
    }
    return info

def print_combo(combo):
    for el in combo:
        print(el)


    

    