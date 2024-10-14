import numpy as np
import sympy as sp
import copy
import scipy.optimize as sciopt

# from sympy.utilities.lambdify import lambdify

import multimode_systems.scattering as mss

from IPython.display import display, Math

def render(sympy_expr):
    display(Math(sp.latex(sympy_expr)))

def extend_matrix(S, num_modes, varnames='c', real_cs=False):
    input_shape = S.shape[0]
    if num_modes < input_shape:
        raise Exception('number of modes smaller than size of S')
        
    new_S = copy.deepcopy(S)
    free_parameters = []
    for current_shape in range(input_shape, num_modes):
        new_column = [sp.Symbol(varnames+'%i'%(i+len(free_parameters)), real=real_cs) for i in range(current_shape)]
        free_parameters.extend(new_column)
        new_S = new_S.col_insert(current_shape, sp.Matrix(new_column))
        
        new_row = [sp.Symbol(varnames+'%i'%(i+len(free_parameters)), real=real_cs) for i in range(current_shape+1)]
        free_parameters.extend(new_row)
        new_S = new_S.row_insert(current_shape, sp.Matrix([new_row]))
        
    return free_parameters, new_S

def setup_conditions_subset(operators, S, conditions_using_adjugate=True, kappa_rescaling=True, symbolic=True):
    num_dimensions = len(operators)
    num_modes = len(set([op.mode for op in operators]))

    if S.shape != (num_dimensions, num_dimensions):
        raise Exception('S has wrong shape')
    if num_modes != num_dimensions:
        raise Exception('operators are not allowed to correspond to the same modes')
    if not kappa_rescaling:
        raise NotImplementedError()

    identity = sp.eye(num_dimensions)

    #inverse of this matrix gives the dynamical matrix, inverse=adjugate/det
    inv_dynamical_matrix = sp.Matrix(S) - identity
    if symbolic:
        adjugate_matrix = inv_dynamical_matrix.adjugate()
        det = inv_dynamical_matrix.det()
        conjugate_func = sp.conjugate
        dynamical_matrix = adjugate_matrix/det
    else:
        inv_dynamical_matrix = np.asarray(inv_dynamical_matrix, dtype='complex')
        det = np.linalg.det(inv_dynamical_matrix)
        dynamical_matrix = np.linalg.inv(inv_dynamical_matrix)
        adjugate_matrix = det * dynamical_matrix
        conjugate_func = np.conj
    
    conditions_diag = []
    conditions_off_diag = []

    for idx in range(num_dimensions):
        conditions_diag.append(sp.re(dynamical_matrix[idx,idx])+sp.S(1)/sp.S(2))
    
    for idx1 in range(num_dimensions):
        for idx2 in range(idx1):
            if type(operators[idx1]) is type(operators[idx2]):
                sign = -1
            else:
                sign = +1

            if conditions_using_adjugate:
                conditions_off_diag.append(adjugate_matrix[idx1,idx2]-sign*conjugate_func(adjugate_matrix[idx2,idx1]))
            else:
                conditions_off_diag.append(dynamical_matrix[idx1,idx2]-sign*conjugate_func(dynamical_matrix[idx2,idx1]))

    all_conditions = conditions_diag + conditions_off_diag

    info = {
        'dynamical_matrix': dynamical_matrix,
        'adjugate_matrix': adjugate_matrix,
        'determinant': det
    }

    return all_conditions, info

def setup_conditions(S, conditions_using_adjugate=True, S_upper_left=False, kappa_rescaling=True, symbolic=True):
    num_dimensions = S.shape[0]
    if not S_upper_left and num_dimensions%2 != 0:
        raise Exception('with S_upper_left=False S needs an even number of dimensions')
    if not kappa_rescaling:
        raise NotImplementedError()

    identity = sp.eye(num_dimensions)
    
    #inverse of this matrix gives the dynamical matrix, inverse=adjugate/det
    inv_dynamical_matrix = sp.Matrix(S) - identity
    if symbolic:
        adjugate_matrix = inv_dynamical_matrix.adjugate()
        det = inv_dynamical_matrix.det()
        conjugate_func = sp.conjugate
        dynamical_matrix = adjugate_matrix/det
    else:
        inv_dynamical_matrix = np.asarray(inv_dynamical_matrix, dtype='complex')
        det = np.linalg.det(inv_dynamical_matrix)
        dynamical_matrix = np.linalg.inv(inv_dynamical_matrix)
        adjugate_matrix = det * dynamical_matrix
        conjugate_func = np.conj
    
    conditions_diag = []
    conditions_off_diag_A = []
    conditions_As_identical = []
    conditions_B_symmetric = []
    conditions_Bs_identical = []

    if S_upper_left:
        num_modes = num_dimensions
    else:
        num_modes = num_dimensions//2

    for idx in range(num_modes):
        conditions_diag.append(sp.re(dynamical_matrix[idx,idx])+sp.S(1)/sp.S(2))
    
    for idx1 in range(num_modes):
        for idx2 in range(idx1):
            if conditions_using_adjugate:
                conditions_off_diag_A.append(adjugate_matrix[idx1,idx2]+conjugate_func(adjugate_matrix[idx2,idx1]))
            else:
                conditions_off_diag_A.append(dynamical_matrix[idx1,idx2]+conjugate_func(dynamical_matrix[idx2,idx1]))

    if not S_upper_left:
        num_modes = num_dimensions//2
        for idx1 in range(num_modes):
            for idx2 in range(num_modes):
                if conditions_using_adjugate:
                    conditions_As_identical.append(adjugate_matrix[idx1,idx2]-conjugate_func(adjugate_matrix[idx1+num_modes,idx2+num_modes]))
                else:
                    conditions_As_identical.append(dynamical_matrix[idx1,idx2]-conjugate_func(dynamical_matrix[idx1+num_modes,idx2+num_modes]))

        for idx1 in range(num_modes):
            for idx2 in range(idx1):
                if conditions_using_adjugate:
                    conditions_B_symmetric.append(adjugate_matrix[idx1,idx2+num_modes]-adjugate_matrix[idx2,idx1+num_modes])
                else:
                    conditions_B_symmetric.append(dynamical_matrix[idx1,idx2+num_modes]-dynamical_matrix[idx2,idx1+num_modes])

        for idx1 in range(num_modes):
            for idx2 in range(num_modes):
                if conditions_using_adjugate:
                    conditions_Bs_identical.append(adjugate_matrix[idx1,idx2+num_modes]-conjugate_func(adjugate_matrix[idx1+num_modes,idx2]))
                else:
                    conditions_Bs_identical.append(dynamical_matrix[idx1,idx2+num_modes]-conjugate_func(dynamical_matrix[idx1+num_modes,idx2]))
    
    all_conditions = conditions_diag + conditions_off_diag_A + conditions_As_identical + conditions_B_symmetric + conditions_Bs_identical
    info = {
        'dynamical_matrix': dynamical_matrix,
        'adjugate_matrix': adjugate_matrix,
        'determinant': det
    }

    return all_conditions, info

def setup_unitary_conditions(S, symbolic=True):
    num_dimensions = S.shape[0]
    modes = [Mode().a for _ in range(num_dimensions)]
    
    return setup_Bogoliubov_conditions(modes, S, symbolic=symbolic)

def setup_Bogoliubov_conditions(operators, S, symbolic=True):
    num_dimensions = len(operators)
    if S.shape != (num_dimensions, num_dimensions):
        raise Exception('S has wrong shape')
    sigmaz_diag = []
    for idx, operator in enumerate(operators):
        if isinstance(operator, Annihilation_operator):
            sigmaz_diag.append(1)
        elif isinstance(operator, Creation_operator):
            sigmaz_diag.append(-1)
        else:
            raise Exception('operator %i has the wrong data type'%idx)
    
    if symbolic:
        sigmaz = sp.diag(*sigmaz_diag)
        product = S@sigmaz@sp.conjugate(S.T)
    else:
        sigmaz = np.diag(sigmaz_diag)
        S = np.asarray(S, dtype='complex')
        product = S@sigmaz@np.conjugate(S.T)

    conditions = []
    for idx in range(num_dimensions):
        conditions.append(product[idx,idx] - sigmaz[idx, idx])
    for idx1 in range(num_dimensions):
        for idx2 in range(idx1):
            conditions.append(product[idx1,idx2])
    
    info = {
        'product': product,
        'sigmaz': sigmaz
    }

    return conditions, info


def create_solutions_dict(parameters, solution):
    return {variable: value for variable, value in zip(parameters, solution)}

def extract_coupling_rates(dyn_matrix, verbose=True, subs_dict={}):
    num_modes = dyn_matrix.shape[0]
    
    extracted_couplings = []
    for idx in range(num_modes):
        delta = sp.im(dyn_matrix[idx,idx])/(-1j)
        extracted_couplings.append([idx, idx, float(delta.subs(subs_dict))])
        if verbose:
            display(Math('\Delta_%i='%(idx) + sp.latex(delta)))
    
    for idx2 in range(num_modes):
        for idx1 in range(idx2):
            J = dyn_matrix[idx1, idx2]/(-1j)
            extracted_couplings.append([idx1, idx2, complex(J.subs(subs_dict))])
            if verbose:
                display(Math('J_{%i%i}='%(idx1, idx2) + sp.latex(J)))
                
    return extracted_couplings

def find_numerical_solution_real_cs(conditions, free_parameters, initial_parameter_guess, further_subs_dict={}, method='lm'):
    # method='lm' doesn't require conditions and free parameters to have same shape, for more info regarding options jac and method, see scipy documentation
    
    conditions_subs = [condition.subs(further_subs_dict) for condition in conditions]
    conditions_lambdified = sp.utilities.lambdify(list(free_parameters), conditions_subs)
    conditions_lambdified_mod = lambda x: conditions_lambdified(*x)
    
    if method == 'least-squares':
        xsol = sciopt.least_squares(conditions_lambdified_mod, initial_parameter_guess)
    else:
        xsol = sciopt.root(conditions_lambdified_mod, initial_parameter_guess, jac=False, method=method)

    xsol['conditions_lambdified'] = conditions_lambdified_mod

    return create_solutions_dict(free_parameters, xsol.x), xsol

def find_numerical_solution_complex_cs(conditions, free_parameters, initial_parameter_guess, further_subs_dict={}):

    num_parameters = len(free_parameters)
    
    paras_real = [sp.Symbol('para_real%i'%idx) for idx in range(num_parameters)]
    paras_imag = [sp.Symbol('para_imag%i'%idx) for idx in range(num_parameters)]
    paras = paras_real + paras_imag
    paras_initial_guess = np.hstack((initial_parameter_guess.real, initial_parameter_guess.imag))
    
    subs_dict = {free_parameters[idx]: paras_real[idx]+1j*paras_imag[idx] for idx in range(num_parameters)}
    subs_dict.update(further_subs_dict)
    
    conditions_subs = [condition.subs(subs_dict) for condition in conditions]
    
    conditions_lambdified = lambdify(paras, conditions_subs)
    
    def conditions_lambdified_mod(x):
        output = np.asarray(conditions_lambdified(*x))
        return np.hstack((output.real, output.imag))
    
    xsol = sciopt.least_squares(conditions_lambdified_mod, paras_initial_guess)    
    
    solution_complex = xsol.x[:num_parameters] + 1.j*xsol.x[num_parameters:]
    
    return create_solutions_dict(free_parameters, solution_complex), xsol

def check_if_equation_system_has_trivial_non_solvable_equation(equ_system, symbols):
    # check if one of the equations has a equation which does not depend on any of the free variables
    # this equation would be ignored by sympy solve
    # if such an equation exists, and this equation is not be fulfilled, return True, otherwise False

    for equ in equ_system:
        if ~np.any([symbol in equ.free_symbols for symbol in symbols]):
            # check if the equation is fulfilled or not, if not return empty solution
            if sp.simplify(equ) != 0:
                return True
    
    return False

def symbolic_solver(equ_system, symbols, *args, **kwargs):
    if check_if_equation_system_has_trivial_non_solvable_equation(equ_system, symbols):
        return []

    return sp.solve(equ_system, symbols, *args, **kwargs)

class Particle_operators_base():
    dagger_operator = None
    def __init__(self, mode):
        self.mode = mode

    def set_dagger_operator(self, op):
        self.dagger_operator = op
    
    def dagger(self):
        return self.dagger_operator
    
class Creation_operator(Particle_operators_base):
    def set_dagger_operator(self, op):
        if isinstance(op, Annihilation_operator):
            self.dagger_operator = op
        else:
            raise Exception('dagger operator has to be of type Annihilation_operator')

class Annihilation_operator(Particle_operators_base):
    def set_dagger_operator(self, op):
        if isinstance(op, Creation_operator):
            self.dagger_operator = op
        else:
            raise Exception('dagger operator has to be of type Creation_operator')

class Mode():
    def __init__(self):
        self.a = Annihilation_operator(self)
        self.adag = Creation_operator(self)
        self.a.set_dagger_operator(self.adag)
        self.adag.set_dagger_operator(self.a)


def initialize_dynamical_matrix(operators, real=False):
    num_dimensions = len(operators)
    
    free_variables = []

    diag = [sp.Symbol('d%i%i'%(idx,idx), real=True) for idx in range(num_dimensions)]
    dyn_matrix = sp.Matrix(np.asarray(sp.symbols('d:%i:%i'%(num_dimensions,num_dimensions), real=real)).reshape([num_dimensions,num_dimensions]))

    for idx in range(num_dimensions):
        if real:
            dyn_matrix[idx,idx] = -sp.S(1)/sp.S(2)
        else:
            dyn_matrix[idx,idx] = sp.I*diag[idx] - sp.S(1)/sp.S(2)
            free_variables.append(diag[idx])

    for idx1 in range(num_dimensions):
        for idx2 in range(idx1):
            if type(operators[idx1]) is type(operators[idx2]):
                sign = -1
            else:
                sign = +1

            dyn_matrix[idx1,idx2] = sign*sp.conjugate(dyn_matrix[idx2,idx1])
            free_variables.append(dyn_matrix[idx2, idx1])

    info = {
        'free_variables': free_variables
    }

    return dyn_matrix, info

def symbolically_calc_scattering_matrix(dynamical_matrix):
    num_dimensions = dynamical_matrix.shape[0]
    
    adjugate_matrix = dynamical_matrix.adjugate()
    det = dynamical_matrix.det()
    return sp.eye(num_dimensions) + adjugate_matrix/det

def setup_conditions_dynamical_to_scattering(dynamical_matrix, S_target, split_complex_conditions_into_reals=False):
    num_dimensions = dynamical_matrix.shape[0]
    
    S_actual = symbolically_calc_scattering_matrix(dynamical_matrix)

    conditions = []
    for idx1 in range(num_dimensions):
        for idx2 in range(num_dimensions):
            if len(S_target[idx1,idx2].free_symbols) == 0: #condition should be improved in future
                if split_complex_conditions_into_reals:
                    conditions.append(sp.re(S_actual[idx1,idx2]-S_target[idx1,idx2]))
                    conditions.append(sp.im(S_actual[idx1,idx2]-S_target[idx1,idx2]))
                else:
                    conditions.append(S_actual[idx1,idx2]-S_target[idx1,idx2])

    return conditions

def initialize_dynamical_matrix_split_real_variables(operators, rescale_with_kappas=True):
    num_dimensions = len(operators)

    if len(set([op.mode for op in operators])) != num_dimensions:
        raise Exception('operators are not allowed to correspond to the same modes')

    kappas = {'kappalog%i'%idx: sp.Symbol('k%i'%idx, real=True, positive=True) for idx in range(num_dimensions)}
    Deltas = {'Delta%i'%idx: sp.Symbol('Delta%i'%idx, real=True) for idx in range(num_dimensions)}
    greals = {'greal%i%i'%(idx1, idx2): sp.Symbol('gre%i%i'%(idx1,idx2), real=True) for idx2 in range(num_dimensions) for idx1 in range(idx2)}
    gimags = {'gimag%i%i'%(idx1, idx2): sp.Symbol('gim%i%i'%(idx1,idx2), real=True) for idx2 in range(num_dimensions) for idx1 in range(idx2)}

    all_paras = {}
    for to_update in [kappas, Deltas, greals, gimags]:
        all_paras.update(to_update)

    kappas_matrix_inv_sqrt = sp.zeros(num_dimensions)
    for idx in range(num_dimensions):
        kappas_matrix_inv_sqrt[idx, idx] = sp.S(1)/sp.sqrt(sp.exp(kappas['kappalog%i'%idx]))

    dyn_matrix = sp.zeros(num_dimensions)

    for idx in range(num_dimensions):
        dyn_matrix[idx,idx] = - sp.I*Deltas['Delta%i'%idx] - sp.exp(kappas['kappalog%i'%idx])*sp.S(1)/sp.S(2)

    for idx2 in range(num_dimensions):
        for idx1 in range(idx2):
            if type(operators[idx1]) is type(operators[idx2]):
                sign = -1
            else:
                sign = +1

            dyn_matrix[idx1,idx2] = -sp.I * (greals['greal%i%i'%(idx1,idx2)] + sp.I * gimags['gimag%i%i'%(idx1,idx2)])
            dyn_matrix[idx2,idx1] = sign*sp.conjugate(dyn_matrix[idx1,idx2])

    dyn_matrix_dimensionless = kappas_matrix_inv_sqrt@dyn_matrix@kappas_matrix_inv_sqrt 

    info = {
        'Deltas': Deltas,
        'kappas': kappas,
        'greals': greals,
        'gimags': gimags,
        'dyn_matrix': dyn_matrix,
        'dyn_matrix_dimensionless': dyn_matrix_dimensionless
    }

    return dyn_matrix_dimensionless, all_paras, info

def lambdify(variables, list_of_funcs):
    num_inputs = len(variables)
    list_mod = [sp.utilities.lambdify(variables, func, modules='jax') for func in list_of_funcs]
    return [unpack_func_input(func, num_inputs) for func in list_mod]

def unpack_func_input(func, num_inputs):
    return lambda x: func(*[x[...,idx] for idx in range(num_inputs)])

def split_complex_variables(variables, real_imag_part=False):
    real_variables = []
    complex_variables = []
    for var in variables:
        if var.is_real:
            real_variables.append(var)
        else:
            complex_variables.append(var)

    num_complex_variables = len(complex_variables)
    if real_imag_part:
        raise NotImplementedError()
    else:
        abs_variables = []
        phase_variables = []
        subs_dict = {}
        for var in complex_variables:
            abs_variable = sp.Symbol(var.name+'abs', real=True)
            phase_variable = sp.Symbol(var.name+'phase', real=True)
            abs_variables.append(abs_variable)
            phase_variables.append(phase_variable)
            subs_dict[var] = abs_variable*sp.exp(sp.I*phase_variable)
    
    splitted_variables = real_variables+abs_variables+phase_variables
    variable_types = ['abs_variable']*len(real_variables) + ['abs_variable']*len(abs_variables) + ['phase_variable']*len(phase_variables)

    return splitted_variables, subs_dict, variable_types

def split_complex_equations(conditions, simplify=True):

    def append(list_to_append, expr):
        if expr != 0:
            if simplify:
                list_to_append.append(sp.simplify(expr))
            else:
                list_to_append.append(expr)

    out_conditions = []
    for idx in range(len(conditions)):
        # print(idx)
        append(out_conditions, sp.re(conditions[idx]))
        append(out_conditions, sp.im(conditions[idx]))
    
    return out_conditions

def check_validity_of_dynamical_matrix(operators, scattering_matrix):
    if len(scattering_matrix.free_symbols) != 0:
        raise Exception('scattering matrix is not allowed to have free symbols!')

    conditions, info = setup_conditions_subset(operators, scattering_matrix, conditions_using_adjugate=False, symbolic=False)

    info['conditions'] = conditions

    return info

def give_adjugate_element(scattering_matrix, i, j):
    # give adjugate matrix element ij of scattering_matrix
    S_sub = scattering_matrix.copy()
    S_sub.row_del(j)
    S_sub.col_del(i)
    return (-sp.S(1))**(i+j) * S_sub.det()

def give_adjugate_element_numpy(scattering_matrix, i, j):
    # give adjugate matrix element ij of scattering_matrix
    S_sub = np.delete(np.delete(scattering_matrix, j, 0), i, 1)
    return (-1)**(i+j) * np.linalg.det(S_sub)
    

def find_numerical_solution(calc_conditions, free_parameters, initial_parameter_guess, method='lm', max_nfev=None, jac='2-point', **kwargs):
    # method='lm' doesn't require conditions and free parameters to have same shape, for more info regarding options jac and method, see scipy documentation
    
    if len(free_parameters) > 0:
        try:
            if method == 'least-squares':
                xsol = sciopt.least_squares(calc_conditions, initial_parameter_guess, max_nfev=max_nfev, jac=jac, **kwargs)
            else:
                raise NotImplementedError()
        except:
            xsol = sciopt.least_squares(calc_conditions, initial_parameter_guess, max_nfev=1, jac=jac, **kwargs)

    else:
        xsol = {
            'x': np.zeros(0),
            'fun': calc_conditions(np.zeros(0)),
            'cost': None,
            'optimality': None,
            'nfev': None,
            'message': 'no free variables, no continuous optimization performed, invert matrix instead'
        }
    
    return create_solutions_dict(free_parameters, xsol['x']), xsol

    