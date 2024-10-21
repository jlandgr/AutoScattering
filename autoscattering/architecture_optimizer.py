import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
import pickle

from tqdm import trange, tqdm
from itertools import permutations, product, combinations

import multimode_systems.symbolic as sym
import multimode_systems.constraints as msc
import multimode_systems.architecture as arch

import time

SUCCESS_DYNAMICAL_AND_SCATTERING = 2
SUCCESS_SCATTERING = 1
SUCCESS_FAILED = 0

def setup_Bogoliubov_conditions(operators, calc_S, constraints=[], port_intrinsic_losses=None):
    num_dimensions = len(operators)

    use_port_instrinsic_losses = False
    if port_intrinsic_losses is not None:
        if np.sum(port_intrinsic_losses) > 0:
            use_port_instrinsic_losses = True
            num_port_intrinsic_losses = np.sum(port_intrinsic_losses)
            mask_port_instrinsic_losses = jnp.array(np.where(port_intrinsic_losses)[0])
    
    sigmaz_diag = []
    for idx, operator in enumerate(operators):
        if isinstance(operator, sym.Annihilation_operator):
            sigmaz_diag.append(1)
        elif isinstance(operator, sym.Creation_operator):
            sigmaz_diag.append(-1)
        else:
            raise Exception('operator %i has the wrong data type'%idx)

    sigmaz = jnp.diag(jnp.array(sigmaz_diag))
    identity = jnp.eye(num_dimensions)
    
    def calc_conditions(x): 
        S = calc_S(*x)
        Sdag = jnp.conjugate(S.T)

        lhs = S@sigmaz@Sdag

        if use_port_instrinsic_losses:
            intrinsic_losses_array = jnp.zeros(num_dimensions)
            intrinsic_losses_array = jnp.diag(intrinsic_losses_array.at[mask_port_instrinsic_losses].set(x[-num_port_intrinsic_losses:]))
            rhs = sigmaz - (S-identity)@sigmaz@intrinsic_losses_array@(Sdag-identity)
        else:
            rhs = sigmaz

        indices_upper_triangle = jnp.triu_indices(num_dimensions)

        difference = lhs[indices_upper_triangle] - rhs[indices_upper_triangle]

        inverse_matrix = jnp.linalg.inv(S-jnp.eye(num_dimensions))
        # determinant = jnp.linalg.det(S-jnp.eye(num_dimensions))

        if len(constraints) > 0:
            # evaluated_constraints = jnp.hstack((c(S, determinant) for c in constraints))
            evaluated_constraints = jnp.hstack((c(S, inverse_matrix) for c in constraints))
            conditions = jnp.hstack((jnp.real(difference), jnp.imag(difference), evaluated_constraints))
        else:
            conditions = jnp.hstack((jnp.real(difference), jnp.imag(difference)))
        
        return conditions

    return calc_conditions

def check_validity_of_dynamical_matrix(operators, scattering_matrix, kappa_int_rescaled=None):
    scattering_matrix = np.asarray(scattering_matrix)
# setup_conditions_subset(operators, S, conditions_using_adjugate=True, kappa_rescaling=True, symbolic=True):
    num_dimensions = len(operators)
    num_modes = len(set([op.mode for op in operators]))
    if num_modes != num_dimensions:
        raise Exception('operators are not allowed to correspond to the same modes')
    
    if kappa_int_rescaled is None:
        kappa_int_rescaled = np.zeros(num_modes)

    identity = np.eye(num_dimensions)

    #inverse of this matrix gives the dynamical matrix, inverse=adjugate/det
    inv_dynamical_matrix = scattering_matrix - identity
    dynamical_matrix = np.linalg.inv(inv_dynamical_matrix)
    
    conditions_diag = []
    conditions_off_diag = []

    for idx in range(num_dimensions):
        conditions_diag.append(sp.re(dynamical_matrix[idx,idx])+(1+kappa_int_rescaled[idx])/2.)
    
    for idx1 in range(num_dimensions):
        for idx2 in range(idx1):
            if type(operators[idx1]) is type(operators[idx2]):
                sign = -1
            else:
                sign = +1

            conditions_off_diag.append(dynamical_matrix[idx1,idx2]-sign*np.conj(dynamical_matrix[idx2,idx1]))

    all_conditions = conditions_diag + conditions_off_diag

    info = {
        'dynamical_matrix': dynamical_matrix,
        'determinant': np.linalg.det(inv_dynamical_matrix),
        'conditions': all_conditions,
        'conditions_diag': conditions_diag,
        'conditions_off_diag': conditions_off_diag
    }

    return info


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
    variable_types = [VAR_ABS]*len(real_variables) + [VAR_ABS]*len(abs_variables) + [VAR_PHASE]*len(phase_variables)

    return splitted_variables, subs_dict, variable_types


AUTODIFF_FORWARD = 'autodiff_forward'
AUTODIFF_REVERSE = 'autodiff_reverse'
DIFFERENCE_QUOTIENT = '2-point'


# Variable types
# VAR_USER_DEFINED = 'user_defined'
VAR_ABS = 'abs_variable'
VAR_ERROR_ABS = 'error_abs_variable'
VAR_PHASE = 'phase_variable'
VAR_INTRINSIC_LOSS = 'intrinsic_loss_variable'
VAR_USER_DEFINED = 'user_defined'

INIT_ABS_RANGE_DEFAULT = [-1., 1.]
INIT_KAPPA_RANGE_DEFAULT = [0.1, 1.]
USER_DEFINED_RANGE_DEFAULT = [-np.pi, np.pi]
INIT_ERROR_ABS_RANGE_DEFAULT = [-0.1, 0.1]
BOUNDS_ERROR_ABS_DEFAULT = [-0.1, 0.1]
BOUNDS_ABS_DEFAULT = [-np.inf, np.inf]

class Architecture_Optimizer():
    def __init__(
            self, S_target, num_modes, allow_errors_on_S_target=False,
            phase_constraints_for_squeezing=False,
            enforced_constraints=[], mode_types='no_squeezing',
            port_intrinsic_losses=False,
            gradient_method=AUTODIFF_FORWARD,
            make_test_without_additional_constraints=True,
            max_violation_success=1.e-5,
            kwargs_optimization={},
            kwargs_solver={}):
        
        self.kwargs_optimization = kwargs_optimization
        self.kwargs_optimization['interrupt_if_successful'] = True
        self.kwargs_optimization['xtol'] = None
        self.kwargs_optimization['max_violation_success'] = max_violation_success

        self.kwargs_solver = kwargs_solver

        self.S_target = S_target
        self.num_modes = num_modes
        self.num_port_modes = S_target.shape[0]
        self.allow_errors_on_S_target = allow_errors_on_S_target
        self.phase_constraints_for_squeezing = phase_constraints_for_squeezing

        self.__initialize_variables__(port_intrinsic_losses)

        if mode_types == 'no_squeezing':
            self.mode_types = [True for _ in range(num_modes)]
        else:
            self.mode_types = mode_types
        
        self.operators = []
        for idx in range(num_modes):
            if self.mode_types[idx]:
                self.operators.append(sym.Mode().a)
            else:
                self.operators.append(sym.Mode().adag)

        self.enforced_constraints = enforced_constraints
        self.num_enforced_constraints = len(self.enforced_constraints)

        self.all_possible_constraints = self.setup_all_constraints()

        self.calc_conditions_and_all_constraints = jax.jit(setup_Bogoliubov_conditions(
            self.operators,
            self.S_extended_lambdified,
            constraints=self.enforced_constraints+self.all_possible_constraints,
            port_intrinsic_losses=self.port_intrinsic_losses
        ))

        offset = num_modes**2 + num_modes + self.num_enforced_constraints

        self.mask_calc_all_possible_constraints = (jnp.arange(offset, offset+len(self.all_possible_constraints)))
        def calc_all_possible_constraints(x):
            return self.calc_conditions_and_all_constraints(x)[self.mask_calc_all_possible_constraints]
        self.calc_all_possible_constraints = calc_all_possible_constraints #jax.jit(calc_all_possible_constraints)

        self.gradient_method = gradient_method
        if gradient_method == AUTODIFF_FORWARD:
            self.calc_grad_conditions_and_all_constraints = jax.jit(jax.jacfwd(self.calc_conditions_and_all_constraints))
        elif gradient_method == AUTODIFF_REVERSE:
            self.calc_grad_conditions_and_all_constraints = jax.jit(jax.jacrev(self.calc_conditions_and_all_constraints))
        elif gradient_method == DIFFERENCE_QUOTIENT:
            self.calc_grad_conditions_and_all_constraints = None
        else:
            raise NotImplementedError()
        
        #make run without additional conditions
        if make_test_without_additional_constraints:
            success, _, _ = self.repeated_optimization(constraints=[], **self.kwargs_optimization)
            if not success:
                print('unconditioned system cannot be solved, interrupting')
            # return []
        
        #make run for each possible constraint
        # self.all_possible_constraints = []
        self.invalid_combinations = []
        self.valid_combinations = []
        self.tested_complexities = []
        self.valid_combinations_solutions = []

    def __initialize_variables__(self, port_intrinsic_losses):
        if self.allow_errors_on_S_target:
            self.errorabss = {'errorabs%i%i'%(idx1,idx2): sp.Symbol('errorabs%i%i'%(idx1,idx2), real=True) for idx1 in range(self.num_port_modes) for idx2 in range(self.num_port_modes)}
            self.errorphases = {'errorphase%i%i'%(idx1,idx2): sp.Symbol('errorphase%i%i'%(idx1,idx2), real=True) for idx1 in range(self.num_port_modes) for idx2 in range(self.num_port_modes)}
            self.error_array = sp.zeros(self.num_port_modes)
            for idx1 in range(self.num_port_modes):
                for idx2 in range(self.num_port_modes):
                    self.error_array[idx1,idx2] = self.errorabss['errorabs%i%i'%(idx1,idx2)] * sp.exp(sp.I*self.errorphases['errorphase%i%i'%(idx1,idx2)])
        else:
            self.errorabss = {}
            self.errorphases = {}
            self.error_array = sp.zeros(self.num_port_modes)

        
        self.user_defined_variables = list(self.S_target.free_symbols)
        self.user_defined_variables.sort(key=str)
        for var in self.user_defined_variables:
            if not var.is_real:
                raise Exception('variable '+var.name+' is complex, only real variables are allowed')

        if port_intrinsic_losses is True:
            self.port_intrinsic_losses = [True for _ in range(self.num_port_modes)]
        elif port_intrinsic_losses is False:
            self.port_intrinsic_losses = [False for _ in range(self.num_port_modes)]
        else:
            self.port_intrinsic_losses = port_intrinsic_losses
        
        self.paras_intrinsic_losses = []
        for mode_idx in range(self.num_port_modes):
            if self.port_intrinsic_losses[mode_idx]:
                self.paras_intrinsic_losses.append(sp.Symbol('kappa_int%i'%mode_idx, real=True))

        self.internal_mode_parameters, self.S_extended = sym.extend_matrix(self.S_target+self.error_array, self.num_modes, varnames='int')
        self.splitted_internal_mode_parameters, self.subs_dict_split, self.variable_types_split = split_complex_variables(self.internal_mode_parameters)
        self.S_extended_subs = self.S_extended.subs(self.subs_dict_split)
        
        self.variables = self.user_defined_variables + self.splitted_internal_mode_parameters + self.paras_intrinsic_losses + list(self.errorabss.values()) + list(self.errorphases.values())
        self.variable_types = [VAR_USER_DEFINED]*len(self.user_defined_variables) + self.variable_types_split + [VAR_INTRINSIC_LOSS] * np.sum(self.port_intrinsic_losses) + [VAR_ERROR_ABS] * len(self.errorabss) + [VAR_PHASE] * len(self.errorphases)
        
        self.S_extended_lambdified = jax.jit(sp.utilities.lambdify(list(self.variables), self.S_extended_subs, modules='jax'))

    def setup_all_constraints(self):
        all_constraints = []
        for idx in range(self.num_modes):
            all_constraints.append(msc.Constraint_coupling_zero(idx, idx))
        for idx2 in range(self.num_modes):
            for idx1 in range(idx2):
                if type(self.operators[idx1]) is type(self.operators[idx2]):
                    all_constraints.append(msc.Constraint_coupling_zero(idx1, idx2))
                    all_constraints.append(msc.Constraint_coupling_phase_zero(idx1, idx2))
                else:
                    all_constraints.append(msc.Constraint_coupling_zero(idx1, idx2))
                    if self.phase_constraints_for_squeezing:
                        all_constraints.append(msc.Constraint_coupling_phase_zero(idx1, idx2))

        # for cond in self.enforced_constraints:
        #     if type(cond) is msc.Constraint_coupling_zero:
        #         all_constraints.remove(cond)
        #         phase_cond_to_remove = msc.Constraint_coupling_phase_zero(cond.idxs[0], cond.idxs[1])
        #         if phase_cond_to_remove in all_constraints:
        #             all_constraints.remove(phase_cond_to_remove)
        #     elif type(cond) is msc.Constraint_coupling_phase_zero:
        #         if cond in all_constraints:
        #             all_constraints.remove(cond)
        
        return all_constraints

    def create_initial_guess(self, init_abs_range=INIT_ABS_RANGE_DEFAULT, init_user_defined_range=USER_DEFINED_RANGE_DEFAULT, init_kappa_int_range=INIT_KAPPA_RANGE_DEFAULT, init_error_abs_range=INIT_ERROR_ABS_RANGE_DEFAULT,):
        random_values = []
        for var_type in self.variable_types:
            if var_type == VAR_ABS:
                random_values.append(np.random.uniform(init_abs_range[0], init_abs_range[-1]))
            elif var_type == VAR_USER_DEFINED:
                random_values.append(np.random.uniform(init_user_defined_range[0], init_user_defined_range[-1]))
            elif var_type == VAR_ERROR_ABS:
                random_values.append(np.random.uniform(init_error_abs_range[0], init_error_abs_range[-1]))
            elif var_type == VAR_PHASE:
                random_values.append(np.random.uniform(-np.pi, np.pi))
            elif var_type == VAR_INTRINSIC_LOSS:
                random_values.append(np.random.uniform(init_kappa_int_range[0], init_kappa_int_range[-1]))
            else:
                raise NotImplementedError()
            
        return jnp.array(random_values)
    
    def setup_bounds(self, bounds_abs=BOUNDS_ABS_DEFAULT, bounds_error_abs=BOUNDS_ERROR_ABS_DEFAULT):
        bounds = []
        for var_type in self.variable_types:
            if var_type == VAR_PHASE or var_type == VAR_INTRINSIC_LOSS or var_type == VAR_USER_DEFINED:
                bounds.append([-np.inf, np.inf])
            elif var_type == VAR_ABS:
                bounds.append(bounds_abs)
            elif var_type == VAR_ERROR_ABS:
                bounds.append(bounds_error_abs)
            else:
                raise NotImplementedError()
        
        return np.asarray(bounds).T

    def create_conditions_maps(self, constraints, enforced_constraints=True):
        num_Bogoliubov_conditions = self.num_modes**2 + self.num_modes  # 2*number of elements in the upper triangle matrix
        index_matrix = list(np.arange(num_Bogoliubov_conditions))
        if enforced_constraints:
            index_matrix += list(np.arange(num_Bogoliubov_conditions, num_Bogoliubov_conditions+self.num_enforced_constraints))
        for idx in range(len(self.all_possible_constraints)):
            if self.all_possible_constraints[idx] in constraints:
                index_matrix += [idx+num_Bogoliubov_conditions+self.num_enforced_constraints,]

        return index_matrix

    def prepare_calc_conditions_and_gradients(self, constraints, enforced_constraints=True):
        msc.check_overlapping_constraints(constraints)

        indices = self.create_conditions_maps(constraints, enforced_constraints=enforced_constraints)
        def calc_conditions(x):
            result_conditions = np.array(self.calc_conditions_and_all_constraints(x))[indices]
            # self.history_conditions.append(result_conditions)
            # self.history_params.append(x)
            return result_conditions
        
        if self.gradient_method == AUTODIFF_FORWARD or self.gradient_method == AUTODIFF_REVERSE:
            def calc_gradients(x):
                return np.array(self.calc_grad_conditions_and_all_constraints(x))[indices,:]
        else:
            calc_gradients = self.gradient_method

        return calc_conditions, calc_gradients

    def extract_kappa_int_rescaled_from_solution(self, solution):
        kappa_int_rescaled = np.zeros(self.num_port_modes)
        kappa_int_rescaled[self.port_intrinsic_losses] = [solution[kappa_int] for kappa_int in  self.paras_intrinsic_losses]
        return np.concatenate((kappa_int_rescaled, np.zeros(self.num_modes-self.num_port_modes)))
    
    def optimize_given_constraints(self, constraints, verbose=False, max_nfev=None, enforced_constraints=True, init_abs_range=INIT_ABS_RANGE_DEFAULT, init_user_defined_range=USER_DEFINED_RANGE_DEFAULT, init_kappa_int_range=INIT_KAPPA_RANGE_DEFAULT, init_error_abs_range=INIT_ERROR_ABS_RANGE_DEFAULT, bounds_abs=BOUNDS_ABS_DEFAULT, bounds_error_abs=BOUNDS_ERROR_ABS_DEFAULT, max_violation_success=1.e-5, calc_conditions_and_gradients=None, **kwargs_solver):

        # start_time = time.time()
        if calc_conditions_and_gradients is None:
            calc_conditions, calc_gradients = self.prepare_calc_conditions_and_gradients(constraints, enforced_constraints=enforced_constraints)
        else:
            calc_conditions, calc_gradients = calc_conditions_and_gradients

        initial_guess = self.create_initial_guess(
            init_abs_range=init_abs_range,
            init_user_defined_range=init_user_defined_range,
            init_kappa_int_range=init_kappa_int_range,
            init_error_abs_range=init_error_abs_range
        )
        bounds = self.setup_bounds(bounds_abs, bounds_error_abs)

        # calc_conditions(initial_guess)

        # print('prepare:', time.time()-start_time)
        # start_time = time.time()
        solution, info_solution = sym.find_numerical_solution(calc_conditions, self.variables, initial_guess, method='least-squares', max_nfev=max_nfev, jac=calc_gradients, bounds=bounds, **kwargs_solver)

        # print('solve:', time.time()-start_time)
        # start_time = time.time()

        scattering_matrix_solution = self.S_extended_lambdified(*info_solution['x'])

        info_check = check_validity_of_dynamical_matrix(
            self.operators,
            scattering_matrix_solution,
            kappa_int_rescaled=self.extract_kappa_int_rescaled_from_solution(solution)
        )

        max_violation_conditions = np.max(np.abs(info_solution['fun']))
        max_violation_dynamical_matrix = np.max(np.abs(info_check['conditions']))

        if max_violation_conditions < max_violation_success and max_violation_dynamical_matrix < max_violation_success:
            success = SUCCESS_DYNAMICAL_AND_SCATTERING
            optimizer_message = 'valid scattering matrix and valid dynamical matrix'
        elif max_violation_conditions < max_violation_success:
            success = SUCCESS_SCATTERING
            optimizer_message = 'valid scattering matrix, but invalid dynamical matrix'
        else:
            success = SUCCESS_FAILED
            optimizer_message = 'optimization failed'

        info_out = {
            'initial_guess': initial_guess,
            'solution': solution,
            'solution_array': info_solution['x'],
            'final_cost': info_solution['cost'],
            'optimality': info_solution['optimality'],
            'conditions': info_solution['fun'],
            'maximal_violation': max_violation_conditions,
            'maximal_violation_dynamical_matrix': max_violation_dynamical_matrix,
            'success': success,
            'least_square_message': info_solution['message'],
            'optimizer_message': optimizer_message,
            'scattering_matrix': scattering_matrix_solution,
            'dynamical_matrix': info_check['dynamical_matrix'],
            'conditions_dynamical_matrix': info_check['conditions'],
            'nfev': info_solution['nfev'],
            'bounds': bounds
            # 'njev': info_solution.njev,
            # 'solver_status': info_solution.status,
            # 'solver_message': info_solution.message,
            # 'solver_success': info_solution.success
        }

        if verbose:
            print(info_out['optimizer_message'])
            print('optimality:', info_solution['optimality'])
            print('maximal violation:', max_violation_conditions)
            print('maximal violation of being a dynamical matrix:', max_violation_dynamical_matrix)

        # print('check and output:', time.time()-start_time)
        # start_time = time.time()

        return solution, info_out
    
    def repeated_optimization(self, num_tests, constraints, verbosity_level=0, max_nfev=None, enforced_constraints=True, init_abs_range=INIT_ABS_RANGE_DEFAULT, init_user_defined_range=USER_DEFINED_RANGE_DEFAULT, init_kappa_int_range=INIT_KAPPA_RANGE_DEFAULT, init_error_abs_range=INIT_ERROR_ABS_RANGE_DEFAULT, bounds_abs=BOUNDS_ABS_DEFAULT, bounds_error_abs=BOUNDS_ERROR_ABS_DEFAULT, max_violation_success=1.e-5, interrupt_if_successful=False, ignore_conditions_dynamical_matrix=False, **kwargs_solver):
        all_infos = []
        success_idxs = []

        if ignore_conditions_dynamical_matrix:
            success_criteria = SUCCESS_SCATTERING
        else:
            success_criteria = SUCCESS_DYNAMICAL_AND_SCATTERING


        if verbosity_level > 0:
            if len(constraints) > 0:
                print('optimizing with the following constraints:')
                for c in constraints:
                    print(c)
            else:
                print('optimizing without additional constraints')

        calc_conditions_and_gradients = self.prepare_calc_conditions_and_gradients(constraints, enforced_constraints=enforced_constraints)

        for test_idx in range(num_tests):
            _, info = self.optimize_given_constraints(
                constraints,
                verbose=verbosity_level>1,
                max_nfev=max_nfev,
                init_abs_range=init_abs_range,
                init_kappa_int_range=init_kappa_int_range,
                init_error_abs_range=init_error_abs_range,
                init_user_defined_range=init_user_defined_range,
                max_violation_success=max_violation_success,
                calc_conditions_and_gradients=calc_conditions_and_gradients,
                bounds_abs=bounds_abs,
                bounds_error_abs=bounds_error_abs,
                **kwargs_solver
            )
            all_infos.append(info)
            if info['success'] == success_criteria:
                success_idxs.append(test_idx)
                if interrupt_if_successful:
                    break

        if len(success_idxs) > 0:
            success = True
        else:
            success = False

        if verbosity_level > 0 and not interrupt_if_successful:
            if success:
                print('success')
            else:
                print('failed')
            print('%i out of %i runs were successfull'%(len(success_idxs), num_tests))
        
        if verbosity_level > 0 and interrupt_if_successful:
            if success:
                print('found a solution after %i runs, continue with next solution set'%(test_idx+1))
            else:
                print('no solution found within %i runs'%num_tests)
        
        return success, all_infos, success_idxs
    
    def prepare_all_possible_combinations(self):
        idxs_upper_triangle = np.triu_indices(self.num_modes)
        # idxs_lower_triangle = np.tril_indices(self.num_modes)

        # self.possible_matrix_entries = []
        # for idx1, idx2 in np.array(idxs_upper_triangle).T:
        #     if idx1 == idx2:
        #         self.possible_matrix_entries.append([arch.NO_COUPLING, arch.DETUNING])
        #     else:
        #         if type(self.operators[idx1]) is type(self.operators[idx2]):
        #             self.possible_matrix_entries.append([arch.NO_COUPLING, arch.COUPLING_WITHOUT_PHASE, arch.COUPLING_WITH_PHASE])
        #         else:
        #             if self.phase_constraints_for_squeezing:
        #                 self.possible_matrix_entries.append([arch.NO_COUPLING, arch.COUPLING_WITHOUT_PHASE, arch.COUPLING_WITH_PHASE])
        #             else:
        #                 self.possible_matrix_entries.append([arch.NO_COUPLING, arch.COUPLING_WITH_PHASE])

        # self.possible_matrix_entries = []
        # for idx1, idx2 in np.array(idxs_upper_triangle).T:
        #     allowed_entries = [arch.NO_COUPLING]
        #     if msc.Constraint_coupling_zero(idx1,idx2) in self.enforced_constraints:
        #         pass #no further possiblities are added
        #     elif idx1 == idx2:
        #         allowed_entries.append(arch.DETUNING)
        #     elif msc.Constraint_coupling_phase_zero(idx1,idx2) in self.enforced_constraints:
        #         allowed_entries.append(arch.COUPLING_WITHOUT_PHASE)
        #     else:
        #         if type(self.operators[idx1]) is type(self.operators[idx2]) or self.phase_constraints_for_squeezing:
        #             allowed_entries.append(arch.COUPLING_WITHOUT_PHASE)
        #         allowed_entries.append(arch.COUPLING_WITH_PHASE)

        #     self.possible_matrix_entries.append(allowed_entries)

        self.possible_matrix_entries = []
        for idx1, idx2 in np.array(idxs_upper_triangle).T:
            if idx1 == idx2:
                if msc.Constraint_coupling_zero(idx1,idx1) in self.enforced_constraints:
                    allowed_entries = [arch.NO_COUPLING]
                else:
                    allowed_entries = [arch.NO_COUPLING, arch.DETUNING]
            else:
                allowed_entries = [arch.NO_COUPLING, arch.COUPLING_WITHOUT_PHASE, arch.COUPLING_WITH_PHASE]
                if msc.Constraint_coupling_zero(idx1,idx2) in self.enforced_constraints:
                    allowed_entries.remove(arch.COUPLING_WITHOUT_PHASE)
                    allowed_entries.remove(arch.COUPLING_WITH_PHASE)
                elif msc.Constraint_coupling_phase_zero(idx1,idx2) in self.enforced_constraints:
                    allowed_entries.remove(arch.COUPLING_WITHOUT_PHASE)
                
                if type(self.operators[idx1]) != type(self.operators[idx2]):
                    if not self.phase_constraints_for_squeezing:
                        if arch.COUPLING_WITHOUT_PHASE in allowed_entries:
                            allowed_entries.remove(arch.COUPLING_WITHOUT_PHASE)

            self.possible_matrix_entries.append(allowed_entries)

            
            # allowed_entries = [arch.NO_COUPLING]
            # if msc.Constraint_coupling_zero(idx1,idx2) in self.enforced_constraints:
            #     pass #no further possiblities are added
            # elif idx1 == idx2:
            #     allowed_entries.append(arch.DETUNING)
            # elif msc.Constraint_coupling_phase_zero(idx1,idx2) in self.enforced_constraints:
            #     allowed_entries.append(arch.COUPLING_WITHOUT_PHASE)
            # else:
            #     if type(self.operators[idx1]) is type(self.operators[idx2]) or self.phase_constraints_for_squeezing:
            #         allowed_entries.append(arch.COUPLING_WITHOUT_PHASE)
            #     allowed_entries.append(arch.COUPLING_WITH_PHASE)

            # self.possible_matrix_entries.append(allowed_entries)


        self.list_of_upper_triangular_coupling_matrices = []
        self.complexity_levels = []
        for p_coupl in tqdm(product(*self.possible_matrix_entries)):
            # coupling_matrix = np.zeros([self.num_modes, self.num_modes])
            # coupling_matrix[idxs_upper_triangle] = p_coupl
            # coupling_matrix[idxs_lower_triangle] = (coupling_matrix.T)[idxs_lower_triangle]
            # list_of_coupling_matrices.append(coupling_matrix)
            self.list_of_upper_triangular_coupling_matrices.append(np.array(p_coupl, dtype='int8'))
            self.complexity_levels.append(sum(p_coupl))

        self.complexity_levels = np.array(self.complexity_levels)
        self.unique_complexity_levels = np.flip(sorted(np.unique(self.complexity_levels)))
        self.list_of_upper_triangular_coupling_matrices = np.array(self.list_of_upper_triangular_coupling_matrices)


    def find_valid_combinations(self, complexity_level, combinations_to_test=None):
        self.tested_complexities.append(complexity_level)
        newly_added_combos = []
        
        if combinations_to_test is None:
            potential_combinations = self.identify_potential_combinations(complexity_level)
        else:
            potential_combinations = combinations_to_test

        for combo_idx in trange(len(potential_combinations)):
            combo = potential_combinations[combo_idx]
            conditions = arch.translate_upper_triangle_coupling_matrix_to_conditions(combo)
            if not arch.check_if_subgraph_upper_triangle(combo, np.asarray(newly_added_combos)):
                success, all_infos, _ = self.repeated_optimization(constraints=conditions, **self.kwargs_optimization, **self.kwargs_solver)
                if success:
                    solution_array = all_infos[-1]['solution_array']
                    all_constraints = self.check_all_possible_constraints(solution_array)
                    self.valid_combinations_solutions.append(solution_array)

                    idxs_fulfilled_constraints = np.where(np.abs(all_constraints) < self.kwargs_optimization['max_violation_success'])[0]
                    fulfilled_constraints = [self.all_possible_constraints[idx] for idx in idxs_fulfilled_constraints]
                    # msc.print_combo(fulfilled_constraints)
                    valid_combo_to_add = arch.translate_conditions_to_upper_triangle_coupling_matrix(fulfilled_constraints, self.num_modes)
                    self.valid_combinations.append(valid_combo_to_add)
                    newly_added_combos.append(valid_combo_to_add)
                else:
                    self.invalid_combinations.append(combo)
    
    def identify_potential_combinations(self, complexity_level):
        all_idxs_with_desired_complexity = np.where(self.complexity_levels == complexity_level)[0]
        if len(all_idxs_with_desired_complexity) == 0:
            raise Warning('no architecture with the requrested complexity_level exists')
            return None
        
        potential_combinations = []
        for combo_idx in all_idxs_with_desired_complexity:
            coupling_matrix_combo = self.list_of_upper_triangular_coupling_matrices[combo_idx]

            #check if suggested graph is subgraph of an invalid graph
            cond1 = not arch.check_if_subgraph_upper_triangle(np.asarray(self.invalid_combinations), coupling_matrix_combo)

            #check if a valid architecture is a subgraph to the suggested graph 
            cond2 = not arch.check_if_subgraph_upper_triangle(coupling_matrix_combo, np.asarray(self.valid_combinations))

            if cond1 and cond2:
                potential_combinations.append(coupling_matrix_combo)

        return potential_combinations
    
    def cleanup_valid_combinations(self):
        all_unique_valid_combinations_array = np.unique(np.asarray(self.valid_combinations), axis=0)
        cleaned_valid_combinations = []
        num_valid_combinations = all_unique_valid_combinations_array.shape[0]

        for combo_idx, valid_combo in enumerate(all_unique_valid_combinations_array):
            idxs_combis_to_compare_against = np.setdiff1d(np.arange(num_valid_combinations),combo_idx)
            #check if any other of the valid architecture is a subgraph of the current architecture 
            if not arch.check_if_subgraph_upper_triangle(valid_combo, all_unique_valid_combinations_array[idxs_combis_to_compare_against]):
                cleaned_valid_combinations.append(valid_combo)
        
        self.valid_combinations = cleaned_valid_combinations

    def calc_number_of_potential_combinations(self, num_constraints):
        return len(self.identify_potential_combinations(num_constraints))

    def check_all_possible_constraints(self, x):
        return self.calc_all_possible_constraints(x)
    
    def calc_len_of_valid_combinations(self):
        return np.asarray([len(combo) for combo in self.valid_combinations])
    
def check_for_overlap_with_invalid_combination(combination, invalid_combinations, return_debug_info=False):
    #returns only True if there is no overlap
    combo = set(combination)
    for invalid_combo in invalid_combinations:
        if invalid_combo.issubset(combo):
            if return_debug_info:
                return False, invalid_combo
            return False
    if return_debug_info:
        return True, None
    else:
        return True

def check_if_not_subset_of_valid_combination(combination, valid_combinations, return_debug_info=False):
    #returns True if combination is not a subset of any element in valid_combinations
    combo = set(combination)
    for valid_combo in valid_combinations:
        if combo.issubset(valid_combo):
            if return_debug_info:
                return False, valid_combo
            else:
                return False
    if return_debug_info:
        return True, None
    else:
        return True

def cleanup_list_of_constraints(combo):
    combo_cleaned = combo.copy()
    for idx2 in range(len(combo)):
        for idx1 in range(idx2):
            constraint1 = combo[idx1]
            constraint2 = combo[idx2]
            if set([type(constraint1), type(constraint2)]) == set([msc.Constraint_coupling_zero, msc.Constraint_coupling_phase_zero]):
                if set(constraint1.idxs) == set(constraint2.idxs):
                    if type(constraint1) == msc.Constraint_coupling_phase_zero:
                        combo_cleaned.remove(constraint1)
                    if type(constraint2) == msc.Constraint_coupling_phase_zero:
                        combo_cleaned.remove(constraint2)
                
    return combo_cleaned

def print_combo(combo):
    for el in combo:
        print(el)