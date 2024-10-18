import jax
import numpy as np
import jax.numpy as jnp
import sympy as sp
import scipy.optimize as sciopt
from tqdm import trange, tqdm
from itertools import product

import multimode_systems.constraints as msc
import multimode_systems.symbolic as sym
import multimode_systems.architecture as arch

from multimode_systems.architecture import translate_upper_triangle_coupling_matrix_to_conditions


AUTODIFF_FORWARD = 'autodiff_forward'
AUTODIFF_REVERSE = 'autodiff_reverse'
DIFFERENCE_QUOTIENT = '2-point'

VAR_ABS = 'abs_variable'
VAR_PHASE = 'phase_variable'
VAR_INTRINSIC_LOSS = 'intrinsic_loss_variable'
VAR_USER_DEFINED = 'user_defined'

ZERO_LOSS_MODE = 'zero_loss_mode'
LOSSY_MODE = 'lossy_mode'

INIT_ABS_RANGE_DEFAULT = [-1., 1.]
INIT_INTRINSIC_LOSS_RANGE_DEFAULT = [-1., 1.]
# BOUNDS_ABS_DEFAULT = [-np.inf, np.inf]
BOUNDS_INTRINSIC_LOSS_DEFAULT = [-np.inf, np.inf]


def calc_scattering_matrix_from_coupling_matrix(coupling_matrix, kappa_int_matrix):
    num_modes = coupling_matrix.shape[0]
    identity = jnp.eye(num_modes)
    scattering_matrix = identity + jnp.linalg.inv(-1.j*coupling_matrix - (identity+kappa_int_matrix)/2.)
    return scattering_matrix

def create_solutions_dict(variables, values):
    return {variable.name: value for variable, value in zip(variables, values)}

class Architecture_Optimizer():
    def __init__(
            self,
            S_target, num_auxiliary_modes, num_zero_loss_modes=0, mode_types='no_squeezing',
            gradient_method=AUTODIFF_REVERSE,
            signs_zero_loss_detunings=None,
            kwargs_optimization={},
            solver_options={},
            S_target_free_symbols_init_range=None,
            make_initial_test=True,
            phase_constraints_for_squeezing=False,
            free_gauge_phases=True,
            port_intrinsic_losses=False,
            method=None,
            enforced_constraints=[]):
        
        self.kwargs_optimization = {
            'num_tests': 10,
            'verbosity': 0,
            'init_abs_range': None,
            'init_intrinsinc_loss_range': None,
            'bounds_intrinsic_loss': None,
            'max_violation_success': 1.e-10,
            'init_intrinsinc_loss_range': None
        }
        self.kwargs_optimization.update(kwargs_optimization)
        self.kwargs_optimization['interrupt_if_successful'] = True

        if method is None:
            if self.kwargs_optimization['bounds_intrinsic_loss'] is None:
                method='BFGS'
            else:
                method='L-BFGS-B'

        self.kwargs_optimization['method'] = method
        
        if method == 'BFGS':
            self.solver_options = {'maxiter': 200}
        elif method == 'L-BFGS-B':
            self.solver_options = {'maxfun': np.inf, 'maxiter': 1000, 'ftol': 0., 'gtol': 0.}
        else:
            self.solver_options = {}
        self.solver_options.update(solver_options)

        self.S_target = S_target

        self.num_port_modes = S_target.shape[0]
        self.num_lossy_internal_modes = num_auxiliary_modes
        self.num_lossy_modes = self.num_port_modes + num_auxiliary_modes
        self.num_zero_loss_modes = num_zero_loss_modes
        self.num_modes = self.num_port_modes + self.num_lossy_internal_modes + self.num_zero_loss_modes
        self.mode_loss_info = [LOSSY_MODE]*self.num_lossy_modes + [ZERO_LOSS_MODE]*self.num_zero_loss_modes

        self.signs_zero_loss_detunings = signs_zero_loss_detunings
        self.free_gauge_phases = free_gauge_phases

        if port_intrinsic_losses is True:
            self.port_intrinsic_losses = [True for _ in range(self.num_port_modes)]
        elif port_intrinsic_losses is False:
            self.port_intrinsic_losses = [False for _ in range(self.num_port_modes)]
        else:
            self.port_intrinsic_losses = port_intrinsic_losses

        self.__prepare_mode_types__(mode_types)
        self.phase_constraints_for_squeezing = phase_constraints_for_squeezing
        self.enforced_constraints = enforced_constraints
        for idx2 in range(self.num_lossy_modes, self.num_modes):
            for idx1 in range(self.num_lossy_modes, idx2+1):
                constraint = msc.Coupling_Constraint(idx1, idx2)
                if not constraint in self.enforced_constraints:
                    self.enforced_constraints.append(constraint)
        self.__setup_all_constraints__()

        self.coupling_matrix = self.__initialize_coupling_matrix__()
        self.__initialize_parameters__(S_target_free_symbols_init_range)
        self.conditions_func = jax.jit(self.__initialize_conditions_func__())
        
        self.gradient_method = gradient_method
        if gradient_method == AUTODIFF_FORWARD:
            self.jacobian = jax.jit(jax.jacfwd(self.conditions_func, has_aux=True))
            # self.hessian = jax.jit(jax.jacfwd(self.jacobian), has_aux=True)
        elif gradient_method == AUTODIFF_REVERSE:
            self.jacobian = jax.jit(jax.jacrev(self.conditions_func, has_aux=True))
            # self.hessian = jax.jit(jax.jacrev(self.jacobian), has_aux=True)
        elif gradient_method == DIFFERENCE_QUOTIENT:
            self.jacobian = None
            self.hessian = None
        else:
            raise NotImplementedError()
        
        self.valid_combinations = []
        self.invalid_combinations = []
        self.tested_complexities = []
        
        # make run without additional conditions
        if make_initial_test:
            success, _, _ = self.repeated_optimization(conditions=[], **self.kwargs_optimization, **self.solver_options)
            if not success:
                raise(Exception('fully connected graph is invalid, interrupting'))
            else:
                print('fully connected graph is a valid graph')

    def __setup_all_constraints__(self):
        self.all_possible_constraints = []
        for idx in range(self.num_modes):
            self.all_possible_constraints.append(msc.Constraint_coupling_zero(idx, idx))
        for idx2 in range(self.num_modes):
            for idx1 in range(idx2):
                if type(self.operators[idx1]) is type(self.operators[idx2]):
                    self.all_possible_constraints.append(msc.Constraint_coupling_zero(idx1, idx2))
                    self.all_possible_constraints.append(msc.Constraint_coupling_phase_zero(idx1, idx2))
                else:
                    self.all_possible_constraints.append(msc.Constraint_coupling_zero(idx1, idx2))
                    if self.phase_constraints_for_squeezing:
                        self.all_possible_constraints.append(msc.Constraint_coupling_phase_zero(idx1, idx2))

    def check_all_constraints(self, coupling_matrix, kappa_int_matrix, max_violation):
        fulfilled_constraints = []
        for c in self.all_possible_constraints:
            if np.abs(c(None, coupling_matrix, kappa_int_matrix))**2/2 < max_violation:
                fulfilled_constraints.append(c)
        
        return arch.translate_conditions_to_upper_triangle_coupling_matrix(fulfilled_constraints, self.num_modes)

    def __prepare_mode_types__(self, mode_types):
        if mode_types == 'no_squeezing':
            self.mode_types = [True for _ in range(self.num_modes)]
        else:
            self.mode_types = mode_types
        self.operators = []
        for idx in range(self.num_modes):
            if self.mode_types[idx]:
                self.operators.append(sym.Mode().a)
            else:
                self.operators.append(sym.Mode().adag)

    def __init_gabs__(self, idx1, idx2, beamsplitter=True, append=False):
        if beamsplitter:
            varname = '|g_{'
        else:
            varname = '|\\nu_{'
        varname += str(idx1) + ','
        varname += str(idx2)
        varname += '}|'
        new_variable = sp.Symbol(varname, real=True)

        if append and not new_variable in self.gabs:
            self.gabs.append(new_variable)

        return new_variable
    
    def __init_gphase__(self, idx1, idx2, beamsplitter=True, append=False):
        if beamsplitter:
            varname = '\mathrm{arg}(g_{'
        else:
            varname = '\mathrm{arg}(\\nu_{'
        varname += str(idx1) + ','
        varname += str(idx2)
        varname += '})'
        new_variable = sp.Symbol(varname, real=True)

        if append and not new_variable in self.gphases:
            self.gphases.append(new_variable)

        return new_variable
    
    def __init_Delta__(self, idx, append=False):        
        new_variable = sp.Symbol('Delta%i'%idx, real=True)
        if append and not new_variable in self.Deltas:
            self.Deltas.append(new_variable)

        return new_variable
    
    def __give_coupling_element__(self, idx1, idx2, operators, with_phase=True, append=False):
        op1 = operators[idx1]
        op2 = operators[idx2]

        idxmin, idxmax = min(idx1, idx2), max(idx1, idx2)

        # detuning
        if idx1 == idx2:
            detuning = self.__init_Delta__(idxmin, append=append)
            if isinstance(op1, sym.Annihilation_operator):
                return - detuning
            else:
                return detuning

        # beamsplitter
        elif type(op1) == type(op2):
            gabs = self.__init_gabs__(idxmin, idxmax, beamsplitter=True, append=append)
            if with_phase:
                gphase = self.__init_gphase__(idxmin, idxmax, beamsplitter=True, append=append)
            else:
                gphase = sp.S(0)
            coupling = gabs * sp.exp(sp.I*gphase)
            if idx1 < idx2:
                pass
            else:
                coupling = sp.conjugate(coupling) #beamsplitter coupling matrix is Hermitian

            if isinstance(op1, sym.Annihilation_operator):
                return coupling
            else:
                return - sp.conjugate(coupling)
            
        # squeezing
        else:
            gabs = self.__init_gabs__(idxmin, idxmax, beamsplitter=False, append=append)
            if with_phase:
                gphase = self.__init_gphase__(idxmin, idxmax, beamsplitter=False, append=append)
            else:
                gphase = sp.S(0)
            coupling = gabs * sp.exp(sp.I*gphase)
            if isinstance(op1, sym.Annihilation_operator):
                return coupling
            else:
                return -sp.conjugate(coupling)

    def __initialize_coupling_matrix__(self):
        append = True
        self.gabs = []
        self.gphases = []
        self.Deltas = []
        coupling_matrix_dimensionless = sp.zeros(self.num_modes)
        
        for idx in range(self.num_modes):            
            if not msc.Constraint_coupling_zero(idx, idx) in self.enforced_constraints:
                coupling_matrix_dimensionless[idx, idx] = self.__give_coupling_element__(idx, idx, operators=self.operators, append=append)
        
        for idx1 in range(self.num_modes):
            for idx2 in range(self.num_modes):
                if idx1 != idx2:
                    if not msc.Constraint_coupling_zero(idx1, idx2) in self.enforced_constraints:
                        if not msc.Constraint_coupling_phase_zero(idx1, idx2) in self.enforced_constraints:
                            with_phase = True
                        else:
                            with_phase = False
                        coupling_matrix_dimensionless[idx1, idx2] = self.__give_coupling_element__(idx1, idx2, with_phase=with_phase, operators=self.operators, append=append)

        return coupling_matrix_dimensionless

    # def __init_gabs__(self, idx1, idx2):
    #     idx1, idx2 = sorted([idx1, idx2])
    #     return sp.Symbol('gabs%i%i'%(idx1,idx2), real=True)
    
    # def __init_gphase__(self, idx1, idx2):
    #     idx1, idx2 = sorted([idx1, idx2])
    #     return sp.Symbol('gphase%i%i'%(idx1,idx2), real=True)
    
    # def __init_Delta__(self, idx):
    #     return sp.Symbol('Delta%i'%idx, real=True)

    # def __initialize_coupling_matrix__(self):
    #     # initializes the coupling matrix

    #     self.Deltas = []
    #     self.gabs = [] 
    #     self.gphases = [] 

    #     if self.signs_zero_loss_detunings is None:
    #         self.signs_zero_loss_detunings = np.ones(self.num_zero_loss_modes, dtype='int')
    #     else:
    #         self.signs_zero_loss_detunings = np.array(self.signs_zero_loss_detunings)
    #     # coupling_matrix_not_lossy = sp.diag(*self.signs_zero_loss_detunings.tolist())

    #     coupling_matrix = sp.zeros(self.num_modes)
    #     for idx in range(self.num_modes):
    #         if self.mode_loss_info[idx] == LOSSY_MODE:
    #             if not msc.Constraint_coupling_zero(idx, idx) in self.enforced_constraints:
    #                 Delta = self.__init_Delta__(idx)
    #                 self.Deltas.append(Delta)
    #                 coupling_matrix[idx,idx] = Delta
    #             else:
    #                 coupling_matrix[idx,idx] = 0
    #         else:
    #             coupling_matrix[idx,idx] = self.signs_zero_loss_detunings[idx-self.num_lossy_modes]
            
    #     for idx2 in range(self.num_modes):
    #         for idx1 in range(idx2):
    #             if type(self.operators[idx1]) is type(self.operators[idx2]):
    #                 sign = sp.S(1)
    #             else:
    #                 sign = -sp.S(1)
                
    #             gabs = self.__init_gabs__(idx1, idx2)
    #             gphase = self.__init_gphase__(idx1, idx2)

    #             if self.mode_loss_info[idx1] != ZERO_LOSS_MODE or self.mode_loss_info[idx2] != ZERO_LOSS_MODE:
    #                 if msc.Constraint_coupling_zero(idx1, idx2) in self.enforced_constraints:
    #                     coupling_matrix_idx1_idx2 = 0
    #                 elif msc.Constraint_coupling_phase_zero(idx1, idx2) in self.enforced_constraints:
    #                     coupling_matrix_idx1_idx2 = gabs
    #                     self.gabs.append(gabs)
    #                 else:
    #                     coupling_matrix_idx1_idx2 = gabs * sp.exp(sp.I * gphase)
    #                     self.gabs.append(gabs)
    #                     self.gphases.append(gphase)
                    
    #                 coupling_matrix[idx1,idx2] = coupling_matrix_idx1_idx2
    #                 coupling_matrix[idx2,idx1] = sign*sp.conjugate(coupling_matrix_idx1_idx2)
    #             else:
    #                 pass
        
    #     self.coupling_matrix = coupling_matrix

    def __initialize_parameters__(self, S_target_free_symbols_init_range):

        # free parameters in S_target
        if S_target_free_symbols_init_range is None:
            self.parameters_S_target = list(self.S_target.free_symbols)
        else:
            self.parameters_S_target = list(S_target_free_symbols_init_range.keys())
        for var in self.parameters_S_target:
            if not var.is_real:
                raise Exception('variable '+var.name+' is complex, only real variables are allowed')
        
        # free Gauge phases
        if self.free_gauge_phases:
            self.gauge_phases = [sp.Symbol('gauge%i'%idx, real=True) for idx in range(self.num_port_modes)]
            self.gauge_factors = [sp.exp(sp.I*phase) for phase in self.gauge_phases]
            self.gauge_matrix = sp.zeros(self.num_port_modes)
            for idx1 in range(self.num_port_modes):
                for idx2 in range(self.num_port_modes):
                    self.gauge_matrix[idx1,idx2] = self.gauge_factors[idx1]*sp.conjugate(self.gauge_factors[idx2])
            # S_target_used = sp.matrices.dense.matrix_multiply_elementwise(self.S_target, self.gauge_matrix)
        else:
            self.gauge_phases = []
            self.gauge_factors, self.gauge_factors, self.gauge_matrix = None, None, None
            # S_target_used = self.S_target
        
        # internal losses
        self.variables_intrinsic_losses = []
        kappa_int_matrix_diag = []
        for mode_idx in range(self.num_port_modes):
            if self.port_intrinsic_losses[mode_idx]:
                kappa_int = sp.Symbol('kappa_int%i'%mode_idx, real=True)
                self.variables_intrinsic_losses.append(kappa_int)
                kappa_int_matrix_diag.append(kappa_int)
            else:
                kappa_int_matrix_diag.append(sp.S(0))
        kappa_int_matrix_diag += [sp.S(0) for _ in range(self.num_lossy_internal_modes)] 
        self.kappa_int_matrix = sp.diag(*kappa_int_matrix_diag)

        self.all_variables_list = \
            self.Deltas + \
            self.gabs + \
            self.gphases + \
            self.gauge_phases + \
            self.parameters_S_target + \
            self.variables_intrinsic_losses 
        self.all_variables_types = \
            [VAR_ABS]*len(self.Deltas+self.gabs) +\
            [VAR_PHASE]*len(self.gphases) + \
            [VAR_PHASE]*len(self.gauge_phases) + \
            [VAR_USER_DEFINED]*len(self.parameters_S_target) + \
            [VAR_INTRINSIC_LOSS]*len(self.variables_intrinsic_losses) 
        self.S_target_free_symbols_init_range = S_target_free_symbols_init_range

        self.S_target_jax = sp.utilities.lambdify(self.all_variables_list, self.S_target, modules='jax') 
        self.coupling_matrix_jax = sp.utilities.lambdify(self.all_variables_list, self.coupling_matrix, modules='jax') 
        self.kappa_int_matrix_jax = sp.utilities.lambdify(self.all_variables_list, self.kappa_int_matrix, modules='jax') 
        if self.free_gauge_phases:
            self.gauge_matrix_jax = sp.utilities.lambdify(self.all_variables_list, self.gauge_matrix, modules='jax') 
        else:
            self.gauge_matrix_jax = None
    
    def create_initial_guess(self, conditions=[], init_abs_range=None, phase_range=None, init_intrinsinc_loss_range=None):
        if init_abs_range is None:
            init_abs_range = [-1., 1.]
        if phase_range is None:
            phase_range = [-np.pi, np.pi]
        if init_intrinsinc_loss_range is None:
            init_intrinsinc_loss_range = INIT_INTRINSIC_LOSS_RANGE_DEFAULT

        idxs_free = self.give_free_variable_idxs(conditions)
        
        random_guess = []
        for var_idx, var_type in enumerate(self.all_variables_types):
            if var_type == VAR_ABS:
                random_guess.append(np.random.uniform(init_abs_range[0], init_abs_range[-1]))
            elif var_type == VAR_PHASE:
                random_guess.append(np.random.uniform(phase_range[0], phase_range[-1]))
            elif var_type == VAR_INTRINSIC_LOSS:
                random_guess.append(np.random.uniform(init_intrinsinc_loss_range[0], init_intrinsinc_loss_range[-1]))
            elif var_type == VAR_USER_DEFINED:
                if self.S_target_free_symbols_init_range is None:
                    user_defined_range = [-np.pi, np.pi]
                else:
                    user_defined_range = self.S_target_free_symbols_init_range[self.all_variables_list[var_idx]]
                random_guess.append(np.random.uniform(user_defined_range[0], user_defined_range[-1]))
        
        return np.array(random_guess)[idxs_free], idxs_free

    def setup_bounds(self, bounds_intrinsic_loss=None, free_idxs=None):
        if bounds_intrinsic_loss is None:
            return None
        else:
            bounds = []
            for var_type in self.all_variables_types:
                if var_type==VAR_ABS or var_type == VAR_PHASE or var_type == VAR_USER_DEFINED:
                    bounds.append([-np.inf, np.inf])
                elif var_type == VAR_INTRINSIC_LOSS:
                    bounds.append(bounds_intrinsic_loss)
                else:
                    raise NotImplementedError()
            
            return np.asarray(bounds)[free_idxs]

    def __initialize_conditions_func__(self):
        self.enforced_constraints_beyond_coupling_constraint = []
        for c in self.enforced_constraints:
            if not isinstance(c, msc.Coupling_Constraint):
                self.enforced_constraints_beyond_coupling_constraint.append(c)

        if self.num_zero_loss_modes > 0:
            def coupling_matrix_effective(input_array):
                coupling_matrix = self.coupling_matrix_jax(*input_array)
                coupling_matrix_lossy_modes = coupling_matrix[:self.num_lossy_modes,:self.num_lossy_modes]
                coupling_matrix_lossy_not_lossy = coupling_matrix[:self.num_lossy_modes,self.num_lossy_modes:]
                coupling_matrix_not_lossy_lossy = coupling_matrix[self.num_lossy_modes:,:self.num_lossy_modes]
                coupling_matrix_not_lossy = coupling_matrix[self.num_lossy_modes:,self.num_lossy_modes:]

                return coupling_matrix_lossy_modes - coupling_matrix_lossy_not_lossy @ jnp.linalg.inv(coupling_matrix_not_lossy) @ coupling_matrix_not_lossy_lossy
        else:
            def coupling_matrix_effective(input_array):
                return self.coupling_matrix_jax(*input_array)
        
        self.coupling_matrix_effective = coupling_matrix_effective

        if self.free_gauge_phases:
            def calc_target_scattering_matrix(input_array):
                return jnp.multiply(self.S_target_jax(*input_array), self.gauge_matrix_jax(*input_array))
        else:
            def calc_target_scattering_matrix(input_array):
                return self.S_target_jax(*input_array)
        
        def calc_conditions(input_array):
            scattering_matrix_target = calc_target_scattering_matrix(input_array)
            coupling_matrix = self.coupling_matrix_effective(input_array)
            kappa_int_matrix = self.kappa_int_matrix_jax(*input_array)
            scattering_matrix = calc_scattering_matrix_from_coupling_matrix(coupling_matrix, kappa_int_matrix)

            shape_target = self.S_target.shape
            difference = (scattering_matrix[:shape_target[0],:shape_target[1]] - scattering_matrix_target).flatten()

            if len(self.enforced_constraints_beyond_coupling_constraint) > 0:
                full_coupling_matrix = self.coupling_matrix_jax(*input_array)
                additional_constraints = jnp.hstack([c(scattering_matrix, full_coupling_matrix, kappa_int_matrix) for c in self.enforced_constraints_beyond_coupling_constraint])
                evaluated_conditions = jnp.hstack((jnp.real(difference), jnp.imag(difference), additional_constraints))
            else:
                evaluated_conditions = jnp.hstack((jnp.real(difference), jnp.imag(difference)))
            return jnp.sum(jnp.abs(evaluated_conditions)**2)/2, {'scattering_matrix': scattering_matrix, 'coupling_matrix': coupling_matrix}

        return calc_conditions

    def calc_scattering_matrix_from_parameter_dictionary(self, parameter_dictionary):
        input_array = np.array([parameter_dictionary[var.name] for var in self.all_variables_list])
        coupling_matrix = self.coupling_matrix_jax(*input_array)
        kappa_int_matrix = self.kappa_int_matrix_jax(*input_array)
        scattering_matrix = calc_scattering_matrix_from_coupling_matrix(coupling_matrix, kappa_int_matrix)
        return scattering_matrix

    def give_free_variable_idxs(self, conditions):
        free_variable_idxs = [idx for idx in range(len(self.all_variables_list))]
        for c in conditions:
            if type(c) == msc.Constraint_coupling_zero:
                if c.idxs[0] != c.idxs[1]:
                    beamsplitter = self.mode_types[c.idxs[0]] == self.mode_types[c.idxs[1]]
                    gabs = self.__init_gabs__(c.idxs[0], c.idxs[1], beamsplitter=beamsplitter)
                    gphase = self.__init_gphase__(c.idxs[0], c.idxs[1], beamsplitter=beamsplitter)
                    if gabs in self.all_variables_list:
                        free_variable_idxs.remove(self.all_variables_list.index(gabs))
                    if gphase in self.all_variables_list:
                        free_variable_idxs.remove(self.all_variables_list.index(gphase))
                else:
                    Delta = self.__init_Delta__(c.idxs[0])
                    if Delta in self.all_variables_list:
                        free_variable_idxs.remove(self.all_variables_list.index(Delta))
            elif type(c) == msc.Constraint_coupling_phase_zero:
                gphase = self.__init_gphase__(c.idxs[0], c.idxs[1])
                if gphase in self.all_variables_list:
                    free_variable_idxs.remove(self.all_variables_list.index(gphase))
            else:
                raise Exception('only architectural constraints are allowed')
            
        return free_variable_idxs

    def give_conditions_func_with_conditions(self, conditions):
        idxs_free_variables = self.give_free_variable_idxs(conditions)
        np_idxs_free_variables = np.array(idxs_free_variables)

        num_total_variables = len(self.all_variables_list)
        def calc_conditions_constrained(partial_input_array):
            full_input_array = np.zeros(num_total_variables)
            full_input_array[idxs_free_variables] = partial_input_array
                
            return self.conditions_func(full_input_array)
        
        if self.gradient_method == DIFFERENCE_QUOTIENT:
            calc_jacobian = '2-point'
            calc_hessian = '2-point'
        else:
            def calc_jacobian_constrained(partial_input_array):
                full_input_array = np.zeros(num_total_variables)
                full_input_array[idxs_free_variables] = partial_input_array
                # return self.jacobian(full_input_array)[jnp.array(idxs_free_variables)]
                jacobian, aux_dict = self.jacobian(full_input_array)
                return np.array(jacobian)[np_idxs_free_variables]
            
            def calc_hessian_constrained(partial_input_array):
                raise NotImplementedError()
                # full_input_array = np.zeros(num_total_variables)
                # full_input_array[idxs_free_variables] = partial_input_array
                # return np.take(np.take(np.array(self.hessian(full_input_array)), np_idxs_free_variables, axis=0), np_idxs_free_variables, axis=1)
            
            calc_jacobian = calc_jacobian_constrained
            calc_hessian = calc_hessian_constrained
  
        return calc_conditions_constrained, calc_jacobian, calc_hessian

    def complete_variable_arrays_with_zeros(self, variable_array, conditions):
        free_variable_idxs = self.give_free_variable_idxs(conditions)
        complete_variable_array = np.zeros(len(self.all_variables_list))
        complete_variable_array[free_variable_idxs] = variable_array
        return complete_variable_array

    def optimize_given_conditions(self,
                conditions=None, triu_matrix=None, verbosity=False,
                init_abs_range=INIT_ABS_RANGE_DEFAULT,
                init_intrinsinc_loss_range=INIT_INTRINSIC_LOSS_RANGE_DEFAULT,
                bounds_intrinsic_loss=BOUNDS_INTRINSIC_LOSS_DEFAULT,
                max_violation_success=1.e-5, 
                calc_conditions_and_gradients=None,
                method=None,
                **kwargs_solver
            ):
        if conditions is None:
            conditions = translate_upper_triangle_coupling_matrix_to_conditions(triu_matrix)
        
        if calc_conditions_and_gradients is None:
            calc_conditions, calc_gradients, _ = self.give_conditions_func_with_conditions(conditions)
        else:
            calc_conditions, calc_gradients, _ = calc_conditions_and_gradients

        initial_guess, free_idxs = self.create_initial_guess(conditions=conditions, init_abs_range=init_abs_range, init_intrinsinc_loss_range=init_intrinsinc_loss_range)
        if method is None:
            if bounds_intrinsic_loss is None:
                method = 'BFGS'
            elif bounds_intrinsic_loss is not None:
                method = 'L-BFGS-B'

        bounds = self.setup_bounds(bounds_intrinsic_loss, free_idxs)

        parameter_history = []
        scattering_matrix_history = []
        loss_history = []

        def callback(Xi, *args):
            
            loss, aux_dict = calc_conditions(Xi)
            parameter_history.append(Xi)
            scattering_matrix_history.append(aux_dict['scattering_matrix'])
            loss_history.append(loss)
            if loss < max_violation_success:
                raise StopIteration('loss below threshold for success')

        xsol = sciopt.minimize(lambda x: calc_conditions(x)[0], initial_guess, jac=calc_gradients, bounds=bounds, callback=callback, method=method, options=kwargs_solver)

        success = np.all(np.abs(xsol['fun']) < self.kwargs_optimization['max_violation_success'])
        solution_complete_array = self.complete_variable_arrays_with_zeros(xsol.x, conditions)
        solution_effective_coupling_matrix = self.coupling_matrix_effective(solution_complete_array)
        kappa_int_matrix = self.kappa_int_matrix_jax(*solution_complete_array)

        scattering_matrix_target_func = self.S_target_jax(*solution_complete_array)
        if self.free_gauge_phases:
            gauge_matrix = self.gauge_matrix_jax(*solution_complete_array)
            scattering_matrix_target_times_gauge_matrix = scattering_matrix_target_func*gauge_matrix
        else:
            gauge_matrix = None
            scattering_matrix_target_times_gauge_matrix = scattering_matrix_target_func

        solution_dict = self.dict_extract_relevant_information(xsol.x, conditions)
        info_out = {
            'initial_guess': self.complete_variable_arrays_with_zeros(initial_guess, conditions),
            'free_idxs': free_idxs,
            'solution': solution_complete_array,
            'solution_dict_complete': create_solutions_dict(self.all_variables_list, solution_complete_array),
            'solution_dict': solution_dict,
            'parameters_for_analysis': self.extract_cooperativities_and_human_defined_parameters(conditions, solution_dict),
            'final_cost': xsol['fun'],
            'success': success,
            'optimizer_message': xsol['message'],
            'effective_coupling_matrix': solution_effective_coupling_matrix,
            'coupling_matrix': self.coupling_matrix_jax(*solution_complete_array),
            'kappa_int_matrix': self.kappa_int_matrix_jax(*solution_complete_array),
            'scattering_matrix': calc_scattering_matrix_from_coupling_matrix(solution_effective_coupling_matrix, kappa_int_matrix),
            'scattering_matrix_target_func': scattering_matrix_target_func,
            'scattering_matrix_target_times_gauge_matrix': scattering_matrix_target_times_gauge_matrix,
            'gauge_matrix': gauge_matrix,
            'nit': xsol['nit'],
            'parameter_history': parameter_history,
            'loss_history': loss_history,
            'scattering_matrix_history': scattering_matrix_history,
            'bounds': bounds,
        }

        return success, info_out
    
    def repeated_optimization(self,
                num_tests, conditions=None, triu_matrix=None, verbosity=False,
                init_abs_range=INIT_ABS_RANGE_DEFAULT,
                init_intrinsinc_loss_range=INIT_INTRINSIC_LOSS_RANGE_DEFAULT,
                bounds_intrinsic_loss=BOUNDS_INTRINSIC_LOSS_DEFAULT,
                max_violation_success=1.e-5,
                interrupt_if_successful=True,
                **kwargs_solver
            ):
        
        if conditions is None:
            conditions = translate_upper_triangle_coupling_matrix_to_conditions(triu_matrix)
        
        calc_conditions_and_gradients = self.give_conditions_func_with_conditions(conditions)

        successes = []
        infos = []
        for _ in range(num_tests):
            success, info = self.optimize_given_conditions(
                conditions=conditions, triu_matrix=triu_matrix, verbosity=verbosity,
                init_abs_range=init_abs_range,
                init_intrinsinc_loss_range=init_intrinsinc_loss_range,
                bounds_intrinsic_loss=bounds_intrinsic_loss,
                max_violation_success=max_violation_success,
                calc_conditions_and_gradients=calc_conditions_and_gradients,
                **kwargs_solver
            )
            successes.append(success)
            infos.append(info)

            if success and interrupt_if_successful:
                break
        
        return np.any(successes), infos, np.where(successes)
    
    def prepare_all_possible_combinations(self):
        idxs_upper_triangle = np.triu_indices(self.num_modes)

        self.possible_matrix_entries = []
        for idx1, idx2 in np.array(idxs_upper_triangle).T:
            if idx1 == idx2:
                if self.mode_loss_info[idx1] == ZERO_LOSS_MODE:
                    allowed_entries = [arch.DETUNING]
                elif msc.Constraint_coupling_zero(idx1,idx1) in self.enforced_constraints:
                    allowed_entries = [arch.NO_COUPLING]
                else:
                    allowed_entries = [arch.NO_COUPLING, arch.DETUNING]
            else:
                if self.mode_loss_info[idx1] == ZERO_LOSS_MODE and self.mode_loss_info[idx2] == ZERO_LOSS_MODE:
                    allowed_entries = [arch.NO_COUPLING]
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

        self.list_of_upper_triangular_coupling_matrices = []
        self.complexity_levels = []
        for p_coupl in tqdm(product(*self.possible_matrix_entries)):
            self.list_of_upper_triangular_coupling_matrices.append(np.array(p_coupl, dtype='int8'))
            self.complexity_levels.append(sum(p_coupl))

        self.complexity_levels = np.array(self.complexity_levels)
        self.unique_complexity_levels = np.flip(sorted(np.unique(self.complexity_levels)))
        self.list_of_upper_triangular_coupling_matrices = np.array(self.list_of_upper_triangular_coupling_matrices)
        self.num_possible_graphs = len(self.list_of_upper_triangular_coupling_matrices)


    def find_valid_combinations(self, complexity_level, combinations_to_test=None, perform_graph_reduction_of_successfull_graphs=True):
        
        newly_added_combos = []
        
        if combinations_to_test is None:
            potential_combinations = self.identify_potential_combinations(complexity_level)
        else:
            potential_combinations = combinations_to_test

        count_tested = 0

        for combo_idx in trange(len(potential_combinations)):
            combo = potential_combinations[combo_idx]
            conditions = arch.translate_upper_triangle_coupling_matrix_to_conditions(combo)
            if not arch.check_if_subgraph_upper_triangle(combo, np.asarray(newly_added_combos)):
                success, all_infos, _ = self.repeated_optimization(conditions=conditions, **self.kwargs_optimization, **self.solver_options)
                count_tested += 1
                if success:
                    if perform_graph_reduction_of_successfull_graphs:
                        valid_combo_to_add = self.check_all_constraints(all_infos[-1]['coupling_matrix'], all_infos[-1]['kappa_int_matrix'], self.kwargs_optimization['max_violation_success'])
                    else:
                        valid_combo_to_add = combo
                    self.valid_combinations.append(valid_combo_to_add)
                    newly_added_combos.append(valid_combo_to_add)
                else:
                    self.invalid_combinations.append(combo)
        
        self.tested_complexities.append([complexity_level, count_tested])
    
    def identify_potential_combinations(self, complexity_level, skip_check_for_valid_subgraphs=False):
        all_idxs_with_desired_complexity = np.where(self.complexity_levels == complexity_level)[0]
        if len(all_idxs_with_desired_complexity) == 0:
            raise Warning('no architecture with the requrested complexity_level exists')
            return None
        
        potential_combinations = []
        for combo_idx in all_idxs_with_desired_complexity:
            coupling_matrix_combo = self.list_of_upper_triangular_coupling_matrices[combo_idx]

            #check if suggested graph is subgraph of an invalid graph
            cond1 = not arch.check_if_subgraph_upper_triangle(np.asarray(self.invalid_combinations), coupling_matrix_combo)

            if cond1:
                #check if a valid architecture is a subgraph to the suggested graph 
                if skip_check_for_valid_subgraphs:
                    cond2 = True
                else:
                    cond2 = not arch.check_if_subgraph_upper_triangle(coupling_matrix_combo, np.asarray(self.valid_combinations))
                
                if cond2:
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

    def perform_depth_first_search(self):
        print('prepare list of all possible graphs')
        self.prepare_all_possible_combinations()
        print('%i graphs identified'%len(self.list_of_upper_triangular_coupling_matrices))
        print('start depth-first search')
        for c in self.unique_complexity_levels:
            print('test all graphs with %i degrees of freedom:'%c)
            self.find_valid_combinations(c)
            self.cleanup_valid_combinations()
        print('optimisation finished, list of irreducible graphs has %i elements'%len(self.valid_combinations))
        return np.array(self.valid_combinations, dtype='int8')
    
    def dict_extract_relevant_information(self, solution_array, conditions=None, triu_matrix=None):
        if conditions is None:
            conditions = translate_upper_triangle_coupling_matrix_to_conditions(triu_matrix)
        free_idxs = self.give_free_variable_idxs(conditions)
        free_variables = [self.all_variables_list[idx] for idx in free_idxs]
        return create_solutions_dict(free_variables, solution_array)

    def extract_cooperativities_and_human_defined_parameters(self, conditions, solution_dict):
        cooperativity_dict = {}
        
        free_idxs = self.give_free_variable_idxs(conditions)
        free_variables = [self.all_variables_list[idx] for idx in free_idxs]

        for idx1 in range(self.num_modes):
            for idx2 in range(self.num_modes):
                key = self.__init_gabs__(idx1, idx2, beamsplitter=True)
                if key in free_variables:
                    cooperativity = 4 * np.abs(solution_dict[key.name])**2
                    cooperativity_dict['C_{%i,%i}'%(idx1, idx2)] = cooperativity
                key = self.__init_gabs__(idx1, idx2, beamsplitter=False)
                if key in free_variables:
                    cooperativity = 4 * np.abs(solution_dict[key.name])**2
                    cooperativity_dict['C_{%i,%i}'%(idx1, idx2)] = cooperativity

        for var in self.parameters_S_target:
            key = var.name
            cooperativity_dict[key] = solution_dict[key]

        return cooperativity_dict