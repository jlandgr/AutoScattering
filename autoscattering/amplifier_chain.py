import numpy as np
import jax.numpy as jnp
import jax
import yaml
import inspect

from functools import partial

import multimode_systems.scattering as mss

def unpack_params(params):
    int_losses = jnp.concatenate((jnp.array([0.1]), params['int_losses_intermediate'], jnp.array([0.1])))
    ext_losses = jnp.zeros_like(int_losses)
    ext_losses = ext_losses.at[0].set(1)
    ext_losses = ext_losses.at[-1].set(1)
    params['ext_losses'] = ext_losses

    return mss.prepare_parameters_1D_chain(
        mode_freqs=params['mode_freqs'],
        on_site_adag_adag=params['on_site_adag_adag_abs'] * jnp.exp(1.j * params['on_site_adag_adag_phase']),
        coupling_adag_a=jnp.concatenate((
            params['coupling_adag_a_abs'] * jnp.exp(1.j * params['coupling_adag_a_phase']),
            jnp.array([0.])
        )),
        coupling_adag_adag=jnp.concatenate((
            params['coupling_adag_adag_abs'] * jnp.exp(1.j * params['coupling_adag_adag_phase']),
            jnp.array([0.])
        )),
        int_losses=int_losses,  # params['int_losses'],
        ext_losses=params['ext_losses']
    )

def calc_bandwidth(omegas, signal, threshold, max_idx=None):
    if max_idx is None:
        max_idx = jnp.argmax(signal)

    signal_rolled = jnp.roll(signal, -max_idx)

    argwhere_rolled = jnp.argwhere(signal_rolled < threshold, size=len(omegas))
    idx_left = max_idx - (len(omegas)-1 - jnp.max(argwhere_rolled))
    idx_right = max_idx + argwhere_rolled[0][0]

    bandwidth = omegas[idx_right] - omegas[idx_left]
    
    return bandwidth

def get_trainable_keys(setup_filename):
    with open(setup_filename, 'r') as fp:
        model_file = yaml.safe_load(fp)
    
    keys = list(model_file['initial_parameters'].keys())
    keys.sort()
    
    return keys

def select_particle(params, selected_idx):
    output_dict = {}
    for key in params.keys():
        if len(params[key].shape) < 2:
            raise Exception('invalid parameter shape')
        
        output_dict[key] = params[key][selected_idx]
    return output_dict

def merge_param_dicts(dict1, dict2):
    dict_combined = {}
    for key in dict1.keys():
        dict_combined[key] = np.concatenate((dict1[key], dict2[key]))
    return dict_combined

def initialize_system(setup_filename):
    
    with open(setup_filename, 'r') as fp:
        model_file = yaml.safe_load(fp)
        
    N_modes = model_file['system']['N_modes']
    
    omegas = np.linspace(
        model_file['omegas']['omega_start'],
        model_file['omegas']['omega_end'],
        model_file['omegas']['N_omegas']
    )

    return omegas, get_trainable_keys(setup_filename), N_modes

def calc_params_shape(key, N_modes):
    if key == 'mode_freqs' or key == 'on_site_adag_adag_abs' or key == 'on_site_adag_adag_phase':
        return N_modes
    elif key == 'coupling_adag_a_abs' or key == 'coupling_adag_a_phase':
        return N_modes - 1
    elif key == 'coupling_adag_adag_abs' or key == 'coupling_adag_adag_phase':
        return N_modes - 1
    elif key == 'int_losses_intermediate':
        return N_modes - 2
    else:
        raise Exception('invalid parameter keyword')
        
def give_params_bounds(key):
    if key == 'mode_freqs' or key == 'on_site_adag_adag_abs' or key == 'on_site_adag_adag_phase':
        return [-np.inf, np.inf]
    elif key == 'coupling_adag_a_abs' or key == 'coupling_adag_a_phase':
        return [-np.inf, np.inf]
    elif key == 'coupling_adag_adag_abs' or key == 'coupling_adag_adag_phase':
        return [-np.inf, np.inf]
    elif key == 'int_losses_intermediate':
        return [0., np.inf]
    else:
        raise Exception('invalid parameter keyword')
        
def give_bounds(keys, N_modes):
    bounds = np.zeros((calc_swarm_dimension(keys, N_modes), 2))
    start_idx = 0
    for key in keys:
        end_idx = start_idx + calc_params_shape(key, N_modes)
        bounds[start_idx:end_idx] = give_params_bounds(key)
        start_idx = end_idx
    return (bounds[:,0], bounds[:,1])

def calc_swarm_dimension(keys, N_modes):
    dimensions = 0
    for key in keys:
        dimensions += calc_params_shape(key, N_modes)
    return dimensions
    
def flatten_params_dict(params_trainable, N_modes):
    dimensions = calc_swarm_dimension(params_trainable, N_modes)
    keys_sorted = list(params_trainable.keys())
    keys_sorted.sort()

    flatten_shape = params_trainable[keys_sorted[0]].shape[:-1] + (dimensions,)
    params_flatten = np.zeros(flatten_shape)
    
    start_idx = 0
    for key in keys_sorted:
        end_idx = start_idx + calc_params_shape(key, N_modes)
        params_flatten[...,start_idx:end_idx] = params_trainable[key]
        start_idx = end_idx
        
    return params_flatten

def reshape_params_dict(params_flatten, keys, N_modes):
    keys.sort()
    
    params = {}
    start_idx = 0
    for key in keys:
        end_idx = start_idx + calc_params_shape(key, N_modes)
        params[key] = params_flatten[start_idx:end_idx]
        start_idx = end_idx
    
    return params

def check_stability(params_trainable, params_not_trainable, N_modes):
    
    keys_trainable = sorted(list(params_trainable.keys()))
    params_trainable_flatten = flatten_params_dict(params_trainable, N_modes)
    params_trainable_flatten_single = params_trainable_flatten[0]
    return check_stability_all_particles(jnp.array(params_trainable_flatten), params_not_trainable, tuple(keys_trainable), N_modes)

@partial(jax.jit, static_argnames=['keys_trainable', 'N_modes'])
def check_stability_single_particle(params_trainable_flatten_single, params_not_trainable, keys_trainable, N_modes):
    params_trainable_single = reshape_params_dict(params_trainable_flatten_single, list(keys_trainable), N_modes)
    params_in = {}
    params_in.update(params_trainable_single)
    params_in.update(params_not_trainable)
    dyn_matrix, _, _ = mss.create_dynamical_matrix(*unpack_params(params_in))
    eigvals = calc_eigvals_cpu(dyn_matrix)
    return ~jnp.any(jnp.real(eigvals)>0)

calc_eigvals_cpu = jax.jit(jnp.linalg.eigvals, device=jax.devices("cpu")[0])
check_stability_all_particles = jax.vmap(check_stability_single_particle, in_axes=[0]+(len(inspect.signature(check_stability_single_particle).parameters.values())-1)*[None])

def initialize_parameters(setup_filename, debug=False):
    
    def initialize_trainable_parameters():
        params = {}
        for key in get_trainable_keys(setup_filename):
            params[key] = jnp.zeros((0, calc_params_shape(key, N_modes)))
        return params
    
    def randomly_generate_trainable_parameters(num_particles):
        params = {}
        for key in get_trainable_keys(setup_filename):
            params[key] = jnp.asarray(
                np.random.uniform(
                    model_file['initial_parameters'][key][0],
                    model_file['initial_parameters'][key][1],
                    size=(num_particles, calc_params_shape(key, N_modes))
                )
            )
            
        return params
    
    with open(setup_filename, 'r') as fp:
        model_file = yaml.safe_load(fp)
        
    num_particles = model_file['training']['n_particles']
    num_particles_min_stable = model_file['training']['n_particles_min_stable']
    num_particles_max_unstable = num_particles - num_particles_min_stable
    
    N_modes = model_file['system']['N_modes']
    seed = model_file['seed']

    np.random.seed(seed)

    params_not_trainable = {}    
    for key in model_file['not_trainable'].keys():
        params_not_trainable[key] = jnp.asarray(np.full(
            (calc_params_shape(key, N_modes),),
            model_file['not_trainable'][key]
        ))
    
    i = 0
    params = initialize_trainable_parameters()
    added_particles = 0
    added_stable_particles = 0
    
    while list(params.values())[0].shape[0]<num_particles:
        
        particle_suggestions = randomly_generate_trainable_parameters(num_particles)
        stability = check_stability(particle_suggestions, params_not_trainable, N_modes)
        
        num_found_stable = np.sum(stability)
        num_found_unstable = np.sum(~stability)
        
        added_particles_here = min(num_particles-added_particles, num_found_stable)
        stable_particles_idxs = (np.arange(num_particles)[stability])[:added_particles_here]
        params = merge_param_dicts(params, select_particle(particle_suggestions, stable_particles_idxs))
        added_stable_particles += added_particles_here
        added_particles += added_particles_here
        
        
        added_particles_here = min(num_particles-added_particles, num_found_unstable, num_particles_max_unstable-(added_particles-added_stable_particles))
        unstable_particles_idxs = (np.arange(num_particles)[~stability])[:added_particles_here]
        params = merge_param_dicts(params, select_particle(particle_suggestions, unstable_particles_idxs))
        added_particles += added_particles_here
        
        if debug:
            print('initialization iteration %i: %i stable points found (minimum of %i required)'%(i, added_stable_particles, num_particles_min_stable))
        i = i+1
    
    params = select_particle(params, np.random.choice(np.arange(num_particles), size=num_particles, replace=False))
    print('finished initialization after %i rounds'%i)
    
    return params, params_not_trainable