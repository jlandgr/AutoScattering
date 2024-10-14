import numpy as np
import jax
import jax.numpy as jnp
import yaml
from functools import partial
import os

import multimode_systems.scattering as mss
import multimode_systems.jax_functions as jf

def unpack_params(params_trainable, params_not_trainable):
    params = {}
    params.update(params_trainable)
    params.update(params_not_trainable)

    def calc_mode_idx(path_idx, mode_in_path_idx):
        mode_idx = 1 + path_idx * modes_per_path + mode_in_path_idx
        return mode_idx

    coupling_parallel = params['coupling_parallel_abs'] * jnp.exp(1.j * params['coupling_parallel_phase'])
    coupling_transversal = params['coupling_transversal_abs'] * jnp.exp(1.j * params['coupling_transversal_phase'])
    int_losses = params['trans_mode_int_losses']  # jnp.exp(params['log_int_losses'])

    N_paths, modes_per_path = params['trans_mode_freqs'].shape
    N_modes = 2 + N_paths * modes_per_path

    system = mss.Multimode_system(N_modes, jnp.array([0, -1]))

    system.add_adag_a_coupling(params['target_mode_freqs'][0], 0, 0)
    system.add_adag_a_coupling(params['target_mode_freqs'][-1], -1, -1)

    for path_idx in range(N_paths):
        for mode_in_path_idx in range(modes_per_path):
            mode_idx = calc_mode_idx(path_idx, mode_in_path_idx)
            system.add_intrinsic_loss(int_losses[path_idx, mode_in_path_idx], mode_idx)
            system.add_adag_a_coupling(params['trans_mode_freqs'][path_idx, mode_in_path_idx], mode_idx, mode_idx)

    for path_idx in range(N_paths):
        system.add_adag_a_coupling(
            coupling_parallel[path_idx, 0],
            0, calc_mode_idx(path_idx, 0)
        )

        system.add_adag_a_coupling(
            coupling_parallel[path_idx, -1],
            calc_mode_idx(path_idx, modes_per_path - 1), -1
        )

        for mode_in_path_idx in range(modes_per_path - 1):
            system.add_adag_a_coupling(
                coupling_parallel[path_idx, mode_in_path_idx + 1],
                calc_mode_idx(path_idx, mode_in_path_idx), calc_mode_idx(path_idx, mode_in_path_idx + 1)
            )

    for mode_in_path_idx in range(modes_per_path):
        for path_idx in range(N_paths-1):
            system.add_adag_a_coupling(
                coupling_transversal[path_idx, mode_in_path_idx],
                calc_mode_idx(path_idx, mode_in_path_idx), calc_mode_idx(path_idx + 1, mode_in_path_idx)
            )
    return system.return_system_parameters()

def calc_bandwidth(omegas, signal, threshold, heaviside_approx, max_idx=None, max_val=1.):
    if max_idx is None:
        max_idx = jnp.argmax(signal)
    if max_val is None:
        raise NotImplementedError()

    signal_rolled = jnp.roll(signal, -max_idx)
    heaviside_result_rolled = heaviside_approx((signal_rolled - threshold) / (max_val - threshold))

    cumprod = jnp.hstack((
        jnp.flip(np.cumprod(jnp.flip(heaviside_result_rolled))),
        jnp.cumprod(heaviside_result_rolled)
    ))

    cumprod = jax.lax.dynamic_slice(cumprod, (len(signal) - max_idx,), (len(signal),))

    return jf.integrate(omegas, cumprod)


@partial(jax.jit, static_argnames=['omegas', 'target', 'heaviside_approx'])
def loss(omegas, params_trainable, params_not_trainable, target, threshold, heaviside_approx, deviation_control=1.):
    omegas = jnp.array(omegas)
    scattering_matrix, info_dict_scatt = mss.calc_scattering_matrix(
        omegas, *unpack_params(params_trainable, params_not_trainable)
    )

    deviation = target.calc_deviation(scattering_matrix)

    info_dict = {}

    bandwidth = calc_bandwidth(
        omegas,
        signal=(1 - target.calc_deviation(scattering_matrix) / 4.),
        threshold=threshold,
        heaviside_approx=heaviside_approx,
        max_idx=len(omegas) // 2  # jnp.argmin(deviation)
    )

    min_deviation = jnp.min(deviation)
    info_dict['bandwidth'] = bandwidth
    info_dict['min_deviation'] = min_deviation
    loss = jnp.exp(-deviation_control) * deviation[len(omegas) // 2] - bandwidth

    return loss, info_dict

def initialize_system(setup_filename):
    with open(setup_filename, 'r') as fp:
        model_file = yaml.safe_load(fp)

    omegas = np.linspace(
        model_file['omegas']['omega_start'],
        model_file['omegas']['omega_end'],
        model_file['omegas']['N_omegas']
    )

    N_paths = model_file['system']['N_paths']
    modes_per_path = model_file['system']['modes_per_path']
    N_modes = 2 + N_paths * modes_per_path
    seed = model_file['seed']

    def calc_params_shape(key):
        if key == 'coupling_parallel_abs' or key == 'coupling_parallel_phase':
            return [N_paths, modes_per_path + 1]
        elif key == 'coupling_transversal_abs' or key == 'coupling_transversal_phase':
            return [N_paths-1, modes_per_path]
        elif key == 'target_mode_freqs':
            return 2
        elif key == 'trans_mode_freqs' or key == 'trans_mode_int_losses':
            return [N_paths, modes_per_path]
        else:
            raise Exception('invalid parameter keyword')

    np.random.seed(seed)

    params = {}
    params_not_trainable = {}
    for key in model_file['initial_parameters'].keys():
        params[key] = jnp.asarray(
            np.random.uniform(
                model_file['initial_parameters'][key][0],
                model_file['initial_parameters'][key][1],
                size=calc_params_shape(key)
            )
        )

    for key in model_file['not_trainable'].keys():
        params_not_trainable[key] = jnp.full(calc_params_shape(key), model_file['not_trainable'][key])

    selection_mask = np.zeros([2 * N_modes, 2 * N_modes], dtype='bool')
    selection_mask[0, 0] = True
    selection_mask[0, N_modes - 1] = True
    selection_mask[N_modes - 1, 0] = True
    selection_mask[N_modes - 1, N_modes - 1] = True

    target = mss.target_class(
        selection_mask=selection_mask,
        target=jnp.array([
            [0., 1.],
            [1., 0.], ], dtype='complex128'
        )
    )

    training_params = model_file['training']

    c = training_params['smoothening_coefficient']
    if training_params['approx_heaviside_function'] == 'sigmoid':
        heaviside_approx = lambda x: jf.sigmoid(x, c=c)
    elif training_params['approx_heaviside_function'] == 'relu':
        heaviside_approx = lambda x: jf.relu(x, c=c)
    else:
        raise Exception('invalid function name')

    os.makedirs('results', exist_ok=True)
    filename_results = 'results/N_paths=%i_modes=%i_seed=%i.npz' % (N_paths, modes_per_path, seed)

    return omegas, params, params_not_trainable, target, training_params, heaviside_approx, filename_results