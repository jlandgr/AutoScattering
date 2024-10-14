import multimode_systems.scattering as mss
import jax.numpy as jnp

def remove_keys(full_list, to_remove):
    full_list = full_list.copy()
    for element in to_remove:
        full_list.remove(element)
    return full_list

def calc_scattering_matrix(paras, omegas, free_paras_keys, fixed_paras, operators):
    N_modes = len(operators)
    all_paras = {key: paras[key_idx] for key_idx, key in enumerate(free_paras_keys)}
    all_paras.update(fixed_paras)
    system = mss.Multimode_system(N_modes)

    for idx in range(N_modes):
        system.add_adag_a_coupling(all_paras['Delta%i'%idx], idx, idx)
    
    for idx2 in range(N_modes):
        for idx1 in range(idx2):
            coupling = all_paras['greal%i%i'%(idx1, idx2)] + 1.j*all_paras['gimag%i%i'%(idx1, idx2)]
            if type(operators[idx1]) is type(operators[idx2]):
                system.add_adag_a_coupling(coupling, idx1, idx2)
            else:
                system.add_adag_adag_coupling(coupling, idx1, idx2)

    system.ext_losses = jnp.exp(jnp.array([all_paras['kappalog%i'%idx] for idx in range(N_modes)]))

    return mss.calc_scattering_matrix(omegas, *system.return_system_parameters())