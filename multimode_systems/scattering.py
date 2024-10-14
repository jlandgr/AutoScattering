import jax
import jax.numpy as jnp
import numpy as np

from jax.experimental import host_callback

class Multimode_system():
    def __init__(self, N_modes, target_modes='all', dtype='complex128'):
        self.N_modes = N_modes
        if target_modes == 'all':
            self.target_modes = jnp.arange(N_modes)
        else:
            self.target_modes = jnp.asarray(target_modes)

        self.adag_a_coupling_matrix = jnp.zeros([N_modes, N_modes], dtype=dtype)
        self.adag_adag_coupling_matrix = jnp.zeros([N_modes, N_modes], dtype=dtype)

        self.ext_losses = jnp.zeros(N_modes, dtype=dtype)
        self.ext_losses = self.ext_losses.at[self.target_modes].set(1.)

        self.int_losses = jnp.zeros(N_modes, dtype=dtype)

        self.sigma_z = jnp.diag(jnp.hstack((jnp.ones(self.N_modes), -jnp.ones(self.N_modes))))

    def add_adag_a_coupling(self, coupling_rate, mode1, mode2):
        # if self.adag_a_coupling_matrix[mode1, mode2] != 0.:
        #     raise Exception('coupling rate was already set')

        # if mode1 == mode2 and jnp.imag(coupling_rate) != 0.:
        #     raise Exception('diagional entries have to be real')

        self.adag_a_coupling_matrix = self.adag_a_coupling_matrix.at[mode1, mode2].set(coupling_rate)
        self.adag_a_coupling_matrix = self.adag_a_coupling_matrix.at[mode2, mode1].set(jnp.conj(coupling_rate))

    def add_adag_adag_coupling(self, coupling_rate, mode1, mode2):
        self.adag_adag_coupling_matrix = self.adag_adag_coupling_matrix.at[mode1, mode2].set(coupling_rate)
        self.adag_adag_coupling_matrix = self.adag_adag_coupling_matrix.at[mode2, mode1].set(coupling_rate)

    def add_intrinsic_loss(self, loss_rate, mode):
        # if jnp.imag(loss_rate) != 0 or loss_rate < 0.:
        #     raise Exception('invalid value for a loss rate')
        #
        # if self.int_losses[mode] != 0.:
        #     raise Exception('loss rate was already set')

        self.int_losses = self.int_losses.at[mode].set(loss_rate)

    def return_system_parameters(self):
        return self.adag_a_coupling_matrix, self.adag_adag_coupling_matrix, self.int_losses, self.ext_losses  
    
    def return_Hamiltonian(self):
        # jnp.vstack((jnp.hstack((upper_left, upper_right)), jnp.hstack((lower_left, lower_right))))
        g = self.adag_a_coupling_matrix
        nu = self.adag_adag_coupling_matrix
        hamiltonian = jnp.vstack((jnp.hstack((g, nu)), jnp.hstack((jnp.conjugate(nu), jnp.conjugate(g)))))
        return hamiltonian
    
    def return_scattering_matrix(self, omegas=jnp.array([0])):
        matrix_ext_losses = jnp.diag(jnp.hstack((self.ext_losses, self.ext_losses)))
        matrix_int_losses = jnp.diag(jnp.hstack((self.int_losses, self.int_losses)))
        identity = jnp.identity(2 * self.N_modes)

        
        hamiltonian = self.return_Hamiltonian()
        scattering_matrix = identity + jnp.linalg.inv(-1.j * self.sigma_z@hamiltonian - (matrix_ext_losses + matrix_int_losses)/2.)
        return scattering_matrix

        # expanded_identity = jnp.expand_dims(identity, 0)
        # ext_losses_square_root = jnp.sqrt(matrix_ext_losses)
        
        # matrix_omegas = jnp.expand_dims(omegas, (1,2))*expanded_identity

        # inverted_matrix = jnp.linalg.inv(jnp.expand_dims(dynamical_matrix, 0) + 1.j*matrix_omegas)
        # scattering_matrix = expanded_identity + ext_losses_square_root@inverted_matrix@ext_losses_square_root

def prepare_parameters_1D_chain(mode_freqs, on_site_adag_adag, coupling_adag_a, coupling_adag_adag, int_losses, ext_losses):
    N_modes = len(mode_freqs)
    dtype = 'complex128'

    adag_a_coupling_matrix = jnp.zeros([N_modes, N_modes], dtype=dtype)
    adag_adag_coupling_matrix = jnp.zeros([N_modes, N_modes], dtype=dtype)

    adag_a_coupling_matrix += jnp.diag(mode_freqs)
    adag_adag_coupling_matrix += 2. * jnp.diag(on_site_adag_adag)

    if N_modes == 1:
        pass
    else:
        fill_value = jnp.diag(coupling_adag_a[:-1], k=1)
        if N_modes != 2:
            fill_value += jnp.eye(N_modes, N_modes, k=-(N_modes-1)) * coupling_adag_a[-1]
        adag_a_coupling_matrix += fill_value
        adag_a_coupling_matrix += jnp.conj(fill_value).T

        fill_value = jnp.diag(coupling_adag_adag[:-1], k=1)
        if N_modes != 2:
            fill_value += jnp.eye(N_modes, N_modes, k=-(N_modes-1)) * coupling_adag_adag[-1]
        adag_adag_coupling_matrix += fill_value
        adag_adag_coupling_matrix += fill_value.T

    return adag_a_coupling_matrix, adag_adag_coupling_matrix, int_losses, ext_losses

def create_dynamical_matrix_1D_chain(*args, **kwargs):
    return create_dynamical_matrix(*prepare_parameters_1D_chain(*args, **kwargs))

# def _eig_host(matrix: jnp.ndarray):
#     """Wraps jnp.linalg.eig so that it can be jit-ed on a machine with GPUs, but not vmapped!"""
#     return host_callback.call(
#         # We force this computation to be performed on the cpu by jit-ing and
#         # explicitly specifying the device.
#         jax.jit(jnp.linalg.eigvals, device=jax.devices("cpu")[0]),
#         matrix.astype(complex),
#         result_shape=jax.ShapeDtypeStruct(matrix.shape[:-1], complex),
#     )

def create_dynamical_matrix(adag_a_coupling_matrix, adag_adag_coupling_matrix, int_losses, ext_losses):
    N_modes = len(adag_a_coupling_matrix)

    # TODO check adag_a_coupling_matrix for symmetry!

    losses = int_losses + ext_losses
    matrix_losses = jnp.diag(jnp.hstack((losses, losses)))
 
    upper_left = adag_a_coupling_matrix - 1.j / 2. * jnp.diag(losses)
    upper_right = (adag_adag_coupling_matrix + adag_adag_coupling_matrix.T) / 2.
    lower_left = -jnp.conj(adag_adag_coupling_matrix + adag_adag_coupling_matrix.T) / 2.
    lower_right = -jnp.conj(adag_a_coupling_matrix) - 1.j / 2. * jnp.diag(losses)

    Hnh = jnp.vstack((jnp.hstack((upper_left, upper_right)), jnp.hstack((lower_left, lower_right))))

    dynamical_matrix = -1.j * Hnh
    eigenvalues = jnp.linalg.eigvals(dynamical_matrix)
    # eigenvalues = _eig_host(dynamical_matrix)
    stable = jnp.all(jnp.real(eigenvalues) < 0)
    # eigenvalues = None
    # stable = None

    return dynamical_matrix, stable, eigenvalues

def calc_scattering_matrix_1D_chain(omegas, *args, **kwargs):
    return calc_scattering_matrix(omegas, *prepare_parameters_1D_chain(*args, **kwargs))

def calc_scattering_matrix(omegas, adag_a_coupling_matrix, adag_adag_coupling_matrix, int_losses, ext_losses):
    dynamical_matrix, stable, eigenvalues = create_dynamical_matrix(adag_a_coupling_matrix, adag_adag_coupling_matrix, int_losses, ext_losses)

    N_modes = len(adag_a_coupling_matrix)
    matrix_ext_losses = jnp.diag(jnp.hstack((ext_losses, ext_losses)))
    identity = jnp.identity(2 * N_modes)

    expanded_identity = jnp.expand_dims(identity, 0)
    ext_losses_square_root = jnp.sqrt(matrix_ext_losses)
    matrix_omegas = jnp.expand_dims(omegas, (1,2))*expanded_identity

    inverted_matrix = jnp.linalg.inv(jnp.expand_dims(dynamical_matrix, 0) + 1.j*matrix_omegas)
    scattering_matrix = expanded_identity + ext_losses_square_root@inverted_matrix@ext_losses_square_root

    info_dict = {
        'dynamical_matrix': dynamical_matrix,
        'stability': stable,
        'eigenvalues': eigenvalues
    }

    return scattering_matrix, info_dict

def interpolate(x0, x1, y0, y1, value):
    return (x1-x0)/(y1-y0)*(value-y0)+x0

def calc_bandwidth(omegas, signal, threshold, ratio=None, max_val=None, max_idx=None):
    if max_val is None:
        max_val = jnp.max(signal)
    if max_idx is None:
        max_idx = jnp.argmax(signal)
    if threshold is None:
        threshold = ratio*max_val

    if signal[max_idx] < threshold:
        return 0
    
    below_threshold_right = jnp.where(signal[max_idx:] <= threshold)[0]
    if len(below_threshold_right) == 0:
        omega_right = omegas[-1]
    else:
        idx_right = max_idx + below_threshold_right[0]
        omega_right = interpolate(omegas[idx_right-1], omegas[idx_right], signal[idx_right-1], signal[idx_right], threshold)
    
    below_threshold_left = jnp.where(signal[:max_idx+1] <= threshold)[0]
    if len(below_threshold_left) == 0:
        omega_left = omegas[0]
    else:
        idx_left = below_threshold_left[-1]
        omega_left = interpolate(omegas[idx_left], omegas[idx_left+1], signal[idx_left], signal[idx_left+1], threshold)
    
    bandwidth = omega_right - omega_left

    return bandwidth

def create_artificial_hermitian_matrix(dynamical_matrix, omegas):
    N_modes = len(dynamical_matrix)//2
    Hnh = 1.j*dynamical_matrix
    omegas_matrix = omegas[:,None,None]*np.identity(2*N_modes)[None]
    artificial_hermitian_H = jnp.zeros([len(omegas), 4*N_modes, 4*N_modes], dtype='complex')
    artificial_hermitian_H = artificial_hermitian_H.at[:,0:2*N_modes:,2*N_modes:].set(omegas_matrix-Hnh[None])
    artificial_hermitian_H = artificial_hermitian_H.at[:,2*N_modes:,0:2*N_modes:].set(omegas_matrix-jnp.conjugate(jnp.transpose(Hnh))[None])

    #the eigenvalues are always real, but the dtype of the eigvals is complex by default
    eigvals = jnp.real(jnp.linalg.eigvals(artificial_hermitian_H))

    return eigvals


class target_class():
    def __init__(self, selection_mask, target):
        self.selection_mask = selection_mask
        self.target = target
        self.N_modes = len(selection_mask) // 2
        self.N_modes_target = np.max(np.sum(selection_mask, 0))

    def apply_selection(self, scattering_matrix):
        return scattering_matrix[:, self.selection_mask].reshape(
            [-1, self.N_modes_target, self.N_modes_target]
        )

    def calc_deviation(self, scattering_matrix):
        selection = self.apply_selection(scattering_matrix)

        return jnp.real(jnp.sum((jnp.abs(selection) - self.target) ** 2, axis=(1, 2)))