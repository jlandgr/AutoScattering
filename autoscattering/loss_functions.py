@jax.jit
def max_at_single_omega(params):
    gc, kappa_e = params
    gs = gc

    on_site_adag_adag = jnp.full(N_modes, -gs/2.)
    coupling_adag_adag = jnp.hstack((jnp.full(N_modes-1, -gc), [0.]))
    ext_losses = jnp.full(N_modes, kappa_e)

    scattering_matrix, info_dict = mss.calc_scattering_matrix(
        mode_freqs, on_site_adag_adag,coupling_adag_a, coupling_adag_adag, int_losses, ext_losses, omegas
    )

    scattering_abs = jnp.abs(scattering_matrix[:, 0, N_modes-1])
    argmax = jnp.argmax(scattering_abs)

    return -scattering_abs[argmax], info_dict

@jax.jit
def max_band_gap(params):
    gc, kappa_e = params
    gs = gc

    on_site_adag_adag = jnp.full(N_modes, -gs/2.)
    coupling_adag_adag = jnp.hstack((jnp.full(N_modes-1, -gc), [0.]))
    ext_losses = jnp.full(N_modes, kappa_e)

    scattering_matrix, info_dict = mss.calc_scattering_matrix(
        mode_freqs, on_site_adag_adag,coupling_adag_a, coupling_adag_adag, int_losses, ext_losses, omegas
    )

    eigvals = create_artificial_hermitian_matrix(info_dict['dynamical_matrix'])

    arg_zero = jnp.argmin(jnp.abs(omegas))
    eigvals_zero_sorted = jnp.sort(jnp.abs(eigvals[arg_zero]))
    band_gap = eigvals_zero_sorted[2] + eigvals_zero_sorted[3]

    return -band_gap, info_dict

@jax.jit
def linear_regression(x, y):
    x_mean = jnp.mean(x)
    y_mean = jnp.mean(y)

    a = jnp.sum((x-x_mean)*(y-y_mean))/jnp.sum((x-x_mean)**2)
    b = y_mean - a*x_mean
    return a, b

@jax.jit
def calc_zeta_max(params):
    gc, kappa_e = params
    gs = gc

    on_site_adag_adag = jnp.full(N_modes, -gs/2.)
    coupling_adag_adag = jnp.hstack((jnp.full(N_modes-1, -gc), [0.]))
    ext_losses = jnp.full(N_modes, kappa_e)

    scattering_matrix, info_dict = mss.calc_scattering_matrix(
        mode_freqs, on_site_adag_adag, coupling_adag_a, coupling_adag_adag, int_losses, ext_losses, omegas
    )
    xs = jnp.array(jnp.arange(0, N_modes), dtype='float')
    # ys = jnp.max(jnp.abs(scattering_matrix[:,0,0:N_modes])**2, axis=0)
    arg_selected_omega = jnp.argmin(jnp.abs(omegas+0.5))
    ys = jnp.abs(scattering_matrix[arg_selected_omega,0,0:N_modes])**2
    opt_result = linear_regression(xs, jnp.log(ys))


    info_dict['opt_result'] = opt_result
    return -opt_result[0], info_dict

@jax.jit
def zeta_Delta_product(params):
    gc, kappa_e = params
    gs = gc

    on_site_adag_adag = jnp.full(N_modes, -gs/2.)
    coupling_adag_adag = jnp.hstack((jnp.full(N_modes-1, -gc), [0.]))
    ext_losses = jnp.full(N_modes, kappa_e)

    scattering_matrix, info_dict = mss.calc_scattering_matrix(
        mode_freqs, on_site_adag_adag, coupling_adag_a, coupling_adag_adag, int_losses, ext_losses, omegas
    )

    arg_zero = jnp.argmin(jnp.abs(omegas))  #take omega=0
    xs = jnp.array(jnp.arange(0, N_modes), dtype='float')
    #ys = jnp.max(jnp.abs(scattering_matrix[:,0,0:N_modes])**2, axis=0)  #take max gain
    ys = jnp.abs(scattering_matrix[arg_zero,0,0:N_modes])**2
    opt_result = linear_regression(xs, jnp.log(ys))

    eigvals = create_artificial_hermitian_matrix(info_dict['dynamical_matrix'])
    eigvals_zero_sorted = jnp.sort(jnp.abs(eigvals[arg_zero]))
    band_gap = eigvals_zero_sorted[2] + eigvals_zero_sorted[3]

    # fit_paras = jnp.array([1.])
    # opt_result = jax_opt.minimize(mse, fit_paras, args=(xs, ys), method='BFGS')
    info_dict['opt_result'] = opt_result
    info_dict['band_gap'] = band_gap
    return -opt_result[0]*band_gap, info_dict