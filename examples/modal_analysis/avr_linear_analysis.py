from dynpssimpy.dyn_models.avr import SEXS
from dynpssimpy.dyn_models.utils import get_submodules
from dynpssimpy.modal_analysis import PowerSystemModelLinearization
import numpy as np


if __name__ == '__main__':

    avr = SEXS(
        name=   ['AVR1'],        
        gen=    ['Gen'],       
        K=      [100],
        T_a=    [9.0],
        T_b=    [10.0],
        T_e=    [0.5],
        E_min=  [-3],
        E_max=  [3],
    )

    state_desc = np.empty((0, 2))
    n_states = 0
    mdls = get_submodules(avr)
    for mdl in mdls:
        print(mdl)
        mdl.idx = slice(mdl.idx.start + n_states, mdl.idx.stop + n_states)
        for field in mdl.state_idx_global.dtype.names:
            mdl.state_idx_global[field] += mdl.idx.start
        n_states += mdl.n_states * mdl.n_units
        state_desc = np.vstack([state_desc, mdl.state_desc])

    mdls_with_states = []
    for mdl in mdls:
        if hasattr(mdl, 'state_derivatives'):
            mdls_with_states.append(mdl)
    
    def ode_fun(t, x):

        dx = np.zeros(n_states)
        for mdl in mdls_with_states:
            mdl.state_derivatives(dx, x, None)

        return dx
    
    x = np.zeros(n_states)
    avr.init_from_connections(x, None, dict(output=[1]))
    avr.output(x, None)
    ode_fun(0, x)

    ps = type('', (), {})
    ps.x0 = x.copy()
    ps.n_states = n_states
    ps.ode_fun = ode_fun
    ps.solve_algebraic = lambda t, x: None
    ps.v0 = np.array([])

    lin = PowerSystemModelLinearization(ps)
    lin.linearize()
    lin.eigenvalue_decomposition()
    lin.eigs

    def b_func(ps, disturbance):
        avr.set_input('v_t', disturbance, 0)
    
    lin.linearize_inputs_v3([b_func])
    
    def c_func(t, x, v):
        return avr.output(x, v)[0]

    lin.linearize_outputs_v4([c_func])

    from scipy import signal
    import matplotlib.pyplot as plt
    sys = (lin.a, lin.b, lin.c, lin.d)  # signal.TransferFunction([1], [1, 1])
    w, mag, phase = signal.bode(sys)
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].semilogx(w, mag)    # Bode magnitude plot
    plt.grid()
    ax[1].semilogx(w, phase)  # Bode phase plot
    plt.grid()
    plt.show()