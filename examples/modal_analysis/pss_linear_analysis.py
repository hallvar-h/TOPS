from tops.dyn_models.avr import SEXS
from tops.dyn_models.gov import HYGOV
from tops.dyn_models.pss import STAB1
from tops.dyn_models.utils import get_submodules
from tops.modal_analysis import PowerSystemModelLinearization
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # analysed_model = SEXS(
    #     name=   ['AVR1'],        
    #     gen=    ['Gen'],       
    #     K=      [100],
    #     T_a=    [2.0],
    #     T_b=    [10.0],
    #     T_e=    [0.5],
    #     E_min=  [-3],
    #     E_max=  [3],
    # )
    # input_signal = 'v_t'
    # output_signal = 'output'

    # analysed_model = HYGOV(
    #     name=           ['HYGOV1'],
    #     gen=            [    'G1'],
    #     R=              [    0.04],
    #     r=              [     0.1],
    #     T_f=            [     0.1],
    #     T_r=            [      10],
    #     T_g=            [     0.5],
    #     A_t=            [       1],
    #     T_w=            [       1],
    #     q_nl=           [    0.01],
    #     D_turb=         [    0.01],
    #     G_min=          [       0],
    #     V_elm=          [    0.15],
    #     G_max=          [       1],
    #     P_N=            [       0],
    # )
    # input_signal = 'input'
    # output_signal = 'output'

    analysed_model = STAB1(
        name=           ['PSS1'],
        gen=            [    'G1'],
        K=              [   50     ],
        T=              [   10.0  ],
        T_1=            [   0.5  ],
        T_2=            [   0.5  ],
        T_3=            [   0.05  ],
        T_4=            [   0.05  ],
        H_lim=          [   0.03  ],
    )
    input_signal = 'input'
    output_signal = 'output'

    state_desc = np.empty((0, 2))
    n_states = 0
    mdls = get_submodules(analysed_model)
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
    if hasattr(analysed_model, 'init_from_connections'):
        analysed_model.init_from_connections(x, None, dict(output=[1]))
    analysed_model.output(x, None)
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
    
    plt.scatter(lin.eigs.real, lin.eigs.imag/(2*np.pi))
    plt.grid()
    

    def b_func(ps, disturbance):
        analysed_model.set_input(input_signal, disturbance, 0)
    
    lin.linearize_inputs_v3([b_func])
    
    def c_func(t, x, v):
        return getattr(analysed_model, output_signal)(x, v)[0]

    lin.linearize_outputs_v4([c_func])

    from scipy import signal
    sys = (lin.a, lin.b, lin.c, lin.d)  # signal.TransferFunction([1], [1, 1])
    w, mag, phase = signal.bode(sys)
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].semilogx(w, mag)    # Bode magnitude plot
    ax[0].grid()
    ax[1].semilogx(w, phase)  # Bode phase plot
    ax[1].grid()
    plt.show()