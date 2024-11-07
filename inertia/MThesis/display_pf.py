
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
import tops.solvers as dps_sol
import importlib
import cmath
importlib.reload(dps)
import numpy as np

def format_complex(complex_num, polar = True):
    if polar == True:
        mag = round(abs(complex_num),5)
        phase = round(np.angle(complex_num, True))
        return mag, phase
    else:
        real = round(complex_num.real, 5)
        imag = round(complex_num.imag, 5)
        return real, imag

def pf_calc(ps, v):
    ybus = ps.build_y_bus()
    S_calc = v*(ybus@v).conjugate()


def display_lf(sol):

    print('-'*50, 'POWER FLOW' , '-'*50)
    print('Bus'.center(10))
    print(' nr.')


    for i in range(len(sol.v)):
        txt = " {nr} {V_mag}, {V_angle}"
        vmag, ph = format_complex(sol.v[i])
        print(txt.format(nr = i+1, V_mag = vmag, V_angle = ph))
    
    return None

if __name__ == '__main__':



    # Load model
    import tops.ps_models.k2a as model_data
    importlib.reload(model_data)
    model = model_data.load()

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))


    y_bus = ps.build_y_bus_lf()

    t_end = 10
    x_0 = ps.x_0.copy()

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

    display_lf(sol)








        


        
