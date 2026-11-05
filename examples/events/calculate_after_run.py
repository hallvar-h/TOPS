"""Outputs calculated after the simulation with ResultKeeper.calculate, with the events replayed.

Only the generator speeds are recorded during the run. The line flows and the electrical power of the generators are
calculated afterwards. Since ResultKeeper keeps a copy of the events, each step is evaluated with the line connected
or disconnected as it was at that moment, so the flow in L7-8-1 is zero while it is out of service.

    uv run python examples/events/calculate_after_run.py            # shows the plots
    uv run python examples/events/calculate_after_run.py --save calculate.png
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

import tops.dynamic as dps
from tops.simulator import Simulator, ResultKeeper


def main(t_trip=1, t_reconnect=3, t_end=6, save=None):
    import tops.ps_models.k2a as model_data

    ps = dps.PowerSystemModel(model=model_data.load())
    ps.init_dyn_sim()

    events = [
        ['t', 'category', 'model', 'unit', 'action'],
        [t_trip, 'lines', 'Line', 'L7-8-1', 'disconnect'],
        [t_reconnect, 'lines', 'Line', 'L7-8-1', 'connect'],
    ]
    sim = Simulator(ps, dt=5e-3, t_end=t_end, events=events)
    res_keeper = ResultKeeper(sim, gen_speed=ps.gen['GEN'].speed)
    sim.interface_functions['ResultKeeper'] = res_keeper.update
    sim.main_loop()

    # Calculated after the run, from the stored states and algebraic variables
    p_line = res_keeper.calculate(ps.lines['Line'].p_from)  # pu, system base
    p_gen = res_keeper.calculate(ps.gen['GEN'].p_e)  # pu, machine base

    t = np.array(res_keeper.t)
    line_names = list(ps.lines['Line'].par['name'])

    fig, axes = plt.subplots(3, sharex=True, figsize=(7, 7))
    axes[0].plot(t, res_keeper.store['gen_speed'])
    axes[0].set_ylabel('Speed [pu]')
    for name in ['L7-8-1', 'L7-8-2']:
        axes[1].plot(t, p_line[:, line_names.index(name)], label=name)
    axes[1].set_ylabel('P from bus 7 [pu]')
    axes[1].legend()
    axes[2].plot(t, p_gen, label=list(ps.gen['GEN'].par['name']))
    axes[2].set_ylabel('Generator P [pu]')
    axes[2].set_xlabel('Time [s]')
    axes[2].legend()
    for ax in axes:
        for t_event in (t_trip, t_reconnect):
            ax.axvline(t_event, color='gray', linestyle=':')
    fig.tight_layout()

    if save:
        fig.savefig(save)
    else:
        plt.show()
    return ps, res_keeper, p_line


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--save', help='write the figure to this file instead of showing it')
    main(save=parser.parse_args().save)
