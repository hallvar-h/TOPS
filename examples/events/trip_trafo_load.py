"""A line trip, a transformer trip and a load change, applied one by one with ps.apply_event.

    uv run python examples/events/trip_trafo_load.py            # shows the plots
    uv run python examples/events/trip_trafo_load.py --save events.png
"""
import argparse

import matplotlib.pyplot as plt

import tops.dynamic as dps
from tops.simulator import Simulator, ResultKeeper


def main(times=(1, 3, 5), t_end=8, save=None):
    import tops.ps_models.k2a as model_data

    model = model_data.load()
    # Every transformer of K2A is the only link of its generator, so one is added in parallel, to be tripped
    model['transformers'].append(['T1b', 'B1', 'B5', 900, 20, 230, 0, 0.15])

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    # What happens, and when: (t, category, model, unit, action, arguments of the action)
    occurrences = [
        (times[0], 'lines', 'Line', 'L7-8-1', 'disconnect', {}),
        (times[1], 'trafos', 'Trafo', 'T1b', 'disconnect', {}),
        (times[2], 'loads', 'Load', 'L1', 'set_power', {'P': 900.0}),  # from 967 MW
    ]

    sim = Simulator(ps, dt=5e-3, t_end=t_end)
    res_keeper = ResultKeeper(sim, gen_speed=ps.gen['GEN'].speed, load_p=ps.loads['Load'].P)
    sim.interface_functions['ResultKeeper'] = res_keeper.update

    while sim.sol.t < t_end:
        while occurrences and occurrences[0][0] <= sim.sol.t:
            _, category, mdl_key, unit, action, kwargs = occurrences.pop(0)
            ps.apply_event(category, mdl_key, unit, action, **kwargs)
            print(f't = {sim.sol.t:.2f} s: {category}/{unit} {action} {kwargs or ""}')
        sim.make_simulation_step()

    fig, axes = plt.subplots(2, sharex=True)
    axes[0].plot(res_keeper.t, res_keeper.store['gen_speed'])
    axes[0].set_ylabel('Generator speed [p.u.]')
    axes[1].plot(res_keeper.t, [p[0] for p in res_keeper.store['load_p']])
    axes[1].set_ylabel('Load L1 [MW]')
    axes[1].set_xlabel('Time [s]')
    for ax in axes:
        for t in times:
            ax.axvline(t, color='gray', linestyle=':')
    fig.tight_layout()

    if save:
        fig.savefig(save)
    else:
        plt.show()
    return ps, res_keeper


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--save', help='write the figure to this file instead of showing it')
    main(save=parser.parse_args().save)
