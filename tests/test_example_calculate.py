"""The example that calculates outputs after the simulation must keep running."""
import importlib.util
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')


def load_example():
    path = Path(__file__).parent.parent / 'examples' / 'events' / 'calculate_after_run.py'
    spec = importlib.util.spec_from_file_location('calculate_after_run', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_example_calculates_line_flows_with_the_events_replayed(tmp_path):
    ps, res_keeper, p_line = load_example().main(t_trip=0.1, t_reconnect=0.3, t_end=0.5, save=tmp_path / 'calc.png')

    t = np.array(res_keeper.t)
    p_tripped = p_line[:, list(ps.lines['Line'].par['name']).index('L7-8-1')]
    assert np.all(p_tripped[(t > 0.1 + 5e-3) & (t < 0.3)] == 0)
    assert np.all(p_tripped[(t < 0.1) | (t > 0.3 + 5e-3)] != 0)
    assert (tmp_path / 'calc.png').stat().st_size > 0
