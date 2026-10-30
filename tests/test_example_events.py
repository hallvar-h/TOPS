"""The example that shows a line trip, a transformer trip and a load change must keep running."""
import importlib.util
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')


def load_example():
    path = Path(__file__).parent.parent / 'examples' / 'events' / 'trip_trafo_load.py'
    spec = importlib.util.spec_from_file_location('trip_trafo_load', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_example_applies_a_line_trip_a_transformer_trip_and_a_load_change(tmp_path):
    ps, res_keeper = load_example().main(times=(0.1, 0.2, 0.3), t_end=0.5, save=tmp_path / 'events.png')

    line, trafo, load = ps.lines['Line'], ps.trafos['Trafo'], ps.loads['Load']
    assert not line.connected[list(line.par['name']).index('L7-8-1')]
    assert not trafo.connected[list(trafo.par['name']).index('T1b')]
    assert load.par['P'][list(load.par['name']).index('L1')] == 900.0
    assert (tmp_path / 'events.png').stat().st_size > 0
    assert np.isfinite(np.array(res_keeper.store['gen_speed'])).all()
