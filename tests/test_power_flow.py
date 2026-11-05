"""Choice of slack bus and slack unit in the power flow.

load_flow_pv() returns P with load convention (a generator producing 100 MW returns -100), so the generator with the
largest output has the most negative value.
"""
import pytest

import tops.dynamic as dps
import tops.ps_models.k2a as model_data


def set_gen(model, name, **values):
    header, *rows = model['generators']['GEN']
    row = next(row for row in rows if row[header.index('name')] == name)
    for key, value in values.items():
        row[header.index(key)] = value


def test_default_slack_bus_is_bus_of_largest_generator():
    # In K2A, G3 has the largest P (719 MW, the others 700 MW)
    model = model_data.load()
    model.pop('slack_bus', None)
    ps = dps.PowerSystemModel(model=model)
    ps.power_flow()
    assert ps.slack_bus == 'B3'


@pytest.mark.parametrize('p_g4, slack_name', [(500, 'G3'), (900, 'G4')])
def test_slack_unit_is_largest_generator_at_slack_bus(p_g4, slack_name):
    # Move G4 to the slack bus B3, next to G3 (719 MW)
    model = model_data.load()
    model['slack_bus'] = 'B3'
    set_gen(model, 'G4', bus='B3', P=p_g4)
    ps = dps.PowerSystemModel(model=model)
    ps.power_flow()
    mdl, unit_idx = ps.slack_unit
    assert mdl.par['name'][unit_idx] == slack_name
