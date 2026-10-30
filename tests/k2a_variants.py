"""K2A with branches added, removed or flagged, for the tests of the branch events and of the 'connected' column."""
import numpy as np

import tops.dynamic as dps

# In K2A every transformer is the only link of its generator, so one is added in parallel to be taken out.
TRAFO = 'T1b'
KINDS = ['Trafo', 'DynTrafo']


def k2a_model():
    import tops.ps_models.k2a as model_data
    return model_data.load()


def with_connected(model, category, connected):
    """Add a 'connected' column to a table; `connected` maps unit names to flags, the others are connected."""
    header, *rows = model[category]
    model[category] = [[*header, 'connected'], *[[*row, connected.get(row[0], True)] for row in rows]]
    return model


def trafo_model(kind, connected=None, ratios=(1, 1)):
    """K2A with its transformers, and T1b in parallel to T1, as `kind` ('Trafo' or 'DynTrafo'), all with `ratios`."""
    model = k2a_model()
    header, *rows = model.pop('transformers')
    rows.append([TRAFO, 'B1', 'B5', 900, 20, 230, 0, 0.15])
    if kind == 'DynTrafo' or tuple(ratios) != (1, 1):
        header, rows = [*header, 'ratio_from', 'ratio_to'], [[*row, *ratios] for row in rows]
    table = [[*header, 'connected'], *[[*row, connected.get(row[0], True)] for row in rows]] if connected is not None \
        else [header, *rows]
    model['trafos'] = {kind: table}
    return model


def removed_trafo_model(kind):
    model = trafo_model(kind)
    header, *rows = model['trafos'][kind]
    model['trafos'] = {kind: [header, *[row for row in rows if row[0] != TRAFO]]}
    return model


def initialized(model):
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    return ps


def y_bus_red(ps):
    return ps.y_bus_red.toarray()


def trafo_index(ps, kind):
    return np.where(ps.trafos[kind].par['name'] == TRAFO)[0][0]


def bay_model(switches=('LS',), via='Line', shunt_mvar=None):
    """K2A with a bus B12 that nothing but `switches` (and a shunt of `shunt_mvar`) connects to B5.

    A switch is a branch without shunt admittance (B = 0), so a bay node whose switches are all open is floating.
    With via='DynTrafo' the one switch is a DynTrafo, which stamps its admittance at every step, not into the
    constant matrix.
    """
    if via == 'DynTrafo':
        model = trafo_model('DynTrafo')
        model['trafos']['DynTrafo'].append(['TS', 'B5', 'B12', 900, 230, 230, 0, 1e-4, 1, 1])
    else:
        model = k2a_model()
        # Each switch has its own impedance, so that taking them out one by one leaves rounding, as real stamps do
        model['lines'] += [[name, 'B5', 'B12', 1, 100, 230, 'p.u.', 1e-5*(1 + 0.37*i), 1e-4*(1 + 0.37*i), 0]
                           for i, name in enumerate(switches)]
    model['buses'].append(['B12', 230])
    if shunt_mvar is not None:
        model['shunts'].append(['C3', 'B12', 230, shunt_mvar, 'Z'])
    return model
