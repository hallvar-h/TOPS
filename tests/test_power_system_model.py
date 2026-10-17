import pytest
import tops.dynamic as dps


def test_missing_json_file_raises():
    with pytest.raises(FileNotFoundError):
        dps.PowerSystemModel(model='does_not_exist.json')


def test_invalid_model_type_raises():
    with pytest.raises(TypeError):
        dps.PowerSystemModel(model=42)


def test_unknown_model_warns():
    import tops.ps_models.k2a as model_data
    model = model_data.load()
    model['loads'] = {'NoSuchLoadModel': model['loads']}
    with pytest.warns(UserWarning, match='NoSuchLoadModel'):
        dps.PowerSystemModel(model=model)


def test_repeated_init_keeps_state_indices():
    import tops.ps_models.k2a as model_data
    ps = dps.PowerSystemModel(model=model_data.load())
    ps.init_dyn_sim()
    idx_first = ps.gen['GEN'].state_idx_global['speed'].copy()
    ps.init_dyn_sim()
    idx_second = ps.gen['GEN'].state_idx_global['speed']

    assert (idx_first == idx_second).all()
    assert (ps.state_desc[idx_second, 1] == 'speed').all()


def test_dataframe_input_matches_list_input():
    import numpy as np
    import pandas as pd
    import tops.ps_models.k2a as model_data

    def to_dataframes(data):
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            return pd.DataFrame(data[1:], columns=data[0])
        if isinstance(data, dict):
            return {key: to_dataframes(val) for key, val in data.items()}
        return data

    ps_list = dps.PowerSystemModel(model=model_data.load())
    ps_list.init_dyn_sim()
    ps_df = dps.PowerSystemModel(model=to_dataframes(model_data.load()))
    ps_df.init_dyn_sim()
    assert np.allclose(ps_df.x0, ps_list.x0)
