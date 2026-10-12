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
