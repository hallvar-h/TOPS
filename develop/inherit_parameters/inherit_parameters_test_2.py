from dynpssimpy.utility_functions import structured_array_from_list
import numpy as np


class SomeModel:
    def __init__(self, par, map=None):
        self.par = par


par = structured_array_from_list(['name',   'Q',    'R'], [
    ['Unit1',     2,      3],
    ['Unit2',   2.2,    3.3]
])


map={'name': 'name', 'R': 'r'}
mdl = SomeModel(par, map)

par_subset = par[list(map.keys())]

new_dtypes = []
new_names = []
formats = []
offsets = []
for nam, (fmt, offs) in par_subset.dtype.fields.items():
    new_names.append(map[nam])
    formats.append(fmt)
    offsets.append(offs)

new_dtypes = np.dtype({'names': new_names, 'formats': formats, 'offsets': offsets, 'itemsize': par_subset.dtype.itemsize})
new_par = par_subset.view(dtype=new_dtypes)

new_par['r'][0] = 1
print(par['R'], new_par['r'])