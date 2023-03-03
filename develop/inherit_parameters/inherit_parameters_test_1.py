from dynpssimpy.utility_functions import structured_array_from_list
import numpy as np


class SomeModel:
    def __init__(self, par, map=None):
        self.par = par


par = structured_array_from_list(['name',   'Q',    'R'], [
    ['Unit1',     2,      3],
    ['Unit2',   2.2,    3.3]
])

par['name']

map={'name': 'name', 'R': 'r'}

mdl = SomeModel(par, map)

map_source = list(map.keys())
# par[map_keys]['name'][1] = 'Unit0'
map_dest = list(map.values())

# [dtype[1] for dtype in par.dtype]
par_subset = par[map_source]
# dtypes_subset = [par_subset.dtype[i] for i in range(len(par_subset.dtype))]
# # par_subset['name'][1] = 'HH'
# dtypes_subset_ = [(n, d) for n, d in zip(map_dest, dtypes_subset)]


# par
# dtypes = [dtype for dtype in par.dtype.names]
# dtypes = [('name', '<U5'), ('r', 'float64')]
# par_subset.view(dtype=dtypes)
new_dtypes = par_subset.dtype
new_dtypes

new_dtypes = []
new_names = []
formats = []
offsets = []
for nam, (fmt, offs) in par[map_source].dtype.fields.items():
    new_names.append(map[nam])
    formats.append(fmt)
    offsets.append(offs)

# [[map[nam], fmt, offs] for nam, (fmt, offs) in par_subset.dtype.fields.items()]
new_dtypes = np.dtype({'names': new_names, 'formats': formats, 'offsets': offsets, 'itemsize': par_subset.dtype.itemsize})
    

# new_dtypes = np.dtype({'names': ['name', 'r'], 'formats': ['<U5', '<f8'], 'offsets': [0, 28], 'itemsize': 36})
new_par = par_subset.view(dtype=new_dtypes)
new_par['r'][0] = 1
print(par['R'], new_par['r'])