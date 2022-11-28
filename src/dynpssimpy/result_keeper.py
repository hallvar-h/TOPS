import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy


class ResultKeeper:
    def __init__(self, spec, n_steps):
        self.n_steps = n_steps  # Number of time steps to allocate space for
        self.spec_0 = spec  # Input specification
        self.dtypes = dict.fromkeys(self.spec_0.keys())
        self.spec_dtypes = dict()
        self.head = dict()
        for key in self.spec_dtypes:
            self.spec_dtypes[key] = dict()

        for key, val in self.spec_0.items():
            if callable(val):
                store_fun = val
                tmp_res = store_fun()
                if np.iterable(tmp_res):
                    desc_ = ['{}'.format(i) for i in range(len(tmp_res))]
                    dtype = store_fun().dtype
                    self.head[key] = desc_
                    val = desc_, store_fun
                else:
                    dtype = type(tmp_res)
            else:
                # A description was supplied with the store function
                desc_, store_fun = val
                dtype = store_fun().dtype
                self.head[key] = desc_

            self.dtypes[key] = dtype
            if dtype in self.spec_dtypes.keys():
                self.spec_dtypes[dtype].update({key: val})
            else:
                self.spec_dtypes[dtype] = {key: val}

        self.res_keepers = dict.fromkeys(self.spec_dtypes.keys())
        for key, val in self.spec_dtypes.items():
            self.res_keepers[key] = ResultKeeperSpecific(self.spec_dtypes[key], n_steps=self.n_steps, dtype=key)

    def store(self):
        for key, res_keeper in self.res_keepers.items():
            res_keeper.store()

    def __getitem__(self, args):
        if isinstance(args, str):
            args = (args,)
        res_keeper_key = self.dtypes[args[0]]
        return self.res_keepers[res_keeper_key].__getitem__(args)

    def save_to_file(self, path):

        [setattr(self, attr, None) for attr in ['desc_0', 'desc_dtypes']]
        for key, res_keeper in self.res_keepers.items():
            [setattr(res_keeper, attr, None) for attr in ['desc_0', 'store_funs']]

        with open(path + '.rksav', 'wb') as file_handler:
            pickle.dump(self, file_handler)

    @staticmethod
    def load_from_file(path):
        with open(path + '.rksav', 'rb') as file_handler:
            loaded_object = pickle.load(file_handler)
        return loaded_object


class ResultKeeperSpecific:
    def __init__(self, spec, n_steps, dtype=float):
        self.desc_0 = spec
        self.cols = []
        self.store_funs = dict.fromkeys(self.desc_0.keys())
        self.idx = dict.fromkeys(self.desc_0.keys())
        self.head = dict()
        self.k_store = 0
        k_idx = 0
        for key, val in self.desc_0.items():
            if callable(val):
                store_fun = val
                dtype = type(store_fun())
                idx_inc = 1
                self.cols.append([key])
            else:
                desc_, store_fun = val
                dtype = store_fun().dtype
                idx_inc = len(desc_)
                self.head[key] = desc_
                for desc__ in desc_:
                    # print(desc__)
                    if isinstance(desc__, str):
                        self.cols.append([key] + [desc__])
                    else:
                        self.cols.append([key] + list(desc__))

            self.store_funs[key] = store_fun
            self.idx[key] = slice(k_idx, k_idx + idx_inc)
            # self.dtypes[key] = dtype
            k_idx += idx_inc

        n_lvls = np.max([len(col) for col in self.cols])
        desc_list = []
        for col in self.cols:
            desc_list.append(col + ['']*(n_lvls - len(col)))
        self.desc = np.array(desc_list)

        self.data = np.nan * np.zeros((n_steps, k_idx), dtype=dtype)

    def store(self):
        for key, store_fun in self.store_funs.items():
            idx = self.idx[key]
            self.data[self.k_store, idx] = store_fun()

        self.k_store += 1

    def __getitem__(self, args):
        if isinstance(args, str):
            args = (args,)
        args = np.array(args)
        idx_cmp = ~(args == slice(None))

        row_check = self.desc[:, :len(args)][:, idx_cmp] == args[idx_cmp]
        return self.data[:, np.all(row_check, axis=1)]


if __name__ == '__main__':

    class KFObject:
        def __init__(self):
            self.x = np.zeros(2, dtype=complex)
            self.desc = ['avg', 'D']

        def update(self, t):
            self.x[:] = np.random.randn(2) + 1j * np.random.randn(2)


    class SimObject:
        def __init__(self):
            self.x = np.zeros(3)
            self.desc = np.array([
                ['G1', 'speed'],
                ['G1', 'angle'],
                ['G2', 'speed'],
            ])

        def update(self, t):
            self.x[:] = np.sin(2 * np.pi * 0.5 * t + np.arange(3) * np.pi / 5)

    # Simulation setup
    t_end = 10
    dt = 5e-3
    sim = SimObject()
    kf = KFObject()

    # Initialize variables
    n_steps = int(np.round(t_end/dt))
    t = 0.0
    x = 0

    # def store_fun():
    #     return t, x

    spec = {
        't': lambda: t,
        'x': lambda: x,
        # 'single_vars': ((['x', 't']), store_fun),
        'sim': (sim.desc, lambda: sim.x),
        'kf': (kf.desc, lambda: kf.x),
        'kf_2': lambda: kf.x,
    }
    # Storing results
    res = ResultKeeper(spec=spec, n_steps=n_steps)

    for i_step in range(n_steps):
        t += dt
        sim.update(t)
        kf.update(t)
        x = np.exp(t)
        res.store()
        # res_d.store()

    fig, ax = plt.subplots(3)

    ax[0].plot(res['t'], res['x'])
    ax[1].plot(res['t'], res['sim', 'G1', 'speed'])
    ax[1].plot(res['t'], res['sim', :, 'speed'])
    ax[2].plot(res['t'], res['kf', 'D'].real)
    ax[2].plot(res['t'], res['kf_2', '1'].real)

    # path = 'res_keeper_save_test_file'
    # res.save_to_file(path)
    # res_from_file = ResultKeeperGen.load_from_file(path=path)
    # res_from_file['t']

    plt.show()