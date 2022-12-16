import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import csv
from matplotlib.patches import Rectangle


def generate_plots(ps, result, pf_res, dt, choose_plots='all'):
    py_res_id = ['angle', 'speed', 'e_q_st', 'e_d_st']
    pf_res_id = ['s:phi', 's:speed', 'c:ussq', 'c:ussd']

    gen_names = ps.gen['GEN'].par['name']
    n_gen = len(gen_names)

    if choose_plots == 'all' or choose_plots == 'basic':
        fig, ax_4x1 = plt.subplots(4, 1, sharex=True, squeeze=False)
        fig.suptitle('DynPSSimPy vs DIgSILENT PowerFactory', fontsize=16)
        y_labels = ['Angle', 'Speed', 'Subtransient\nq-axis voltage', 'Subtransient\nd-axis voltage']
        for ax_, label in zip(ax_4x1[:, 0], y_labels):
            ax_.set_ylabel(label)
        ax = ax_4x1[:, 0]
        linewidth = 1
        kwargs = dict(linewidth=linewidth, linestyle='-')
        kwargs_pf = dict(linewidth=linewidth, linestyle=':')  # , marker='.', markevery=35

        for ax_, py_res_id_, pf_res_id_, in zip(ax, py_res_id, pf_res_id):
            pl = []
            pl_pf = []
            for i, gen_name in enumerate(gen_names):
                pl.append(
                    ax_.plot(result[('Global', 't')], result[(gen_name, py_res_id_)], color='C' + str(i), **kwargs)[0])
                pl_pf.append(ax_.plot(pf_res['b:tnow'], pf_res[pf_res_id_][:, i], color='C' + str(i), **kwargs_pf)[0])

        extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        # Create organized list containing all handles for table. Extra represent empty space
        legend_handles = np.vstack([
            [extra, *pl],
            [extra, *pl_pf],
            [extra] * (1 + n_gen),
        ])

        # Define the labels
        legend_labels = np.vstack([
            ['Python', *[''] * n_gen],
            ['PowerFactory', *[''] * n_gen],
            ['', *list(gen_names)],
        ])

        # Create legend
        fig.legend(legend_handles.T.flatten(), legend_labels.T.flatten(), loc='lower center', ncol=1 + n_gen,
                   handletextpad=-1.6)
        fig.subplots_adjust(bottom=0.25)

    if choose_plots == 'all' or choose_plots == 'expanded':
        fig, ax_4xN = plt.subplots(4, ps.gen['GEN'].n_units, sharex=True, sharey='row', squeeze=False)
        fig.suptitle('DynPSSimPy vs DIgSILENT PowerFactory', fontsize=16)
        plt.subplots_adjust(wspace=0, hspace=0)
        y_labels = ['Angle', 'Speed', 'Subtransient\nq-axis voltage', 'Subtransient\nd-axis voltage']
        for ax_, label in zip(ax_4xN[:, 0], y_labels):
            ax_.set_ylabel(label)

        for ax_, gen_name in zip(ax_4xN[-1, :], gen_names):
                ax_.set_xlabel(gen_name)

        for ax_row, py_res_id_, pf_res_id_, in zip(ax_4xN, py_res_id, pf_res_id):
            for i, (ax_, gen_name) in enumerate(zip(ax_row, gen_names)):
                ax_.plot(result[('Global', 't')], result[(gen_name, py_res_id_)], color='C0')
                ax_.plot(pf_res['b:tnow'], pf_res[pf_res_id_][:, i], color='C3')

    if choose_plots == 'all' or choose_plots == 'error':

        fig, ax_4xN = plt.subplots(4, n_gen, sharex=True, sharey='row', squeeze=False)
        fig.suptitle('Relative error', fontsize=16)
        plt.subplots_adjust(wspace=0, hspace=0)
        y_labels = ['Angle', 'Speed', 'Subtransient\nq-axis voltage', 'Subtransient\nd-axis voltage']
        for ax_, label in zip(ax_4xN[:, 0], y_labels):
            ax_.set_ylabel(label)

        for ax_, gen_name in zip(ax_4xN[-1, :], gen_names):
            ax_.set_xlabel(gen_name)

        # Computing the error
        error = 0.0
        for ax_row, py_res_id_, pf_res_id_, in zip(ax_4xN, py_res_id, pf_res_id):
            for i, (ax_, gen_name) in enumerate(zip(ax_row, gen_names)):
                pf_res_new = interp1d(pf_res['b:tnow'], pf_res[pf_res_id_][:, i], fill_value='extrapolate')
                py_time = result[('Global', 't')]
                py_res = result[(gen_name, py_res_id_)]
                error_ = py_res - pf_res_new(py_time)
                error += sum(error_ * dt)
                ax_.plot(py_time, error_ / np.max(abs(py_res)))

    plt.show()


def compute_error(ps, result, pf_res, dt):
    py_res_id = ['angle', 'speed', 'e_q_st', 'e_d_st']
    pf_res_id = ['s:phi', 's:speed', 'c:ussq', 'c:ussd']
    error = 0.0
    gen_names = ps.gen['GEN'].par['name']
    for py_res_id_, pf_res_id_, in zip(py_res_id, pf_res_id):
        for i, gen_name in enumerate(gen_names):
            pf_res_new = interp1d(pf_res['b:tnow'], pf_res[pf_res_id_][:, i], fill_value='extrapolate')
            py_time = result[('Global', 't')]
            py_res = result[(gen_name, py_res_id_)]
            error_ = py_res - pf_res_new(py_time)
            error += sum(np.sqrt(error_**2))/(len(py_time)*len(gen_names))

    return error


def load_pf_res(file_path):
    pf_time_series = np.genfromtxt(file_path, delimiter=',', skip_header=1)

    area = []
    name = []
    extension = []
    variable = []
    unit = []
    print(file_path)
    with open(file_path) as file:
        reader = csv.reader(file)
        row = next(reader, None)
        for item in row:
            delim_1 = item.find('\\')
            delim_2 = item.find('.')
            area.append(item[:delim_1])
            name.append(item[(delim_1 + 1):delim_2])
            extension.append(item[(delim_2 + 1):])
        row = next(reader, None)
        for item in row:
            delim = item.find(' in ')
            if delim >= 0:
                variable.append(item[:delim])
                unit.append(item[(delim + 4):])
            else:
                variable.append(item)
                unit.append('')

    header = [area, name, extension, variable, unit]

    not_isnan = ~np.any(np.isnan(pf_time_series), axis=1)
    pf_time = pf_time_series[not_isnan, 0]
    # plt.plot(pf_time)
    pf_res = dict()
    result_ids = ['b:tnow', 's:phi', 's:speed', 'c:ussd', 'c:ussq']
    for id in result_ids:
        pf_res[id] = pf_time_series[np.ix_(not_isnan, np.where(np.array(header[3]) == id)[0])]

    pf_res['s:phi'] = np.unwrap(pf_res['s:phi'] + np.pi / 2, axis=0)
    pf_res['s:speed'] -= 1
    pf_res['b:tnow'] = pf_res['b:tnow'][:, 0]

    return pf_res