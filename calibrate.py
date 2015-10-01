import os
import time
import copy
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt


def if_earlier(time_str, std_time_str='2014-07-01', fmt_str='%Y-%m-%d'):
    std_struct_time = time.strptime(std_time_str, fmt_str)
    struct_time = time.strptime(time_str, fmt_str)
    return struct_time < std_struct_time


def convert_force(force_volt, time_str):
    """
    Calibration note taken from Mouse+data+Atoh1+Piezo2+UVA+2015-9-25.xlsx
    """
    if if_earlier(time_str):
        force = 0.69 * force_volt
    else:
        force = 175.53 * force_volt
    return force


def convert_displ(displ_volt, time_str):
    """
    Calibration note taken from Mouse+data+Atoh1+Piezo2+UVA+2015-9-25.xlsx
    """
    if if_earlier(time_str):
        displ = 2.5 * displ_volt
    else:
        displ = 2.5 * displ_volt * 1000
    return displ


def calibrate_file(root, fname):
    data_old = loadmat('%s\\%s' % (root, fname))
    data_new = copy.deepcopy(data_old)
    for key, item in data_old.items():
        if key.startswith('OUT_PUT_D'):
            data_new['C' + key] = convert_displ(item, fname[:10])
        elif key.startswith('OUT_PUT_F'):
            data_new['C' + key] = convert_force(item, fname[:10])
    savemat('%s\\%s_calibrated.mat' % (root, fname[:-4]), data_new,
            do_compression=True)
    return data_new


if __name__ == '__main__':
    for root, subdirs, files in os.walk('data'):
        for fname in files:
            if fname.endswith('.mat') and not fname.endswith('calibrated.mat'):
                calibrate_file(root, fname)
