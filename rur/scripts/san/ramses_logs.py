import os
import numpy as np
def get_times_from_log(log_path, levelmax=22, n_thr=5):
    times = []
    dtype = [('path', 'U16'), ('tstep', 'f8'), ('ttot', 'f8'), ('aexp', 'f8'),
             ('icoarse', 'i4'), ('ifine', 'i4'), ('mem', 'f8', 2), ('umem', 'f8'), ('nstar', 'i4'),
             ('ngrids', 'i4', levelmax+1), ('timer', 'O')]
    dtype_timer = [('minimum', 'f8'), ('average', 'f8'), ('maximum', 'f8'),
                   ('stdev', 'f8'), ('stoav', 'f8'), ('rmn', 'i4'), ('rmx', 'i4'), ('routine', 'U32')]
    for pp in log_path:
        times_sub = []
        timer = None
        f = open(pp, 'r')
        nstar = 0
        lines = f.readlines()
        ngrids = np.zeros(levelmax+1, dtype='i4')
        for i in range(len(lines)):
            line = lines[i]
            if 'Fine step=' in line:
                aexp = float(line[50:61])
                ifine = int(line[11:18])
                mem = [float(line[65:69]), float(line[71:75])]
                if(not np.isfinite(aexp)):
                    print(line)
            if 'New star=' in line:
                nstar = int(line[34:44])
            if 'Time elapsed since last' in line:
                tstep = float(line[37:46])
            if 'Used memory' in line:
                umem = float(line[15:23])
                if 'MB' in line:
                    umem /= 1000.
            if 'Total running time:' in line:
                ttot = float(line[20:36])
            if 'Mesh structure' in line:
                while True:
                    i += 1
                    line = lines[i]
                    if(not 'Level' in line):
                        break
                    else:
                        ilevel = int(line[6:9])
                        ngrids[ilevel] = int(line[13:24])
            if 'Main step' in line:
                icoarse = int(line[11:18])
                times_sub.append((os.path.basename(pp), tstep, ttot, aexp, icoarse, ifine, mem, umem, nstar, ngrids.copy(), timer))
                timer = None
            if 'TIMER' in line:
                timer  = []
                while True:
                    i += 1
                    line = lines[i]
                    if('TOTAL' in line):
                        break
                    minimum = float(line[0:13])
                    average = float(line[13:27])
                    maximum = float(line[27:41])
                    stdev = float(line[41:55])
                    stoav = float(line[55:69])
                    percent = float(line[69:77])
                    rmn = float(line[77:83])
                    rmx = float(line[83:87])
                    routine = (line[87:114]).strip(' ')
                    timer.append((minimum, average, maximum, stdev, stoav, rmn, rmx, routine))
                timer = np.array(timer, dtype=dtype_timer)
        times_sub = np.array(times_sub, dtype=dtype)
        if(len(times_sub) >= n_thr):
            times.append(times_sub)
    if(len(times) > 0):
        times = np.concatenate(times)
    times = np.array(times, dtype=dtype)
    return times

