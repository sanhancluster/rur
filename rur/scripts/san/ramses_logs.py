import os
import numpy as np
def get_times_from_log(log_path):
    times = []
    for pp in log_path:
        f = open(pp, 'r')
        lines = f.readlines()
        for line in lines:
            if 'Fine step=' in line:
                aexp = float(line[50:61])
                ifine = int(line[11:18])
                if(not np.isfinite(aexp)):
                    print(line)
            if 'Time elapsed since last' in line:
                tstep = float(line[37:46])
            if 'Total running time:' in line:
                ttot = float(line[20:36])
            if 'Main step' in line:
                icoarse = int(line[11:18])
                times.append((os.path.basename(pp), tstep, ttot, aexp, icoarse, ifine))
    return np.array(times, dtype=[('path', 'U16'), ('tstep', 'f8'), ('ttot', 'f8'), ('aexp', 'f8'), ('icoarse', 'f8'), ('ifine', 'f8')])
