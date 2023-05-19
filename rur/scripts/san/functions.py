import numpy as np
aexp = 0.
def get_times_from_log(log_path):
    steps = []
    timers = []
    for pp in log_path:
        f = open(pp, 'r')
        lines = f.readlines()
        for line in lines:
            if 'Fine step=' in line:
                aexp = float(line[50:61])
                if(not np.isfinite(aexp)):
                    print(line)
            if 'Time elapsed since last' in line:
                steps.append([float(line[37:46]), aexp])
            if 'TIMER' in line:
                timers.append()

    return steps