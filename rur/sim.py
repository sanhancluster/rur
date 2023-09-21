from rur import uri


class NewHorizon(uri.RamsesSnapshot):
    def __init__(self, iout, box=None):
        repo = '/storage6/snapshots/NewHorizon'
        super().__init__(repo, iout, mode='nh', box=box)


class NewHorizon2(uri.RamsesSnapshot):
    def __init__(self, iout, box=None):
        repo = '/storage6/snapshots/NewHorizon2'
        super().__init__(repo, iout, mode='nh2', box=box)


class NewCluster(uri.RamsesSnapshot):
    def __init__(self, iout, box=None):
        repo = '/storage7/snapshots/NewCluster'
        super().__init__(repo, iout, mode='nc', box=box)
