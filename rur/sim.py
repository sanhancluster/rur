from rur import uri


class NewHorizon(uri.RamsesSnapshot):
    def __init__(self, iout, box=None):
        repo = '/storage6/NewHorizon'
        super().__init__(repo, iout, mode='nh', box=box)


class NewHorizon2(uri.RamsesSnapshot):
    def __init__(self, iout, box=None):
        repo = '/storage7/NH2'
        super().__init__(repo, iout, mode='nh2', box=box)


class NewCluster(uri.RamsesSnapshot):
    def __init__(self, iout, box=None):
        repo = '/storage7/NewCluster'
        super().__init__(repo, iout, mode='nc', box=box)
