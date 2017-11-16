

class StartEdge:
    def __init__(self):
        self.s = -1
        self.t = -1
        self.identifier = -1

    def source(self):
        return self.s

    def target(self):
        return self.t 

    def is_valid(self):
        return True

    def id(self):
        return self.identifier
