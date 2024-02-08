from modulus.sym import quantity
from sympy import Symbol

class parameterRangeContainer():
    def __init__(self, ranges):
        self.ranges = {}
        for key in ranges:
            if type(ranges[key][0]) == tuple:
                self.ranges[key] = (quantity(ranges[key][0][0], ranges[key][1]), quantity(ranges[key][0][1], ranges[key][1]))
            else:            
                self.ranges[key] = (quantity(ranges[key][0], ranges[key][1]), quantity(ranges[key][0], ranges[key][1]))

    def nondim(self, nd):
        self.nondimRanges = {}
        for key in self.ranges:
            self.nondimRanges[key] = (nd.ndim(self.ranges[key][0]), nd.ndim(self.ranges[key][1]))
        return self.nondimRanges

    def minval(self):
        minRange = {}
        for key in self.nondimRanges:
            minRange[key] = self.nondimRanges[key][0]
        # print(minRange)
        return minRange

    def maxval(self):
            maxRange = {}
            for key in self.nondimRanges:
                maxRange[key] = self.nondimRanges[key][1]
            return maxRange