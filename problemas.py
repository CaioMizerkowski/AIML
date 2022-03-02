#!/usr/bin/env python
# coding: utf-8
import numpy as np


class PressureVessel():
    def __init__(self):
        self.epsilon = 1e-6
        self.base = 0.0625
        self.bounds = np.array([[0.1, 99], [0.1, 99], [10, 200], [10, 200]])
        self.active_constrains = True

    def g1(self, x1, x3):
        return max(0.0193*x3-x1, 0)  # <= epsilon

    def g2(self, x2, x3):
        return max(0.00954*x3-x2, 0)  # <= epsilon

    def g3(self, x3, x4):
        # <= epsilon
        return max(-np.pi*x3**2*x4-(4/3)*np.pi*x3**3+1_296_000, 0)

    def g4(self, x4):
        return max(x4-240, 0)  # <= epsilon

    def constrains(self, x):
        if not self.active_constrains:
            return 0
        x1, x2, x3, x4 = x
        x1 = self.force_base(x1)
        x2 = self.force_base(x2)
        return (1/self.epsilon)*(self.g1(x1, x3)+self.g2(x2, x3)+self.g3(x3, x4)+self.g4(x4))

    def force_base(self, x):
        return np.round(x/self.base)*self.base

    def pressure_vessel(self, x):
        x1, x2, x3, x4 = x
        x1 = self.force_base(x1)
        x2 = self.force_base(x2)
        f = 0.6224*x1*x3*x4+1.7781*x2*x3**2+3.1661*x1**2*x4+19.84*x1**2*x3
        return f+self.constrains(x)

    def problem(self, x):
        return self.pressure_vessel(x)


class TensionCompressionSpring():
    def __init__(self):
        self.epsilon = 1e-1
        self.base = 0.0625
        self.bounds = np.array([[0.05, 2], [0.25, 1.3], [2, 15]])
        self.active_constrains = True

    def g1(self, d, D, N):
        return max(1-(D**3*N)/(71785*d**4), 0)

    def g2(self, d, D, __):
        return max((4*D**2-d*D)/(12566*(D*d**3-d**4))+1/(5108*d**2)-1, 0)

    def g3(self, d, D, N):
        return max(1-(140.45*d)/(D**2*N), 0)

    def g4(self, d, D, __):
        return max((D+d)/(1.5)-1, 0)

    def constrains(self, x):
        if not self.active_constrains:
            return 0
        return (1/self.epsilon)*(self.g1(*x)+self.g2(*x)+self.g3(*x)+self.g4(*x))

    def force_base(self, x):
        return np.round(x/self.base)*self.base

    def tension_compression_spring(self, x):
        d, D, N = x
        f = (N+2)*D*d**2
        return f+self.constrains(x)

    def problem(self, x):
        return self.tension_compression_spring(x)


class SpeedReducer():
    def __init__(self):
        self.epsilon = 1e-5
        self.base = 0.0625
        self.bounds = np.array([[2.6, 3.6], [0.7, 0.8], [17, 28], [
                               7.3, 8.3], [7.8, 8.3], [2.9, 3.9], [5.0, 5.5]])
        self.active_constrains = True

    def g1(self, x):
        x1, x2, x3, __, __, __, __ = x
        return max(27/(x1*x2**2*x3)-1, 0)

    def g2(self, x):
        x1, x2, x3, __, __, __, __ = x
        return max(397.5/(x1*x2**2*x3**2)-1, 0)

    def g3(self, x):
        __, x2, x3, x4, __, x6, __ = x
        return max(1.93*x4**3/(x2*x3*x6**4)-1, 0)

    def g4(self, x):
        __, x2, x3, __, x5, __, x7 = x
        return max((1.93*x5**3)/(x2*x3*x7**4)-1, 0)

    def g5(self, x):
        __, x2, x3, x4, __, x6, __ = x
        return max(np.sqrt(((745*x4)/(x2*x3))**2+16.9e6)/(110*x6**3)-1, 0)

    def g6(self, x):
        __, x2, x3, __, x5, __, x7 = x
        return max(np.sqrt(((745*x5)/(x2*x3))**2+157.5e6)/(85*x7**3)-1, 0)

    def g7(self, x):
        __, x2, x3, __, __, __, __ = x
        return max(x2*x3/40-1, 0)

    def g8(self, x):
        x1, x2, __, __, __, __, __ = x
        return max(5*x2/x1-1, 0)

    def g9(self, x):
        x1, x2, __, __, __, __, __ = x
        return max(x1/(12*x2)-1, 0)

    def g10(self, x):
        __, __, __, x4, __, x6, __ = x
        return max((1.5*x6+1.9)/x4-1, 0)

    def g11(self, x):
        __, __, __, __, x5, __, x7 = x
        return max((1.1*x7+1.9)/x5-1, 0)

    def constrains(self, x):
        if not self.active_constrains:
            return 0
        return (1/self.epsilon)*(
            self.g1(x) +
            self.g2(x) +
            self.g3(x) +
            self.g4(x) +
            self.g5(x) +
            self.g6(x) +
            self.g7(x) +
            self.g8(x) +
            self.g9(x) +
            self.g10(x) +
            self.g11(x))

    def force_base(self, x):
        return np.round(x/self.base)*self.base

    def speed_reducer(self, x):
        x1, x2, x3, x4, x5, x6, x7 = x
        f = 0.7854*x1*x2**2*(3.3333*x3**2+14.9334*x3-43.0934)-1.508*x1 * \
            (x6**2+x7**2)+7.4777*(x6**3+x7**3)+0.7854*(x4*x6**2+x5*x7**2)
        return f+self.constrains(x)

    def problem(self, x):
        return self.speed_reducer(x)


if __name__ == '__main__':
    pass
