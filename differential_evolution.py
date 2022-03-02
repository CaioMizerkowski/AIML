#!/usr/bin/env python
# coding: utf-8
import numpy as np


class DifferentialEvolution():

    def __init__(self, fitness, bounds, seed=np.random.choice(range(1000)),
                 X=None, G=1000, mutation=0.75, recombination=0.75):
        self._bounds = bounds
        self._dim = bounds.shape[0]
        self._fitness = fitness
        self._pop_size = self._dim*15
        self._mr = mutation
        self._cr = recombination
        self._G = G
        self._g = 0
        self._seed = seed
        self.best_x = None
        self.X_vector = None
        self.X = None
        self._gen_x_orig(X)
        self._mutation = self._mutation_rand_1_bin
        self._shape = self.X.shape

    def _check_bound(self, u):
        for idx, k in enumerate(u):
            if self._bounds[idx][0] > k:
                u[idx] = self._bounds[idx][0]
            if self._bounds[idx][1] < k:
                u[idx] = self._bounds[idx][1]
        return u

    def _selection(self, x, u):
        u = self._check_bound(u)
        x_func = self._fitness(x)
        u_func = self._fitness(u)
        if x_func <= u_func:
            return x
        return u

    def _recombination(self, x, v):
        u = np.zeros(shape=v.shape)
        cidx = np.random.choice(range(self._shape[1]), size=1, replace=False)
        for idx, (x0, v0) in enumerate(zip(x, v)):
            u[idx] = v0 if np.random.uniform() < self._cr or idx == cidx else x0
        return u

    def _mutation_best_1_bin(self, __=None):
        r2, r3 = np.random.choice(range(self._shape[0]), size=2, replace=False)
        return self.best_x+self._mr*(self.X[r2]-self.X[r3])

    def _mutation_target_to_best_1_bin(self, x):
        r2, r3 = np.random.choice(range(self._shape[0]), size=2, replace=False)
        return x+self._mr*(self.best_x-x)+self._mr*(self.X[r2]-self.X[r3])

    def _mutation_rand_1_bin(self, __=None):
        r1, r2, r3 = np.random.choice(
            range(self._shape[0]), size=3, replace=False)
        return self.X[r1]+self._mr*(self.X[r2]-self.X[r3])

    def differential_evolution(self):
        self.X_vector = [self.X]
        np.random.seed(self._seed)
        while self._g < self._G:
            X_new = np.zeros(self.X.shape)
            self.best_x = self.X[np.array(map(self._fitness, self.X)).argmin()]
            for idx, x in enumerate(self.X):
                v = self._mutation(x)
                u = self._recombination(x, v)
                X_new[idx] = self._selection(x, u)
            self.X = X_new.copy()
            self.X_vector.append(self.X)
            self._g += 1

    def run(self):
        self.differential_evolution()

    @property
    def fun(self):
        return min(map(self._fitness, self.X))

    def _gen_x_orig(self, X=None):
        if X:
            self.X = X
        else:
            np.random.seed(self._seed)
            self.X = np.zeros(shape=(self._pop_size, self._dim))
            for i in range(self._dim):
                self.X.T[i] = np.random.uniform(low=self._bounds[i][0],
                                                high=self._bounds[i][1],
                                                size=(self._pop_size))


if __name__ == '__main__':
    pass
