#!/usr/bin/env python
# coding: utf-8
import numpy as np


class GeneticAlgorithm():
    def __init__(self, fitness, bounds, seed=np.random.choice(range(1000)),
                 X=None, G=1000, mutation=None, recombination=None):
        self._bounds = bounds
        self._dim = bounds.shape[0]
        self._fitness = fitness
        self._pop_size = self._dim*15
        self._mr = mutation
        self._cr = recombination
        self._G = G
        self._g = 0
        self._b = 1.2
        self._seed = seed
        self.best_x = None
        self.X_vector = None
        self.X = None
        self._gen_x_orig(X)
        self._mutation = self._mutation_non_uniform
        self._crossover = self._crossover_arithmetical
        self._shape = self.X.shape

    def _crossover_arithmetical(self, x):
        if np.random.uniform() > self._cr:
            return x
        r = np.random.uniform()
        idx = np.random.choice(range(self._shape[1]), size=1, replace=False)[0]
        return r*x+(1-r)*self.X[idx]

    def _check_bound(self, v):
        for idx, k in enumerate(v):
            if self._bounds[idx][0] > k:
                v[idx] = self._bounds[idx][0]
            if self._bounds[idx][1] < k:
                v[idx] = self._bounds[idx][1]
        return v

    def _selection(self, x, v):
        v = self._check_bound(v)
        x_func = self._fitness(x)
        v_func = self._fitness(v)
        if x_func <= v_func:
            return x
        return v

    def _mutation_null(self, x):
        return x

    def _mutation_uniform(self, x):
        if np.random.uniform() > self._mr:
            return x
        k = np.random.choice(range(self._dim), size=1, replace=False)
        if k < self._dim-1:
            left_x = min(x[k-1], x[k+1])
            right_x = max(x[k-1], x[k+1])
        else:
            left_x = min(x[k-1], x[0])
            right_x = max(x[k-1], x[0])
        new_x = x.copy()
        new_x[k] = np.random.uniform(low=left_x, high=right_x)
        return new_x

    def _mutation_non_uniform(self, x):
        if np.random.uniform() > self._mr:
            return x
        k = np.random.choice(range(self._dim), size=1, replace=False)
        d = np.random.choice(range(2), size=1, replace=False)
        if k < self._dim-1:
            left_x = min(x[k-1], x[k+1])
            right_x = max(x[k-1], x[k+1])
        else:
            left_x = min(x[k-1], x[0])
            right_x = max(x[k-1], x[0])

        new_x = x.copy()
        r = np.random.uniform()
        if d:
            new_x[k] = x[k]+(right_x-x[k])*r*(1-self._g/self._G)**self._b
        else:
            new_x[k] = x[k]+(left_x-x[k])*r*(1-self._g/self._G)**self._b
        new_x[k] = np.random.uniform(low=left_x, high=right_x)
        return new_x

    def genetic_algorithm(self):
        self.X_vector = [self.X]
        self._g = 0
        while self._g < self._G:
            X_new = np.zeros(self._shape)
            self.best_x = self.X[np.array(map(self._fitness, self.X)).argmin()]
            for idx, x in enumerate(self.X):
                v = self._mutation(x)
                v = self._crossover(v)
                X_new[idx] = self._selection(x, v)
            self.X = X_new
            self.X_vector.append(self.X)
            self._g += 1

    def run(self):
        self.genetic_algorithm()

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
