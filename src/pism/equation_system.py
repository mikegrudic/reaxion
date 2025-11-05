from collections import defaultdict
import sympy as sp
import numpy as np


class EquationSystem(defaultdict):
    """Dict of symbolic equations with certain superpowers

    This this should be able to do:

    Apply and reverse network reductions from conservation laws

    Put in sensible defaults for symbols that are not specified
    """

    def __init__(self):
        super().__init__(int)

    @property
    def symbols(self):
        all = set()
        for e in self.values():
            all.update(e.free_symbols)
        return all

    @property
    def equations(self):
        return tuple(self.values())

    @property
    def jacobian(self):
        return {k: {s: sp.diff(e, s) for s in self.symbols} for k, e in self.items()}


#    def backward_eulerfy

# def jacobian

# def numerify

# def numerify_jacobian
