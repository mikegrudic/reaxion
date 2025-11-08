"""Implementation of Equation and EquationSystem for representing, manipulationg, and constructing conservation laws"""

import sympy as sp
from . import symbols
from .symbols import d_dt, n_, x_, dt, t, BDF, n_Htot, T
from .data import SolarAbundances
from jax import numpy as jnp
import numpy as np
from .numerics import newton_rootsolve


class Equation(sp.core.relational.Equality):
    """Sympy equation where we overload addition/subtraction to apply those operations to the RHS, for summing rate
    equations"""

    def get_summand(self, other):
        """Value-check the operand and return the quantity to be summed in the operation: the expression itself if an expression, or the RHS"""
        if isinstance(other, sp.core.relational.Equality):
            if self.lhs != other.lhs:
                raise ValueError(
                    "Tried to sum incompatible equations. Equation summation only defined for differential equations with the same LHS."
                )
            else:
                return other.rhs
        elif isinstance(other, sp.logic.boolalg.BooleanAtom):
            return 0
        else:
            return other

    def __add__(self, other):
        summand = self.get_summand(other)
        return Equation(self.lhs, self.rhs + summand)

    def __sub__(self, other):
        summand = self.get_summand(other)
        return Equation(self.lhs, self.rhs - summand)

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        self = self + other
        return self

    def __isub__(self, other):
        self = self - other
        return self


class EquationSystem(dict):
    """Dict of symbolic expressions with certain superpowers for manipulating sets of conservation equations."""

    def copy(self):
        new = EquationSystem()
        for k in self:
            new[k] = self[k]
        return new

    def __getitem__(self, __key: str):
        """Dict getitem method where we initialize a differential equation for the conservation of a species if the key
        does not exist"""
        if __key not in self:
            self.__setitem__(__key, Equation(d_dt(n_(__key)), 0))  # technically should only be n_ if this is a species
            # need to make sure that d/dt's don't add up when composing equations
        return super().__getitem__(__key)

    def __add__(self, other):
        """Return a dict whose values are the sum of the values of the operands"""
        keys = self.keys() | other.keys()
        new = EquationSystem()
        for k in keys:
            new[k] = self[k] + other[k]
        return new

    @property
    def symbols(self):
        """Returns the set of all symbols in the equations"""
        all = set()
        for e in self.values():
            all.update(e.free_symbols)
        if t in all:  # leave time out
            all.remove(t)
        return all

    @property
    def jacobian(self):
        """Returns a dict of dicts representing the Jacobian of the RHS of the system. Keys are the names of the
        conserved quantities and subkeys are the variable of differentiation.
        """
        return {k: {s: sp.diff(e.rhs, s) for s in self.symbols} for k, e in self.items()}

    def subs(self, expr, replacement):
        """Substitute symbolic expressions throughout the whole network."""
        for k, e in self.items():
            self[k] = e.subs(expr, replacement)

    def reduced(self, knowns, time_dependent=[]):
        subsystem = self.copy()
        subsystem.set_time_dependence(time_dependent)
        subsystem.do_conservation_reductions(knowns, time_dependent)
        if "T" in knowns and "T" not in time_dependent:
            del subsystem["heat"]
        return subsystem

    def set_time_dependence(self, time_dependent_vars):
        """Insert backward-difference formulae or set to steady state"""
        # put in backward differences
        for q in self:
            if q in time_dependent_vars:  # insert backward-difference formula
                self[q] = Equation(BDF(q), self[q].rhs)
            else:
                self[q] = Equation(0, self[q].rhs)
        if "T" in time_dependent_vars:  # special behaviour
            self["heat"] = Equation(BDF("T"), self["heat"].rhs)
            self["u"] = Equation(sp.Symbol("u", nonnegative=True), symbols.u)

    def do_conservation_reductions(self, knowns, time_dependent_vars):
        """Eliminate equations from the system using known conservation laws."""
        # charge neutrality
        if "e-" not in time_dependent_vars:
            self.subs(n_("e-"), n_("H+") + n_("He+") + 2 * n_("He++"))  # general: sum(n_species * ion charge)
            del self["e-"]

        if "n_Htot" in knowns:
            #  general: sum(n_(species containing H) / (number of H in species))  - n_("H_2") / 2 #
            if "H+" not in time_dependent_vars:
                self.subs(n_("H+"), n_Htot - n_("H"))
                if "H+" in self:
                    del self["H+"]

            if "He++" not in time_dependent_vars:
                y = sp.Symbol("y")
                self.subs(n_("He++"), n_Htot * y - n_("He") - n_("He+"))
                if "He++" in self:
                    del self["He++"]

            # since we have n_Htot let's convert all other n's to x's
            for s in self.symbols:
                if "n_" in str(s) and "Htot" not in str(s):
                    species = str(s).split("_")[1]
                    self.subs(s, n_Htot * x_(species))

            # general: substitute highest ionization state with n_Htot * x_element - sum of lower ionization states

    @property
    def rhs(self):
        """Return as dict of rhs-lhs instead of equations"""
        return {k: e.rhs - e.lhs for k, e in self.items()}

    @property
    def ccode(self):
        cse, cseval = sp.cse(self.rhs.values(), order="none")
        return sp.ccode(cse, standard="c99"), sp.ccode(cseval, standard="c99")

    @property
    def rhs_scaled(self):
        """Returns a scaled version of the the RHS pulling out the usual factors affecting collision rates"""
        return [r / (T**0.5 * n_Htot * n_Htot * 1e-12) for r in self.rhs.values()]

    def solve(
        self,
        knowns,
        guesses,
        time_dependent=[],
        input_abundances=True,
        output_abundances=True,
        reduce_network=True,
        dt=None,
        model="default",
        verbose=False,
        tol=1e-3,
        careful_steps=10,
    ):
        def printv(*a, **k):
            """Print only if locally verbose=True"""
            if verbose:
                print(*a, **k)

        # first: check knowns and guesses are all same size
        num_params = np.array([len(np.array(guesses[g])) for g in guesses] + [len(np.array(knowns[g])) for g in knowns])
        if not np.all(num_params == num_params[0]):
            raise ValueError("Input parameters and initial guesses must all have the same shape.")
        num_params = num_params[0]

        subsystem = self.reduced(knowns, time_dependent)
        symbols = subsystem.symbols
        num_equations = len(subsystem)

        # are there any symbols for which we can make a reasonable assumption or directly solve the steady-state approximation?
        prescriptions = {"y": SolarAbundances.x("He"), "Y": SolarAbundances.mass_fraction["He"], "Z": 1.0}
        assumed_values = {}
        if len(symbols) > num_equations + len(knowns):
            undetermined_symbols = symbols.difference(set(guesses))
            printv(f"Undetermined symbols: {undetermined_symbols}")
            for s in undetermined_symbols:
                # if we have a prescription for this quantity, plug it in here. This should eventually be specified at the model level.
                if str(s) in prescriptions:
                    # case 1: we have given a value, which we should add to the list of knowns
                    assumed_values[str(s)] = np.repeat(prescriptions[str(s)], num_params)
                    printv(f"{s} not specified; assuming {s}={prescriptions[str(s)]}.")
                    symbols = subsystem.symbols
                    # case 2: we have given an expression in terms of the other available quantities: we need to subs it

        # ok now we should have number of symbols unknowns + knowns
        printv(
            f"Free symbols: {symbols}\n Known values: {list(knowns)}\n Assumed values: {list(assumed_values)} Equations solved: {list(subsystem.rhs)}"
        )
        if len(symbols) != len(knowns | assumed_values) + len(subsystem):
            raise ValueError(
                f"Number of free symbols is {len(symbols)} != number of knowns {len(knowns)} + number of equations {len(subsystem)}\n"
            )
        else:
            printv(
                f"It's morbin time. Solving for {set(guesses)} based on input {set(knowns)} and assumptions about {set(assumed_values)}"
            )

        guessvals = {}
        paramvals = {}
        for s in subsystem.symbols:
            for g in guesses:
                if g == str(s) or f"x_{g}" == str(s):
                    guessvals[s] = guesses[g]
            for k in knowns | assumed_values:
                if k == str(s) or f"x_{k}" == str(s):
                    paramvals[s] = (knowns | assumed_values)[k]

        lambda_args = [list(guessvals.keys()), list(paramvals.keys())]
        func = sp.lambdify(lambda_args, subsystem.rhs_scaled, modules="jax", cse=True)

        tolerance_vars = [x_("H"), x_("He+")]
        if "T" in guesses:
            tolerance_vars += [sp.Symbol("T")]
        if dt is not None:
            tolerance_vars += [sp.Symbol("u"), subsystem["heat"]]  # converge on the internal energy and  cooling rate
        tolfunc = sp.lambdify(lambda_args, tolerance_vars, modules="jax", cse=True)

        def f_numerical(X, *params):
            """JAX function to rootfind"""
            return jnp.array(func(X, params))

        def tolerance_func(X, *params):
            """Solution will terminate if the relative change in this quantity is < tol"""
            return jnp.array(tolfunc(X, params))

        guess_in = jnp.array([g for g in guessvals.values()]).T
        params_in = jnp.array([p for p in paramvals.values()]).T
        sol = newton_rootsolve(
            f_numerical,
            guess_in,
            params_in,
            tolfunc=tolerance_func,
            rtol=tol,
            careful_steps=careful_steps,
            nonnegative=True,
        )
        return sol

    # print(sol)


#        return func, guessvals,
#       print(func(guessvals.values(), paramvals.values()))

# get into ordered array form


#        guess_vals = guesses[keys[u]

#       print(func(guess_vals, param_vals))

# def f_numerical(X, *params):
#     """Function to rootfind - these are the rates"""
#     return 1e20 * jnp.array(func(*X, *params))

# We also specify a function of the parameters to use for our stopping criterion:
# converge electron and H abundance to desired tolerance.
# tolerance_vars = [self.apply_network_reductions(n_("e-")), n_("H")]
# if thermo:
#     tolerance_vars += [sp.Symbol("T")]
# if dt is not None:
#     tolerance_vars += [sp.Symbol("u"), net_heat]  # converge on the cooling rate
# tolfunc = sp.lambdify(unknowns + known_variables, tolerance_vars, modules="jax", cse=True)

# def tolerance_func(X, *params):
#     """Solution will terminate if the relative change in this quantity is < tol"""
#     return jnp.array(tolfunc(*X, *params))

#        indices = # establish indexing for the different equations and


#        f_numerical = subsystem.rhs

# def broadcast_arraydicts(self, dict1, dict2):
#     d1,d2 = dict1.copy(), dict2.copy()

#     lengths1 = {np.array(a).shape for k, a in dict1.items()}
#     lengths2 = {len(a) for k, a in dict2.items()}
#     # case 1: all the same length,
#     if le

# @property
# def steadystate(self, species=None):
#     """Returns the system with all time derivatives set to 0"""
#     return {k: Equation(0, e.rhs) for k, e in self.items()}  # steadystate_equations

# @property
# def network_ions(self):
#     """Returns the list of ions involved in a process"""
#     return [s for s in self.network if is_an_ion(s)]

# def apply_network_reductions(self, expr):
#     """Applies the replacements given by network_reduction_replacements to a symbolic expression"""
#     out = expr
#     for _ in range(2):  # 2 passes to avoid ordering issues
#         for n, r in self.network_reduction_replacements.items():
#             out = out.subs(n, r)
#     return out

# @property
# def reduced_network(self):
#     """
#     Returns the chemistry network after substituting known conservation laws:

#     n_atom = sum(n_{species containing atom} * number of atoms in species)
#     n_e- = sum(ion charge * n_ion) - want to keep n_e- in the explicit updates, so eliminate the highest ions?

#     This reduces the network of N rate equations to N - (num_atoms + 1).
#     """

#     replacements = self.network_reduction_replacements

#     reduced_network = {}
#     for s, rhs in self.network.items():
#         if n_(s) in replacements:
#             continue
#         else:
#             rhs = self.apply_network_reductions(rhs)
#         reduced_network[s] = rhs
#     return reduced_network

# def get_thermochem_network(self, reduced=True):
#     """Returns the network including all chemical processes plus the gas heating-cooling equation"""
#     if reduced:
#         network = self.reduced_network
#     else:
#         network = self.network
#     return network | {"T": self.apply_network_reductions(self.heat)}  # combine the dicts


# def eulerify(symbol):


# @property
# def network_reduction_replacements(self):
#     """Replacements for reducing the chemistry network with conservation laws"""
#     Y = sp.Symbol("Y")  # general: mass fractions of different atoms other than H. n_i,tot = n_i + sum(n_ions of i)
#     nHtot = sp.Symbol("n_Htot")  # basically always want this

#     substitutions = {
#         n_("e-"): n_("H+") + n_("He+") + 2 * n_("He++"),
#         # n_("He+"): n_("e-") - n_("H+") - 2 * n_("He++"),
#         n_("H+"): nHtot - n_("H"),
#         n_("He++"): Y / (4 - 4 * Y) * nHtot - sp.Symbol("n_He") - sp.Symbol("n_He+"),
#     }
#     return substitutions

# def apply_network_reductions(self, expr):
#     """Applies the replacements given by network_reduction_replacements to a symbolic expression"""
#     out = expr
#     for _ in range(2):  # 2 passes to avoid ordering issues
#         for n, r in self.network_reduction_replacements.items():
#             out = out.subs(n, r)
#     return out

# @property
# def reduced_network(self):
#     """
#     Returns the chemistry network after substituting known conservation laws:

#     n_atom = sum(n_{species containing atom} * number of atoms in species)
#     n_e- = sum(ion charge * n_ion) - want to keep n_e- in the explicit updates, so eliminate the highest ions?

#     This reduces the network of N rate equations to N - (num_atoms + 1).
#     """

#     replacements = self.network_reduction_replacements

#     reduced_network = {}
#     for s, rhs in self.network.items():
#         if n_(s) in replacements:
#             continue
#         else:
#             rhs = self.apply_network_reductions(rhs)
#         reduced_network[s] = rhs
#     return reduced_network

# def get_thermochem_network(self, reduced=True):
#     """Returns the network including all chemical processes plus the gas heating-cooling equation"""
#     if reduced:
#         network = self.reduced_network
#     else:
#         network = self.network
#     return network | {"T": self.apply_network_reductions(self.heat)}  # combine the dicts


#    def backward_eulerfy

# def jacobian

# def numerify

# def numerify_jacobian
