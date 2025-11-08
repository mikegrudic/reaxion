"""Definition of various sympy symbols used throughout the module"""

import sympy as sp
from typing import Union
import sympy as sp
from astropy.constants import k_B, m_p
from astropy import units as u

T = sp.Symbol("T", nonnegative=True)  # temperature
T5 = T / 10**5
T6 = T / 10**6
T3 = T / 10**3
T4 = T / 10**4
c_s = sp.Symbol("c_s", nonnegative=True)  # sound speed
G = sp.Symbol("G", nonnegative=True)  # gravitational constant
ρ = sp.Symbol("ρ", nonnegative=True)  # total mass density
n_e = sp.Symbol("n_e-", nonnegative=True)  # electron number density
z = sp.Symbol("z", nonnegative=True)  # cosmological redshift
t = sp.Symbol("t")  # time
dt = sp.Symbol("Δt", nonnegative=True)
n_Htot = sp.Symbol("n_Htot", nonnegative=True)


boltzmann_cgs = k_B.to(u.erg / u.K).value
protonmass_cgs = m_p.to(u.g).value
# write down internal energy density in terms of number densities - this defines the EOS


def d_dt(species: Union[str, sp.core.symbol.Symbol]):
    if isinstance(species, str):
        return sp.diff(sp.Function(n_(species))(t), t)
    else:
        return sp.diff(sp.Function(species)(t), t)


def n_(species: str):
    match species:
        case "heat":
            return sp.Symbol(f"⍴u", nonnegative=True)
        case _:
            return sp.Symbol(f"n_{species}", nonnegative=True)


egy_density = boltzmann_cgs * T * (1.5 * (n_("e-") + n_("H") + n_("H+") + n_("He") + n_("He+") + n_("He++")))
rho = protonmass_cgs * (n_("H") + n_("H+") + 4 * (n_("He") + n_("He+") + n_("He++")))
u = sp.simplify(egy_density / rho)


def x_(species: str):
    return sp.Symbol(f"x_{species}", nonnegative=True)


def BDF(species):
    if species in ("T", "u"):  # this is the heat equation
        return rho * (u - sp.Symbol("u_0", nonnegative=True)) / dt
    else:
        return (n_(species) - sp.Symbol(str(n_(species)) + "_0", nonnegative=True)) / dt
