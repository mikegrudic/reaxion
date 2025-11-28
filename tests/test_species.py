"""Tests for manipulation and parsing of string identifiers for chemical species"""

from jaco.species_strings import base_species, species_charge, add_electron, remove_electron, strip
from jaco.data.atoms import atomic_weights
import pytest

test_species = [
    "e-",
    "C_6H_12O_6",
    "C_6 H_12 O_63+",
    "C_6 H_12 O_6^3+",
    "C_6H_12O_63+",
    "H_2",
    "H_2+",
    "H2",
    "H2+H-",
    "HCO+",
    "He++",
    "He2+",
]


@pytest.mark.parametrize("species", test_species)
def test_species_components(species):
    match species:
        case "e-":
            assert strip(species) == "e-"
            assert species_charge(species) == -1
        case "C_6H_12O_6":
            assert base_species(species) == "C_6H_12O_6"
            assert strip(species) == "C6H12O6"
            assert species_charge(species) == 0
            assert remove_electron(species) == species + "+"
            assert add_electron(species) == species + "-"
        case "C_6 H_12 O_63+" | "C_6H_12O_63+":
            assert strip(species) == "C6H12O63+"
            assert species_charge(species) == 1
            assert remove_electron(species) == species + "+"
            assert add_electron(species) == species.replace("+", "")
        case "C_6 H_12 O_6^3+":
            assert strip(species) == "C6H12O6^3+"
            assert species_charge(species) == 3
            assert remove_electron(species) == "C_6 H_12 O_6^4+"
            assert add_electron(species) == "C_6 H_12 O_6++"
        case "H_2" | "H2":
            assert strip(species) == "H2"
            assert species_charge(species) == 0
            assert remove_electron(species) == species + "+"
            assert add_electron(species) == species + "-"
        case "H-":
            assert strip(species) == "H-"
            assert species_charge(species) == -1
            assert remove_electron(species) == "H"
            assert add_electron(species) == "H--"
        case "HCO+":
            assert strip(species) == "HCO+"
            assert species_charge(species) == 1
            assert remove_electron(species) == "HCO++"
            assert add_electron(species) == "HCO"
        case "He++":  # HeIII
            assert strip(species) == "He++"
            assert species_charge(species) == 2
            assert remove_electron(species) == "He^3+"
            assert add_electron(species) == "He+"
        case "He2+":  # this is intended to denote the dihelium ion
            assert strip(species) == "He2+"
            assert species_charge(species) == 1
            assert remove_electron(species) == "He2++"
            assert add_electron(species) == "He2"
