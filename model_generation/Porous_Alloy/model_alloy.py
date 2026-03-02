#!/usr/bin/env python3

import numpy as np
from ase.build import bulk, make_supercell
from ase.io import read, write
from ase.symbols import symbols2numbers


### Function to randomly substitute element based
### `n_replaced` `elemement1` replaced by `n_replaced` `elemement1` (randomly)
def random_substitute(atoms, element1, element2, n_replaced):
    replaced_element_idx = atoms.symbols.indices()[element1]  # Index of all element 1
    # Select randomly in element 1
    replaced_idx = np.random.choice(replaced_element_idx, n_replaced, replace=False)
    # Replaced selected with element 2
    atoms.numbers[replaced_idx] = symbols2numbers(element2)[0]


### Sort element in order (based on Z)
### Return sorted list of element
def sorted_element(atoms):
    species = list(atoms.symbols.species())  # Chemical Elements
    Z_species = symbols2numbers(species)  # Atomic Numbers
    sorted_species = np.array(species)[np.argsort(Z_species)].tolist()
    print(f"Sorted Species: {sorted_species}")
    return sorted_species


### Unit cell
atoms = bulk("Ni", "fcc", a=3.57, cubic=True)

### Super cell
atoms *= (50, 50, 50)

### Filter by inequation
x = atoms.positions[:, 0]
y = atoms.positions[:, 1]
z = atoms.positions[:, 2]
L = atoms.cell[0, 0]

### n_cell
n_cell = 1
L = L / n_cell

### TPMS Surface
surface = (
    np.sin(2 * np.pi / L * x) * np.cos(2 * np.pi / L * y)
    + np.sin(2 * np.pi / L * y) * np.cos(2 * np.pi / L * z)
    + np.sin(2 * np.pi / L * z) * np.cos(2 * np.pi / L * x)
)

### Filter based on threshold value (which decide relative density)
volume_fraction = 0.3
t_threshold = np.percentile(surface, volume_fraction * 100)
mask = surface < t_threshold
atoms = atoms[mask]

print(
    f"Filtered: Volume Fraction = {mask.sum()}/{mask.size} = {mask.sum() / mask.size}"
)

### Random substitution -> Al0.5CoCrFeNi
n_atoms = len(atoms)
random_substitute(atoms, "Ni", "Al", n_atoms // 9 * 1)
random_substitute(atoms, "Ni", "Fe", n_atoms // 9 * 2)
random_substitute(atoms, "Ni", "Cr", n_atoms // 9 * 2)
random_substitute(atoms, "Ni", "Co", n_atoms // 9 * 2)

### Sort elements
sorted_species = sorted_element(atoms)

### Export
# write("Al0.5CrFeCoNi_gyroid.xsf", atoms, "xsf")
# write("Al0.5CrFeCoNi_gyroid.xyz", atoms, "extxyz")
write(
    "Al0.5CrFeCoNi_gyroid.lmp",
    atoms,
    "lammps-data",
    specorder=sorted_species,
    masses=True,
)
