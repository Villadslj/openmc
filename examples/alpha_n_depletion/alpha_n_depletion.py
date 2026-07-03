"""Example: Fixed-source depletion with (α,n) source recomputation.

This example demonstrates how to run a fixed-source depletion calculation
where (α,n) neutron sources are recomputed from updated material compositions
at each depletion timestep.

The geometry is a simple sphere of UO2 fuel surrounded by a reflector.
An (α,n) source is defined for the fuel material, and an optional external
neutron source is included to demonstrate source combination.

Requirements
------------
- openmc (with C++ library compiled)
- alphanso package (pip install alphanso)
- Nuclear data (OPENMC_CROSS_SECTIONS set)
- Depletion chain file

"""

import openmc
from openmc.alpha_n import AlphaNSource
from openmc.deplete import (
    AlphaNFixedSourceOperator,
    FixedSourceComponent,
    PredictorIntegrator,
)

# =============================================================================
# Materials
# =============================================================================

# Fuel material - UO2 with depletable flag
fuel = openmc.Material(name='UO2 Fuel')
fuel.add_nuclide('U235', 0.05, 'wo')
fuel.add_nuclide('U238', 0.85, 'wo')
fuel.add_nuclide('O16', 0.10, 'wo')
fuel.set_density('g/cm3', 10.0)
fuel.volume = 523.6  # Volume of sphere with r=5 cm [cm³]
fuel.depletable = True

# Reflector material
reflector = openmc.Material(name='Water Reflector')
reflector.add_nuclide('H1', 2.0, 'ao')
reflector.add_nuclide('O16', 1.0, 'ao')
reflector.set_density('g/cm3', 1.0)

materials = openmc.Materials([fuel, reflector])

# =============================================================================
# Geometry
# =============================================================================

fuel_sphere = openmc.Sphere(r=5.0)
outer_sphere = openmc.Sphere(r=15.0, boundary_type='vacuum')

fuel_cell = openmc.Cell(fill=fuel, region=-fuel_sphere)
reflector_cell = openmc.Cell(fill=reflector, region=+fuel_sphere & -outer_sphere)

universe = openmc.Universe(cells=[fuel_cell, reflector_cell])
geometry = openmc.Geometry(universe)

# =============================================================================
# Settings
# =============================================================================

settings = openmc.Settings()
settings.run_mode = 'fixed source'
settings.particles = 5000
settings.batches = 20

# Initial source will be overwritten by the operator, but we need a placeholder
settings.source = [openmc.IndependentSource(
    energy=openmc.stats.Watt(),
    space=openmc.stats.Point(),
)]

# =============================================================================
# Model
# =============================================================================

model = openmc.Model(geometry, materials, settings)

# =============================================================================
# (α,n) Source Configuration
# =============================================================================

# Define (α,n) source for the fuel material
# include_sf=True includes spontaneous fission neutrons in addition to (α,n)
alpha_n_sources = [
    AlphaNSource(material=fuel, include_sf=True),
]

# =============================================================================
# Optional: Additional Fixed Source
# =============================================================================

# An external neutron source (e.g., a startup source or calibration source)
external_source = openmc.IndependentSource(
    energy=openmc.stats.Watt(a=0.988e6, b=2.249e-6),
    space=openmc.stats.Point((0.0, 0.0, 0.0)),
    angle=openmc.stats.Isotropic(),
)

base_sources = [
    FixedSourceComponent(
        source=external_source,
        rate=1.0e5,  # 100,000 n/s from external source
        name="startup_source",
    ),
]

# =============================================================================
# Create Operator
# =============================================================================

op = AlphaNFixedSourceOperator(
    model=model,
    # chain_file="chain_simple.xml",  # Uncomment and provide path
    alpha_n_sources=alpha_n_sources,
    base_sources=base_sources,
    recompute_alpha_n_each_step=True,
    diagnostics_file="alpha_n_diagnostics.json",
)

# =============================================================================
# Run Depletion
# =============================================================================

# Define timesteps and source rates
# Source rates represent the total combined source rate for normalization
timesteps = [30.0, 30.0, 30.0]  # 3 steps of 30 days each
source_rates = [1.0e6, 1.0e6, 1.0e6]  # Total source rate [n/s]

integrator = PredictorIntegrator(
    op,
    timesteps=timesteps,
    source_rates=source_rates,
    timestep_units='d',
)

# Uncomment to actually run (requires nuclear data and alphanso):
# integrator.integrate()

# =============================================================================
# Post-processing: Inspect Diagnostics
# =============================================================================

# After running, inspect how (α,n) sources evolved:
# for diag in op.alpha_n_diagnostics:
#     print(f"Step {diag['step']}:")
#     print(f"  Total (α,n) rate: {diag['total_alpha_n_rate_n_per_s']:.3e} n/s")
#     print(f"  Base source rate: {diag['total_base_source_rate_n_per_s']:.3e} n/s")
#     for mat_id, info in diag['per_material'].items():
#         print(f"  Material {mat_id}: {info['rate_n_per_s']:.3e} n/s "
#               f"(yield: {info['neutron_yield_n_per_s_per_g']:.3e} n/s/g)")

print("Example script completed successfully.")
print("To run the full depletion, uncomment integrator.integrate() above")
print("and ensure nuclear data and alphanso are available.")
