.. _usersguide_alpha_n_depletion:

============================================
Fixed-Source Depletion with (α,n) Sources
============================================

OpenMC supports **fixed-source depletion** where (α,n) neutron sources are
**recomputed from updated material compositions at each depletion timestep**.
This feature builds on the existing :ref:`(α,n) source support
<usersguide_alpha_n>` and the depletion framework to enable coupled
depletion/(α,n) workflows.

.. versionadded:: 0.15.4

Background
----------

In systems where material compositions evolve over time (e.g., spent fuel
storage, waste packages), the (α,n) neutron source strength depends on the
current nuclide inventory. As alpha-emitting nuclides decay and build in, the
(α,n) source term changes. The
:class:`~openmc.deplete.AlphaNFixedSourceOperator` enables this coupling by:

1. Running fixed-source transport with the current (α,n) source
2. Depleting materials using the resulting reaction rates
3. Recomputing (α,n) yields from the updated compositions
4. Repeating for all timesteps

This is distinct from the :class:`~openmc.alpha_n.AlphaNManager` eigenvalue
injection workflow, which computes (α,n) once and does not account for
composition changes over time.

Operator Overview
-----------------

The :class:`~openmc.deplete.AlphaNFixedSourceOperator` extends the
:class:`~openmc.deplete.CoupledOperator` for fixed-source mode with these
additions:

- Accepts :class:`~openmc.alpha_n.AlphaNSource` objects tied to specific
  materials
- Recomputes (α,n) neutron yields and spectra at the beginning of each
  depletion step (BOS-only)
- Combines (α,n) sources with user-specified base sources
- Uses ``source-rate`` normalization for fixed-source transport
- Records per-step diagnostics

Source Combination and Normalization
------------------------------------

The operator treats source components with **absolute rates in neutrons per
second**:

- Each ``AlphaNSource`` computes its rate as:
  ``rate [n/s] = yield [n/s/g] × density [g/cm³] × volume [cm³]``
- Each ``FixedSourceComponent`` has an explicit ``rate`` in [n/s]
- The total source rate is the sum of all component rates
- Each component's ``strength`` in the OpenMC model is set proportional to
  its fraction of the total rate

For normalization during depletion, the ``source_rates`` parameter passed to
the integrator represents the total neutron source rate used to scale reaction
rates. Typically this should match the combined rate from all sources.

.. note::
    The recomputation of (α,n) sources occurs at the **beginning of each
    depletion step only** (BOS). For multi-stage integrators (CE/CM, CF4, etc.),
    intermediate transport evaluations within a single timestep reuse the
    BOS (α,n) source.

Basic Usage
-----------

.. code-block:: python

    import openmc
    from openmc.alpha_n import AlphaNSource
    from openmc.deplete import (
        AlphaNFixedSourceOperator,
        FixedSourceComponent,
        PredictorIntegrator,
    )

    # Define depletable material
    fuel = openmc.Material()
    fuel.add_nuclide('U235', 0.05, 'wo')
    fuel.add_nuclide('U238', 0.85, 'wo')
    fuel.add_nuclide('O16', 0.10, 'wo')
    fuel.set_density('g/cm3', 10.0)
    fuel.volume = 100.0  # cm³ - required for absolute source rate
    fuel.depletable = True

    # Build geometry and settings
    # ...
    model = openmc.Model(geometry, materials, settings)
    model.settings.run_mode = 'fixed source'
    model.settings.particles = 10000
    model.settings.batches = 50

    # Define (α,n) sources
    alpha_n_sources = [
        AlphaNSource(material=fuel, include_sf=True),
    ]

    # Optional: additional fixed sources
    external_source = openmc.IndependentSource(
        energy=openmc.stats.Watt(),
    )
    base_sources = [
        FixedSourceComponent(
            source=external_source,
            rate=2.5e6,  # n/s
            name="startup_source",
        ),
    ]

    # Create operator
    op = AlphaNFixedSourceOperator(
        model=model,
        chain_file="chain_simple.xml",
        alpha_n_sources=alpha_n_sources,
        base_sources=base_sources,
    )

    # Run depletion with standard integrator
    integrator = PredictorIntegrator(
        op,
        timesteps=[30.0, 30.0, 30.0],
        source_rates=[3.5e6, 3.5e6, 3.5e6],  # total n/s
        timestep_units='d',
    )
    integrator.integrate()

    # Inspect diagnostics
    for diag in op.alpha_n_diagnostics:
        print(f"Step {diag['step']}: (α,n) rate = "
              f"{diag['total_alpha_n_rate_n_per_s']:.2e} n/s")

    # Optionally write diagnostics to file
    op.write_diagnostics("alpha_n_diagnostics.json")

API Reference
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    openmc.deplete.AlphaNFixedSourceOperator
    openmc.deplete.FixedSourceComponent

Combining with Other Sources
-----------------------------

The ``base_sources`` parameter allows combining (α,n) neutrons with other
fixed/external sources. Each :class:`~openmc.deplete.FixedSourceComponent`
wraps an OpenMC source object with an absolute rate:

.. code-block:: python

    # AmBe source at a fixed rate
    ambe = openmc.IndependentSource(
        energy=openmc.stats.Tabular([...], [...]),
        space=openmc.stats.Point((0, 0, 0)),
    )
    ambe_component = FixedSourceComponent(
        source=ambe,
        rate=1.0e7,  # 10 million n/s
        name="AmBe_source",
    )

The operator distributes transport source particles among all components
proportional to their rates:

- If (α,n) contributes 2×10⁶ n/s and a base source contributes 10⁷ n/s,
  then ~17% of source particles sample from (α,n) and ~83% from the base
  source.

Diagnostics
-----------

The operator records per-step diagnostics accessible via
:attr:`~openmc.deplete.AlphaNFixedSourceOperator.alpha_n_diagnostics`:

.. code-block:: python

    # Each entry is a dict with:
    # {
    #   'step': int,
    #   'total_alpha_n_rate_n_per_s': float,
    #   'per_material': {
    #       '<material_id>': {
    #           'rate_n_per_s': float,
    #           'neutron_yield_n_per_s_per_g': float,
    #       }
    #   },
    #   'total_base_source_rate_n_per_s': float,
    # }

Diagnostics can be written to a JSON file using
:meth:`~openmc.deplete.AlphaNFixedSourceOperator.write_diagnostics`.

Limitations
-----------

Current limitations of this prototype:

- **(α,n) recomputation is BOS-only**: Multi-stage integrators (CE/CM, CF4)
  reuse the beginning-of-step (α,n) source for all intermediate transport
  evaluations within a single depletion timestep.
- **Homogeneous calculation only**: Only the ``'homogeneous'`` calculation type
  is supported in :class:`~openmc.alpha_n.AlphaNSource`.
- **Spatial distribution**: The (α,n) source is created as a point source at
  the origin by default. Users should ensure their geometry accounts for this
  or provide appropriate spatial distributions in their model.
- **No automatic volume determination**: Material volumes must be set
  explicitly.
- **Python-side orchestration**: The (α,n) recomputation is performed in Python
  before each transport step.

Comparison with Eigenvalue (α,n) Injection
-------------------------------------------

+----------------------------------------------+------------------------------------------+
| **Eigenvalue Injection**                     | **Fixed-Source Depletion**                |
| (:class:`~openmc.alpha_n.AlphaNManager`)     | (:class:`~openmc.deplete.               |
|                                              | AlphaNFixedSourceOperator`)              |
+==============================================+==========================================+
| k-eigenvalue simulations                     | Fixed-source simulations                 |
+----------------------------------------------+------------------------------------------+
| (α,n) computed once before simulation        | (α,n) recomputed each depletion step     |
+----------------------------------------------+------------------------------------------+
| Particles injected into source bank          | Sources defined in model settings        |
+----------------------------------------------+------------------------------------------+
| No depletion coupling                        | Full depletion coupling                  |
+----------------------------------------------+------------------------------------------+
| Uses alphanso batch-level injection          | Uses standard fixed-source transport     |
+----------------------------------------------+------------------------------------------+
