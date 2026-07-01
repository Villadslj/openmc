.. _usersguide_alpha_n:

==================================
(α,n) Neutron Sources
==================================

OpenMC can include neutrons from (α,n) reactions in k-eigenvalue simulations
using the `ALPHANSO <https://github.com/alphanso-org/alphanso>`_ Python package.
This feature computes the (α,n) neutron yield and energy spectrum from material
compositions containing alpha-emitting nuclides and injects the resulting
neutrons into the simulation's neutron bank at each batch.

.. note::
    The ``alphanso`` package is an optional dependency. Install it with::

        pip install openmc[alpha_n]

    Or install ``alphanso`` directly::

        pip install alphanso

Background
----------

In nuclear systems containing alpha-emitting nuclides (e.g., actinides like
U-235, U-238, Pu-238, Am-241) alongside light target nuclides (e.g., O-17,
O-18, Be-9, C-13), alpha particles from radioactive decay can interact with
target nuclei to produce neutrons via (α,n) reactions. These neutrons can
represent a significant source term, especially in:

- Fresh or spent nuclear fuel (from alpha decay of actinides interacting with
  oxygen in UO₂)
- Plutonium-beryllium (PuBe) or americium-beryllium (AmBe) neutron sources
- Nuclear waste storage and transportation scenarios

The ``openmc.alpha_n`` module provides a way to account for these (α,n)
neutrons alongside the fission neutrons in k-eigenvalue calculations.

Basic Usage
-----------

The simplest way to include (α,n) neutrons is through the
:class:`~openmc.alpha_n.AlphaNManager` class:

.. code-block:: python

    import openmc
    from openmc.alpha_n import AlphaNManager

    # Define materials
    uo2 = openmc.Material()
    uo2.add_nuclide('U235', 0.05, 'wo')
    uo2.add_nuclide('U238', 0.85, 'wo')
    uo2.add_nuclide('O16', 0.10, 'wo')
    uo2.set_density('g/cm3', 10.0)
    uo2.volume = 100.0  # cm³ — required for absolute source rate

    # Build your model as usual
    model = openmc.Model(geometry, materials, settings)

    # Create manager and add (α,n) source
    manager = AlphaNManager(model)
    manager.add_source(uo2)

    # Set the fraction of particles per batch from (α,n)
    manager.set_alpha_n_fraction(0.01)  # 1% of batch particles

    # Run the simulation with (α,n) injection
    k_eff = manager.run()

Material Requirements
---------------------

For (α,n) calculations, the material must have:

1. **Nuclides defined** — with mass fractions (``'wo'``) or atom fractions
   (``'ao'``).
2. **Density set** — needed to compute the mass of material.
3. **Volume set** — needed to compute the absolute source rate (n/s).

The nuclide composition is automatically converted to ALPHANSO's format
(ZAID integers with mass fractions).

Controlling Source Strength
---------------------------

The :meth:`~openmc.alpha_n.AlphaNManager.set_alpha_n_fraction` method controls
what fraction of the batch's particle count is contributed by (α,n) neutrons:

.. code-block:: python

    # 5% of batch particles will come from (α,n)
    manager.set_alpha_n_fraction(0.05)

This fraction should be chosen based on the expected ratio of (α,n) source
strength to fission source strength in the system. For most reactor problems,
(α,n) contributions are small (< 1%), while for subcritical systems with
strong alpha emitters, a larger fraction may be appropriate.

Including Spontaneous Fission
-----------------------------

By default, ALPHANSO also computes spontaneous fission (SF) neutron yields.
These are included in the combined yield and spectrum by default. To exclude
SF neutrons:

.. code-block:: python

    manager.add_source(material, include_sf=False)

Advanced Usage
--------------

For finer control, you can work directly with :class:`~openmc.alpha_n.AlphaNSource`
objects and the batch iteration interface:

.. code-block:: python

    import openmc
    import openmc.lib
    from openmc.alpha_n import AlphaNSource, AlphaNManager

    # Create and calculate the source
    source = AlphaNSource(material=uo2)
    source.calculate()

    # Inspect results
    print(f"Neutron yield: {source.neutron_yield:.3e} n/s/g")
    print(f"Source rate: {source.source_rate:.3e} n/s")

    # Manual batch loop
    model.export_to_model_xml()
    manager = AlphaNManager(model, alpha_n_sources=[source])
    manager.set_alpha_n_fraction(0.01)

    with openmc.lib.run_in_memory():
        openmc.lib.simulation_init()
        try:
            for _ in openmc.lib.iter_batches():
                manager.inject_particles()
        finally:
            openmc.lib.simulation_finalize()

ALPHANSO Configuration
-----------------------

Additional keyword arguments can be passed to ALPHANSO's calculation engine
via the ``alphanso_kwargs`` parameter:

.. code-block:: python

    from openmc.alpha_n import AlphaNSource

    source = AlphaNSource(
        material=uo2,
        alphanso_kwargs={
            'num_alpha_groups': 20000,
            'neutron_energy_bins': [0, 20, 201],  # [start, stop, n_points]
        }
    )

Limitations
-----------

- Only the ``'homogeneous'`` calculation type is currently supported. This
  assumes a uniform mixture of alpha emitters and target nuclides within
  the material.
- Material volumes must be set manually (OpenMC does not compute material
  volumes automatically for arbitrary geometries; see :ref:`usersguide_volume`).
- The (α,n) spectrum is computed once and does not change between batches
  (appropriate for steady-state simulations, but not for depletion without
  recalculation).
- Particle positions are sampled from existing fission source sites, which
  approximates the (α,n) spatial distribution as overlapping with the fission
  source.
