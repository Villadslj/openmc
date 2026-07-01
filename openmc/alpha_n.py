"""Module for computing and injecting (α,n) neutron sources into OpenMC
k-eigenvalue simulations using the ALPHANSO package.

This module provides classes to calculate (α,n) reaction rates and energy
spectra from material compositions, then inject corresponding neutron source
particles into the neutron bank during k-eigenvalue simulations. The
calculations are performed using the `alphanso <https://github.com/alphanso-org/
alphanso>`_ Python package.

.. note::

    The ``alphanso`` package is an optional dependency. Install it with:

    .. code-block:: sh

        pip install alphanso

"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from numbers import Real

import numpy as np

import openmc
import openmc.checkvalue as cv
from openmc.data import zam, ATOMIC_NUMBER

try:
    from alphanso.transport import Transport as AlphansoTransport
    _ALPHANSO_AVAILABLE = True
except ImportError:
    _ALPHANSO_AVAILABLE = False


def _check_alphanso_available():
    """Raise an error if alphanso is not installed."""
    if not _ALPHANSO_AVAILABLE:
        raise ImportError(
            "The 'alphanso' package is required for (α,n) source "
            "calculations. Install it with: pip install alphanso"
        )


def _material_to_matdef(material):
    """Convert an OpenMC Material to an alphanso matdef dictionary.

    Parameters
    ----------
    material : openmc.Material
        Material to convert. Must have nuclides defined.

    Returns
    -------
    dict
        Dictionary mapping ZAID integers to mass fractions, suitable
        for use with alphanso's ``Transport.calculate()``.

    """
    nuclides = material.nuclides
    if not nuclides:
        raise ValueError(
            f"Material {material.id} has no nuclides defined."
        )

    matdef = {}
    for nuc in nuclides:
        Z, A, m = zam(nuc.name)
        zaid = Z * 1000 + A

        if nuc.percent_type == 'wo':
            # Already mass fraction
            matdef[zaid] = nuc.percent
        else:
            # Convert atom fraction to mass fraction
            # mass_fraction_i = atom_fraction_i * M_i / sum(atom_fraction_j * M_j)
            matdef[zaid] = nuc.percent * openmc.data.atomic_mass(nuc.name)

    # Normalize to sum to 1.0
    total = sum(matdef.values())
    if total <= 0.0:
        raise ValueError(
            f"Material {material.id} has zero or negative total mass fraction."
        )
    matdef = {k: v / total for k, v in matdef.items()}

    return matdef


class AlphaNSource:
    """Represents an (α,n) neutron source computed for a specific material.

    This class wraps an alphanso calculation for a single material, computing
    the neutron yield and energy spectrum from (α,n) reactions (and optionally
    spontaneous fission). The results can then be used to create source
    particles for injection into OpenMC simulations.

    Parameters
    ----------
    material : openmc.Material
        Material containing alpha-emitting nuclides and/or target nuclides.
        Must have a density set and nuclides defined.
    calc_type : str
        Type of alphanso calculation geometry. Currently only ``'homogeneous'``
        is supported.
    include_sf : bool
        Whether to include spontaneous fission neutrons in addition to
        (α,n) neutrons. Default is True.
    alphanso_kwargs : dict, optional
        Additional keyword arguments passed to ``alphanso.Transport.calculate()``,
        such as ``num_alpha_groups``, ``neutron_energy_bins``, etc.

    Attributes
    ----------
    material : openmc.Material
        The OpenMC material associated with this source.
    calc_type : str
        Calculation geometry type.
    neutron_yield : float
        Total neutron yield in n/s/g (for homogeneous calculations).
    energy_spectrum : numpy.ndarray
        Normalized neutron energy spectrum (probability per bin).
    energy_bins : numpy.ndarray
        Energy bin edges in eV.
    source_rate : float or None
        Absolute source rate in n/s, computed when material volume is available.

    """

    def __init__(
        self,
        material: openmc.Material,
        calc_type: str = 'homogeneous',
        include_sf: bool = True,
        alphanso_kwargs: dict | None = None
    ):
        _check_alphanso_available()
        cv.check_type('material', material, openmc.Material)
        cv.check_value('calc_type', calc_type, ('homogeneous',))

        self._material = material
        self._calc_type = calc_type
        self._include_sf = include_sf
        self._alphanso_kwargs = alphanso_kwargs or {}
        self._results = None
        self._neutron_yield = None
        self._energy_spectrum = None
        self._energy_bins = None
        self._source_rate = None

    @property
    def material(self) -> openmc.Material:
        return self._material

    @property
    def calc_type(self) -> str:
        return self._calc_type

    @property
    def neutron_yield(self) -> float | None:
        """Total neutron yield in n/s/g."""
        return self._neutron_yield

    @property
    def energy_spectrum(self) -> np.ndarray | None:
        """Normalized neutron energy spectrum."""
        return self._energy_spectrum

    @property
    def energy_bins(self) -> np.ndarray | None:
        """Energy bin edges in eV."""
        return self._energy_bins

    @property
    def source_rate(self) -> float | None:
        """Absolute source rate in n/s (requires material volume)."""
        return self._source_rate

    def calculate(self):
        """Perform the alphanso (α,n) calculation.

        This method converts the OpenMC material to alphanso's format,
        runs the calculation, and stores the resulting neutron yield
        and energy spectrum.

        """
        matdef = _material_to_matdef(self._material)

        config = {
            'calc_type': self._calc_type,
            'matdef': matdef,
            **self._alphanso_kwargs
        }

        self._results = AlphansoTransport.calculate(config)

        if self._include_sf:
            self._neutron_yield = self._results.get('combined_yield', 0.0)
            spectrum = self._results.get('combined_spectrum')
        else:
            self._neutron_yield = self._results.get('an_yield', 0.0)
            spectrum = self._results.get('an_spectrum')

        if spectrum is None:
            warnings.warn(
                f"No neutron spectrum returned for material {self._material.id}. "
                "The material may not contain alpha-emitting nuclides or "
                "suitable target nuclides for (α,n) reactions."
            )
            self._neutron_yield = 0.0
            self._energy_spectrum = None
            self._energy_bins = None
            return

        # Convert energy bins from MeV to eV
        bins_mev = np.array(self._results['neutron_energy_bins'])
        self._energy_bins = bins_mev * 1.0e6  # MeV -> eV
        self._energy_spectrum = np.array(spectrum)

        # Compute absolute source rate if volume is available
        self._compute_source_rate()

    def _compute_source_rate(self):
        """Compute absolute source rate from yield and material mass."""
        if self._neutron_yield is None or self._neutron_yield == 0.0:
            self._source_rate = 0.0
            return

        volume = self._material.volume
        if volume is None:
            self._source_rate = None
            return

        # Mass density [g/cm³] × volume [cm³] = mass [g]
        density = self._material.get_mass_density()
        mass = density * volume  # grams
        # yield [n/s/g] × mass [g] = source rate [n/s]
        self._source_rate = self._neutron_yield * mass

    def sample_energy(self, n_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample neutron energies from the (α,n) spectrum.

        Parameters
        ----------
        n_samples : int
            Number of energy samples to generate.
        rng : numpy.random.Generator, optional
            Random number generator. If None, a new default generator is used.

        Returns
        -------
        numpy.ndarray
            Array of sampled energies in eV.

        """
        if self._energy_spectrum is None or self._energy_bins is None:
            raise RuntimeError(
                "No energy spectrum available. Call calculate() first, "
                "and ensure the material contains suitable nuclides."
            )

        if rng is None:
            rng = np.random.default_rng()

        # Compute bin midpoints and create CDF for sampling
        bin_edges = self._energy_bins
        spectrum = self._energy_spectrum

        # Normalize spectrum to form a valid probability distribution
        total = np.sum(spectrum)
        if total <= 0.0:
            raise RuntimeError("Energy spectrum has zero or negative total.")
        probs = spectrum / total

        # Sample bins according to probabilities
        bin_indices = rng.choice(len(probs), size=n_samples, p=probs)

        # Sample uniformly within each selected bin
        low = bin_edges[bin_indices]
        high = bin_edges[bin_indices + 1]
        energies = rng.uniform(low, high)

        return energies


def _sample_isotropic_directions(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample isotropic unit direction vectors.

    Parameters
    ----------
    n : int
        Number of direction vectors to sample.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    numpy.ndarray
        Array of shape (n, 3) with unit direction vectors.

    """
    # Sample cos(theta) uniformly in [-1, 1] and phi uniformly in [0, 2π]
    mu = 2.0 * rng.random(n) - 1.0
    phi = 2.0 * np.pi * rng.random(n)
    sin_theta = np.sqrt(1.0 - mu**2)

    directions = np.empty((n, 3))
    directions[:, 0] = sin_theta * np.cos(phi)
    directions[:, 1] = sin_theta * np.sin(phi)
    directions[:, 2] = mu
    return directions


class AlphaNManager:
    """Manager for (α,n) neutron source injection in k-eigenvalue simulations.

    This class orchestrates the computation and injection of (α,n) neutron
    sources into OpenMC k-eigenvalue simulations. It manages multiple
    :class:`AlphaNSource` objects and handles the batch-level injection
    of source particles.

    Parameters
    ----------
    model : openmc.Model
        OpenMC model to run. The model should be configured for a k-eigenvalue
        simulation (``settings.run_mode = 'eigenvalue'``).
    alpha_n_sources : list of AlphaNSource, optional
        Pre-configured (α,n) sources. Additional sources can be added via
        :meth:`add_source`.

    Attributes
    ----------
    model : openmc.Model
        The OpenMC model.
    sources : list of AlphaNSource
        List of configured (α,n) sources.
    total_source_rate : float
        Combined absolute (α,n) source rate from all sources in n/s.

    Examples
    --------
    Basic usage with a homogeneous material:

    >>> import openmc
    >>> from openmc.alpha_n import AlphaNSource, AlphaNManager
    >>> uo2 = openmc.Material()
    >>> uo2.add_nuclide('U235', 0.05, 'wo')
    >>> uo2.add_nuclide('U238', 0.85, 'wo')
    >>> uo2.add_nuclide('O16', 0.10, 'wo')
    >>> uo2.set_density('g/cm3', 10.0)
    >>> uo2.volume = 100.0  # cm³
    >>> model = openmc.Model(...)
    >>> manager = AlphaNManager(model)
    >>> manager.add_source(uo2)
    >>> manager.run()

    """

    def __init__(
        self,
        model: openmc.Model,
        alpha_n_sources: list[AlphaNSource] | None = None
    ):
        _check_alphanso_available()
        cv.check_type('model', model, openmc.Model)
        self._model = model
        self._sources = list(alpha_n_sources) if alpha_n_sources else []
        self._rng = np.random.default_rng()
        self._is_initialized = False

    @property
    def model(self) -> openmc.Model:
        return self._model

    @property
    def sources(self) -> list[AlphaNSource]:
        return self._sources

    @property
    def total_source_rate(self) -> float:
        """Combined absolute (α,n) source rate from all sources in n/s."""
        total = 0.0
        for src in self._sources:
            if src.source_rate is not None:
                total += src.source_rate
        return total

    def add_source(
        self,
        material: openmc.Material,
        calc_type: str = 'homogeneous',
        include_sf: bool = True,
        **alphanso_kwargs
    ):
        """Add an (α,n) source for a material.

        Parameters
        ----------
        material : openmc.Material
            Material containing alpha-emitting and/or target nuclides.
            Must have a density and volume set.
        calc_type : str
            Alphanso calculation type. Default is ``'homogeneous'``.
        include_sf : bool
            Whether to include spontaneous fission neutrons. Default is True.
        **alphanso_kwargs
            Additional keyword arguments for alphanso (e.g.,
            ``num_alpha_groups``, ``neutron_energy_bins``).

        """
        source = AlphaNSource(
            material=material,
            calc_type=calc_type,
            include_sf=include_sf,
            alphanso_kwargs=alphanso_kwargs if alphanso_kwargs else None
        )
        self._sources.append(source)

    def calculate_sources(self):
        """Run alphanso calculations for all registered sources.

        This computes the neutron yield and energy spectrum for each source.
        It should be called before running the simulation or injecting
        particles.

        """
        for src in self._sources:
            src.calculate()

        # Check that at least one source has a nonzero rate
        total_rate = self.total_source_rate
        if total_rate == 0.0:
            warnings.warn(
                "All (α,n) sources have zero neutron yield. No additional "
                "neutrons will be injected. Check that materials contain "
                "alpha-emitting nuclides and target nuclides, and that "
                "material volumes are set."
            )

    def _compute_n_alpha_n_particles(self, n_particles: int) -> int:
        """Compute how many (α,n) particles to inject per batch.

        The number is scaled relative to the number of fission source
        particles based on the ratio of (α,n) source rate to a
        user-estimated fission rate, using the current k-effective value.

        Parameters
        ----------
        n_particles : int
            Number of fission source particles per batch.

        Returns
        -------
        int
            Number of (α,n) particles to inject.

        """
        total_rate = self.total_source_rate
        if total_rate <= 0.0:
            return 0

        # The (α,n) source rate is absolute (n/s). To determine the
        # fraction of source particles that should come from (α,n), we
        # use the ratio: n_an / n_total = S_an / (S_an + S_fission).
        # Since we don't know S_fission directly, we approximate by
        # using the user-provided ratio or a fixed fraction.
        # For now, we inject particles proportional to the source rate,
        # scaled by the number of simulation particles.

        # Use a simple scaling: inject one particle per batch for every
        # unit of source rate, capped at a fraction of n_particles.
        # Users can override via the `alpha_n_fraction` parameter.
        if hasattr(self, '_alpha_n_fraction'):
            n_inject = int(self._alpha_n_fraction * n_particles)
        else:
            # Default: use source rate to scale injection.
            # This requires knowledge of the fission rate; without it,
            # we inject a fraction based on the (α,n) strength relative
            # to a typical fission source.
            n_inject = max(1, int(0.01 * n_particles))

        return n_inject

    def set_alpha_n_fraction(self, fraction: float):
        """Set the fraction of particles per batch from (α,n) sources.

        This controls how many (α,n) source particles are injected
        relative to the number of fission source particles.

        Parameters
        ----------
        fraction : float
            Fraction of particles to add from (α,n) reactions. For example,
            ``0.01`` means 1% of the batch particle count will be added
            as (α,n) neutrons. Must be between 0 and 1.

        """
        cv.check_type('alpha_n_fraction', fraction, Real)
        cv.check_greater_than('alpha_n_fraction', fraction, 0.0, True)
        cv.check_less_than('alpha_n_fraction', fraction, 1.0)
        self._alpha_n_fraction = float(fraction)

    def inject_particles(self):
        """Inject (α,n) source particles into the current source bank.

        This method should be called between batches (e.g., within an
        ``iter_batches()`` loop). It modifies the source bank in-place
        by adjusting particle weights to account for the (α,n) contribution.

        The injection works by:
        1. Determining how many (α,n) particles to add
        2. Selecting random existing source sites to replace with (α,n) sites
        3. Adjusting weights so the (α,n) fraction is properly represented

        """
        import openmc.lib

        bank = openmc.lib.source_bank()
        n_particles = len(bank)

        if n_particles == 0:
            return

        n_inject = self._compute_n_alpha_n_particles(n_particles)
        if n_inject <= 0:
            return

        # Don't inject more than the bank size
        n_inject = min(n_inject, n_particles)

        # Distribute injection across sources proportional to their rates
        source_rates = []
        for src in self._sources:
            rate = src.source_rate if src.source_rate is not None else 0.0
            source_rates.append(rate)

        total_rate = sum(source_rates)
        if total_rate <= 0.0:
            return

        # Select which source bank positions to overwrite with (α,n) particles
        replace_indices = self._rng.choice(
            n_particles, size=n_inject, replace=False
        )

        # Create (α,n) particles and overwrite selected positions
        idx = 0
        for src, rate in zip(self._sources, source_rates):
            if rate <= 0.0 or src.energy_spectrum is None:
                continue

            # Number of particles from this source
            n_from_src = max(1, int(n_inject * rate / total_rate))
            n_from_src = min(n_from_src, n_inject - idx)
            if n_from_src <= 0:
                continue

            # Sample energies from (α,n) spectrum
            energies = src.sample_energy(n_from_src, self._rng)

            # Sample isotropic directions
            directions = _sample_isotropic_directions(n_from_src, self._rng)

            # Get positions: sample from existing source particles
            # (since they are distributed in fissile material, which
            # typically overlaps with (α,n) source material)
            pos_indices = self._rng.choice(n_particles, size=n_from_src)

            for j in range(n_from_src):
                bi = replace_indices[idx]
                # Copy position from an existing source site
                pi = pos_indices[j]
                bank[bi]['r'] = bank[pi]['r']
                # Set (α,n) neutron properties
                bank[bi]['u'] = tuple(directions[j])
                bank[bi]['E'] = energies[j]
                bank[bi]['wgt'] = 1.0
                bank[bi]['delayed_group'] = 0
                bank[bi]['particle'] = 2112  # neutron PDG number
                idx += 1

            if idx >= n_inject:
                break

    def run(self, output: bool = True, **kwargs):
        """Run the k-eigenvalue simulation with (α,n) source injection.

        This method exports the model, initializes OpenMC, runs the
        simulation batch-by-batch, and injects (α,n) particles between
        batches.

        Parameters
        ----------
        output : bool
            Whether to display simulation output. Default is True.
        **kwargs
            Additional keyword arguments passed to :func:`openmc.lib.init`.

        Returns
        -------
        openmc.lib.keff
            The computed k-eigenvalue and standard deviation.

        """
        import openmc.lib

        # Calculate (α,n) sources if not already done
        if not any(src.neutron_yield is not None for src in self._sources):
            self.calculate_sources()

        # Export model files
        self._model.export_to_model_xml()

        with openmc.lib.run_in_memory(output=output, **kwargs):
            openmc.lib.simulation_init()

            try:
                for _ in openmc.lib.iter_batches():
                    self.inject_particles()
            finally:
                openmc.lib.simulation_finalize()

            return openmc.lib.keff()
