"""Fixed-source depletion operator with timestep-wise (α,n) recomputation.

This module provides a depletion operator for fixed-source problems where
(α,n) neutron sources are recomputed from updated material compositions at
each depletion timestep. It supports combining recomputed (α,n) sources with
user-provided fixed/external neutron sources.

"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Optional
from warnings import warn

import numpy as np
from uncertainties import ufloat

import openmc
import openmc.checkvalue as cv
from openmc.mpi import comm
from .abc import OperatorResult
from .coupled_operator import CoupledOperator
from .results import Results

__all__ = ["AlphaNFixedSourceOperator", "FixedSourceComponent"]


@dataclass
class FixedSourceComponent:
    """A fixed neutron source component with an associated absolute rate.

    This class pairs an OpenMC source definition with an absolute neutron
    emission rate, allowing multiple sources to be combined with physically
    meaningful intensities.

    Parameters
    ----------
    source : openmc.SourceBase
        OpenMC source definition (e.g., :class:`openmc.IndependentSource`).
    rate : float
        Absolute source rate in [n/s].
    name : str, optional
        Human-readable name for this source component.

    Attributes
    ----------
    source : openmc.SourceBase
        The OpenMC source object.
    rate : float
        Source rate in neutrons per second.
    name : str
        Name identifier for this component.

    """
    source: openmc.SourceBase
    rate: float
    name: str = "base_source"

    def __post_init__(self):
        if not isinstance(self.source, openmc.SourceBase):
            raise TypeError(
                f"'source' must be an openmc.SourceBase instance, "
                f"got {type(self.source)}"
            )
        if not isinstance(self.rate, Real):
            raise TypeError(
                f"'rate' must be a real number, got {type(self.rate)}"
            )
        if self.rate < 0.0:
            raise ValueError("'rate' must be non-negative")


class AlphaNFixedSourceOperator(CoupledOperator):
    r"""Transport-coupled depletion operator with (α,n) source recomputation.

    This operator runs fixed-source transport simulations where (α,n) neutron
    sources are recomputed from the current depleted material compositions at
    the beginning of each depletion timestep. This enables coupling between
    material depletion and (α,n) source evolution.

    The operator:

    1. Updates material compositions from the current depletion state
    2. Recomputes (α,n) neutron yields and spectra using alphanso
    3. Constructs transport sources combining (α,n) and user-specified sources
    4. Runs fixed-source transport
    5. Returns reaction rates for depletion

    .. versionadded:: 0.15.4

    Parameters
    ----------
    model : openmc.Model
        OpenMC model configured for fixed-source transport
        (``settings.run_mode = 'fixed source'``).
    chain_file : PathLike or Chain, optional
        Path to the depletion chain XML file or instance of
        openmc.deplete.Chain. Defaults to ``openmc.config['chain_file']``.
    alpha_n_sources : list of openmc.alpha_n.AlphaNSource
        (α,n) source objects associated with depletable materials. Each
        source will have its neutron yield and spectrum recomputed at the
        beginning of each depletion timestep.
    base_sources : list of FixedSourceComponent, optional
        Additional fixed neutron sources with absolute rates in [n/s].
        These are combined with the recomputed (α,n) sources for transport.
    prev_results : Results, optional
        Results from a previous depletion calculation.
    recompute_alpha_n_each_step : bool, optional
        Whether to recompute (α,n) sources at each depletion step.
        Default is True. If False, sources are computed only once at
        initialization.
    diff_burnable_mats : bool, optional
        Whether to differentiate burnable materials with multiple instances.
    reduce_chain_level : int, optional
        Depth of the search when reducing the depletion chain.
    fission_yield_mode : {"constant", "cutoff", "average"}
        Fission product yield scheme to use. Default: ``"constant"``.
    fission_yield_opts : dict, optional
        Optional arguments for the fission yield helper.
    reaction_rate_mode : {"direct", "flux"}, optional
        How one-group reaction rates are calculated. Default: ``"direct"``.
    reaction_rate_opts : dict, optional
        Keyword arguments for the reaction rate helper.
    diagnostics_file : PathLike, optional
        Path to write (α,n) diagnostics (source rates per step).
        If None, diagnostics are stored in memory only.

    Attributes
    ----------
    model : openmc.Model
        The OpenMC model.
    alpha_n_sources : list of openmc.alpha_n.AlphaNSource
        Configured (α,n) source objects.
    base_sources : list of FixedSourceComponent
        User-provided fixed source components.
    alpha_n_diagnostics : list of dict
        Per-step diagnostics including total and per-material (α,n) rates.
    total_alpha_n_rate : float
        Current total (α,n) source rate in [n/s].

    Examples
    --------
    Basic usage with a UO2 material:

    >>> import openmc
    >>> from openmc.alpha_n import AlphaNSource
    >>> from openmc.deplete import AlphaNFixedSourceOperator, FixedSourceComponent
    >>> fuel = openmc.Material()
    >>> fuel.add_nuclide('U235', 0.05, 'wo')
    >>> fuel.add_nuclide('U238', 0.85, 'wo')
    >>> fuel.add_nuclide('O16', 0.10, 'wo')
    >>> fuel.set_density('g/cm3', 10.0)
    >>> fuel.volume = 100.0
    >>> fuel.depletable = True
    >>> # ... build geometry, settings with run_mode='fixed source'
    >>> alpha_sources = [AlphaNSource(material=fuel)]
    >>> op = AlphaNFixedSourceOperator(
    ...     model=model,
    ...     chain_file="chain.xml",
    ...     alpha_n_sources=alpha_sources,
    ... )

    """

    def __init__(
        self,
        model: openmc.Model,
        chain_file=None,
        alpha_n_sources: list | None = None,
        base_sources: list[FixedSourceComponent] | None = None,
        prev_results: Results | None = None,
        recompute_alpha_n_each_step: bool = True,
        diff_burnable_mats: bool = False,
        reduce_chain_level: int | None = None,
        fission_yield_mode: str = "constant",
        fission_yield_opts: dict | None = None,
        reaction_rate_mode: str = "direct",
        reaction_rate_opts: dict | None = None,
        diagnostics_file: str | Path | None = None,
    ):
        from openmc.alpha_n import AlphaNSource

        # Validate inputs
        if alpha_n_sources is None or len(alpha_n_sources) == 0:
            raise ValueError(
                "At least one AlphaNSource must be provided in 'alpha_n_sources'."
            )
        cv.check_type('alpha_n_sources', alpha_n_sources, Iterable)
        for i, src in enumerate(alpha_n_sources):
            if not isinstance(src, AlphaNSource):
                raise TypeError(
                    f"alpha_n_sources[{i}] must be an AlphaNSource instance, "
                    f"got {type(src)}"
                )

        if base_sources is not None:
            cv.check_type('base_sources', base_sources, Iterable)
            for i, bsrc in enumerate(base_sources):
                if not isinstance(bsrc, FixedSourceComponent):
                    raise TypeError(
                        f"base_sources[{i}] must be a FixedSourceComponent "
                        f"instance, got {type(bsrc)}"
                    )

        self._alpha_n_sources = list(alpha_n_sources)
        self._base_sources = list(base_sources) if base_sources else []
        self._recompute_each_step = recompute_alpha_n_each_step
        self._alpha_n_diagnostics = []
        self._total_alpha_n_rate = 0.0
        self._diagnostics_file = Path(diagnostics_file) if diagnostics_file else None
        self._alpha_n_computed = False

        # Ensure model is configured for fixed-source
        if model.settings.run_mode != 'fixed source':
            warn(
                "Model run_mode is not 'fixed source'. Setting it to "
                "'fixed source' for AlphaNFixedSourceOperator.",
                stacklevel=2
            )
            model.settings.run_mode = 'fixed source'

        # Use source-rate normalization for fixed-source
        super().__init__(
            model=model,
            chain_file=chain_file,
            prev_results=prev_results,
            diff_burnable_mats=diff_burnable_mats,
            normalization_mode="source-rate",
            fission_yield_mode=fission_yield_mode,
            fission_yield_opts=fission_yield_opts,
            reaction_rate_mode=reaction_rate_mode,
            reaction_rate_opts=reaction_rate_opts,
            reduce_chain_level=reduce_chain_level,
        )

    @property
    def alpha_n_sources(self):
        """list of AlphaNSource: The configured (α,n) sources."""
        return self._alpha_n_sources

    @property
    def base_sources(self):
        """list of FixedSourceComponent: User-provided fixed sources."""
        return self._base_sources

    @property
    def alpha_n_diagnostics(self):
        """list of dict: Per-step (α,n) diagnostics."""
        return self._alpha_n_diagnostics

    @property
    def total_alpha_n_rate(self):
        """float: Current total (α,n) source rate in [n/s]."""
        return self._total_alpha_n_rate

    def _recompute_alpha_n_sources(self):
        """Recompute (α,n) yields and spectra from current material state.

        This updates each AlphaNSource using the current material compositions,
        recalculates neutron yields and spectra, and rebuilds the model sources.

        """
        total_rate = 0.0
        per_material_rates = {}

        for src in self._alpha_n_sources:
            # Update material composition from the depleted state
            self._sync_material_from_number_densities(src.material)

            # Recompute (α,n) calculation
            src.calculate()

            rate = src.source_rate if src.source_rate is not None else 0.0
            total_rate += rate
            per_material_rates[str(src.material.id)] = {
                'rate_n_per_s': rate,
                'neutron_yield_n_per_s_per_g': src.neutron_yield,
            }

        self._total_alpha_n_rate = total_rate

        # Store diagnostics
        diag = {
            'step': len(self._alpha_n_diagnostics),
            'total_alpha_n_rate_n_per_s': total_rate,
            'per_material': per_material_rates,
            'total_base_source_rate_n_per_s': sum(
                bs.rate for bs in self._base_sources
            ),
        }
        self._alpha_n_diagnostics.append(diag)

        # Rebuild model sources
        self._rebuild_model_sources()
        self._alpha_n_computed = True

    def _sync_material_from_number_densities(self, material):
        """Update a material's nuclide densities from the current depletion state.

        Parameters
        ----------
        material : openmc.Material
            Material to update. Must be in the operator's burnable materials.

        """
        mat_id = str(material.id)
        if mat_id not in self.burnable_mats:
            # Material is not being depleted; use current composition as-is
            return

        mat_index = self.burnable_mats.index(mat_id)
        # Check if this is a local material
        if mat_id not in self.local_mats:
            return

        # Get current number densities and update material
        nuclides = []
        densities = []
        for nuc in self.number.nuclides:
            val = 1.0e-24 * self.number.get_atom_density(mat_id, nuc)
            if val > 0.0:
                nuclides.append(nuc)
                densities.append(val)

        # Clear and rebuild material nuclides
        material._nuclides.clear()
        for nuc, dens in zip(nuclides, densities):
            material.add_nuclide(nuc, dens)

    def _rebuild_model_sources(self):
        """Rebuild the model's source definitions from (α,n) and base sources.

        Combines (α,n) sources and base sources into a list of OpenMC source
        objects with strengths proportional to their absolute rates.

        """
        sources = []
        total_rate = self._total_alpha_n_rate + sum(
            bs.rate for bs in self._base_sources
        )

        if total_rate <= 0.0:
            warn(
                "Total source rate is zero. No neutron sources available "
                "for transport.",
                stacklevel=2
            )
            return

        # Add (α,n) sources
        for src in self._alpha_n_sources:
            rate = src.source_rate if src.source_rate is not None else 0.0
            if rate <= 0.0 or src.energy_spectrum is None:
                continue

            # Create an IndependentSource with the (α,n) energy spectrum
            an_source = self._create_openmc_source_from_alpha_n(src)
            # Set strength as fraction of total rate
            an_source.strength = rate / total_rate
            sources.append(an_source)

        # Add base sources
        for bsrc in self._base_sources:
            source_copy = copy.deepcopy(bsrc.source)
            source_copy.strength = bsrc.rate / total_rate
            sources.append(source_copy)

        if sources:
            self.model.settings.source = sources

    @staticmethod
    def _create_openmc_source_from_alpha_n(alpha_n_src):
        """Create an OpenMC IndependentSource from an AlphaNSource.

        Parameters
        ----------
        alpha_n_src : openmc.alpha_n.AlphaNSource
            The (α,n) source with computed spectrum.

        Returns
        -------
        openmc.IndependentSource
            Source with tabular energy distribution from (α,n) spectrum.

        """
        bins = alpha_n_src.energy_bins
        spectrum = alpha_n_src.energy_spectrum

        # Create tabular energy distribution
        # Use bin midpoints and spectrum values as a histogram
        bin_midpoints = 0.5 * (bins[:-1] + bins[1:])

        energy_dist = openmc.stats.Tabular(
            bin_midpoints, spectrum, interpolation='histogram'
        )

        # Spatial distribution: uniform in the material volume
        # For simplicity, use the material's bounding box or a point source
        # at the origin. Users should set spatial distributions on their
        # (α,n) materials if needed.
        spatial_dist = openmc.stats.Point()

        source = openmc.IndependentSource(
            space=spatial_dist,
            angle=openmc.stats.Isotropic(),
            energy=energy_dist,
            particle='neutron',
        )

        return source

    def __call__(self, vec, source_rate) -> OperatorResult:
        """Run a fixed-source simulation with recomputed (α,n) sources.

        At each call (beginning of step), this method:

        1. Updates material compositions from the depletion vector
        2. Recomputes (α,n) sources (if configured)
        3. Rebuilds model source definitions
        4. Runs fixed-source transport
        5. Returns reaction rates

        Parameters
        ----------
        vec : list of numpy.ndarray
            Total atoms to be used in function.
        source_rate : float
            Source rate in [neutron/sec]. This is the total source rate
            used for normalization. If the (α,n) and base sources have
            absolute rates, those are used for relative source composition,
            and this parameter normalizes the reaction rates.

        Returns
        -------
        openmc.deplete.OperatorResult
            Eigenvalue (always 0 for fixed source) and reaction rates.

        """
        # Reset results in OpenMC
        openmc.lib.reset()

        if self._n_calls > 0:
            openmc.lib.reset_timers()

        self._update_materials_and_nuclides(vec)

        # Recompute (α,n) sources if enabled
        if self._recompute_each_step or not self._alpha_n_computed:
            self._recompute_alpha_n_sources()

        # If the source rate is zero, return zero reaction rates
        if source_rate == 0.0:
            rates = self.reaction_rates.copy()
            rates.fill(0.0)
            return OperatorResult(ufloat(0.0, 0.0), rates)

        # Use the total computed source rate if source_rate is not explicitly
        # provided (i.e., if it equals 1.0 as a sentinel), otherwise use
        # the user-provided rate for normalization
        effective_source_rate = source_rate

        # Run OpenMC fixed-source transport
        openmc.lib.run()

        # Extract results
        rates = self._calculate_reaction_rates(effective_source_rate)

        # Fixed-source: keff is not meaningful, return 0
        keff = ufloat(0.0, 0.0)

        op_result = OperatorResult(keff, rates)
        self._n_calls += 1

        return copy.deepcopy(op_result)

    def get_alpha_n_total_rate(self):
        """Get the total (α,n) source rate computed at the last step.

        Returns
        -------
        float
            Total (α,n) source rate in [n/s].

        """
        return self._total_alpha_n_rate

    def get_combined_source_rate(self):
        """Get the combined source rate from (α,n) and base sources.

        Returns
        -------
        float
            Total source rate in [n/s] from all source components.

        """
        base_total = sum(bs.rate for bs in self._base_sources)
        return self._total_alpha_n_rate + base_total

    def write_diagnostics(self, path: str | Path | None = None):
        """Write (α,n) diagnostics to a JSON file.

        Parameters
        ----------
        path : PathLike, optional
            Path to write the diagnostics file. If None, uses the
            path specified at construction time or defaults to
            'alpha_n_diagnostics.json'.

        """
        if path is None:
            path = self._diagnostics_file or Path('alpha_n_diagnostics.json')
        path = Path(path)

        if comm.rank == 0:
            with open(path, 'w') as f:
                json.dump(self._alpha_n_diagnostics, f, indent=2)

    def finalize(self):
        """Finalize the operator and write diagnostics."""
        if self._diagnostics_file is not None:
            self.write_diagnostics()
        super().finalize()
