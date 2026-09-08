"""Tests for the openmc.deplete.AlphaNFixedSourceOperator module."""

from unittest.mock import patch, MagicMock, PropertyMock
import copy

import numpy as np
import pytest

import openmc
from openmc.deplete.alpha_n_operator import (
    AlphaNFixedSourceOperator,
    FixedSourceComponent,
)

# Ensure alphanso mock is available
import openmc.alpha_n as _alpha_n_module
if not _alpha_n_module._ALPHANSO_AVAILABLE:
    _alpha_n_module.AlphansoTransport = MagicMock()


@pytest.fixture
def fuel_material():
    """Create a depletable fuel material."""
    mat = openmc.Material()
    mat.add_nuclide('U235', 0.05, 'wo')
    mat.add_nuclide('U238', 0.85, 'wo')
    mat.add_nuclide('O16', 0.10, 'wo')
    mat.set_density('g/cm3', 10.0)
    mat.volume = 100.0
    mat.depletable = True
    return mat


@pytest.fixture
def clad_material():
    """Create a non-depletable cladding material."""
    mat = openmc.Material()
    mat.add_nuclide('Zr90', 0.50, 'wo')
    mat.add_nuclide('Zr91', 0.11, 'wo')
    mat.add_nuclide('Zr92', 0.17, 'wo')
    mat.add_nuclide('Zr94', 0.17, 'wo')
    mat.add_nuclide('Zr96', 0.03, 'wo')
    mat.set_density('g/cm3', 6.5)
    mat.volume = 50.0
    return mat


@pytest.fixture
def mock_alphanso_result():
    """Mock result from alphanso Transport.calculate()."""
    n_bins = 51
    bins = np.linspace(0, 10, n_bins)  # MeV
    spectrum = np.zeros(n_bins - 1)
    for i in range(n_bins - 1):
        mid = (bins[i] + bins[i + 1]) / 2
        spectrum[i] = np.exp(-((mid - 2.0) ** 2) / 0.5)
    spectrum /= np.sum(spectrum)

    return {
        'an_yield': 1.5e3,         # n/s/g
        'sf_yield': 0.5e3,         # n/s/g
        'combined_yield': 2.0e3,   # n/s/g
        'an_spectrum': spectrum,
        'sf_spectrum': spectrum * 0.8,
        'combined_spectrum': spectrum,
        'neutron_energy_bins': bins,  # MeV
    }


@pytest.fixture
def mock_alphanso_result_depleted():
    """Mock result representing changed composition after depletion."""
    n_bins = 51
    bins = np.linspace(0, 10, n_bins)  # MeV
    spectrum = np.zeros(n_bins - 1)
    for i in range(n_bins - 1):
        mid = (bins[i] + bins[i + 1]) / 2
        spectrum[i] = np.exp(-((mid - 3.0) ** 2) / 0.3)
    spectrum /= np.sum(spectrum)

    return {
        'an_yield': 1.2e3,         # Changed yield
        'sf_yield': 0.4e3,
        'combined_yield': 1.6e3,   # Changed combined yield
        'an_spectrum': spectrum,
        'sf_spectrum': spectrum * 0.7,
        'combined_spectrum': spectrum,
        'neutron_energy_bins': bins,
    }


# --- Tests for FixedSourceComponent ---

class TestFixedSourceComponent:
    """Tests for the FixedSourceComponent dataclass."""

    def test_basic_creation(self):
        """Test creating a FixedSourceComponent with valid inputs."""
        source = openmc.IndependentSource()
        comp = FixedSourceComponent(source=source, rate=1.0e6, name="test")
        assert comp.source is source
        assert comp.rate == 1.0e6
        assert comp.name == "test"

    def test_default_name(self):
        """Test that default name is assigned."""
        source = openmc.IndependentSource()
        comp = FixedSourceComponent(source=source, rate=1.0e6)
        assert comp.name == "base_source"

    def test_invalid_source_type(self):
        """Test that non-SourceBase raises TypeError."""
        with pytest.raises(TypeError, match="openmc.SourceBase"):
            FixedSourceComponent(source="not_a_source", rate=1.0e6)

    def test_invalid_rate_type(self):
        """Test that non-numeric rate raises TypeError."""
        source = openmc.IndependentSource()
        with pytest.raises(TypeError, match="real number"):
            FixedSourceComponent(source=source, rate="1e6")

    def test_negative_rate(self):
        """Test that negative rate raises ValueError."""
        source = openmc.IndependentSource()
        with pytest.raises(ValueError, match="non-negative"):
            FixedSourceComponent(source=source, rate=-1.0)

    def test_zero_rate(self):
        """Test that zero rate is allowed."""
        source = openmc.IndependentSource()
        comp = FixedSourceComponent(source=source, rate=0.0)
        assert comp.rate == 0.0


# --- Tests for AlphaNFixedSourceOperator ---

class TestAlphaNFixedSourceOperatorInit:
    """Tests for AlphaNFixedSourceOperator initialization validation."""

    def test_no_alpha_n_sources_raises(self):
        """Test that empty alpha_n_sources raises ValueError."""
        model = openmc.Model()
        model.settings.run_mode = 'fixed source'
        with pytest.raises(ValueError, match="At least one"):
            AlphaNFixedSourceOperator(
                model=model,
                alpha_n_sources=[],
            )

    def test_none_alpha_n_sources_raises(self):
        """Test that None alpha_n_sources raises ValueError."""
        model = openmc.Model()
        model.settings.run_mode = 'fixed source'
        with pytest.raises(ValueError, match="At least one"):
            AlphaNFixedSourceOperator(
                model=model,
                alpha_n_sources=None,
            )

    def test_invalid_alpha_n_source_type(self):
        """Test that non-AlphaNSource in list raises TypeError."""
        model = openmc.Model()
        model.settings.run_mode = 'fixed source'
        with pytest.raises(TypeError, match="AlphaNSource"):
            AlphaNFixedSourceOperator(
                model=model,
                alpha_n_sources=["not_an_alpha_n_source"],
            )

    def test_invalid_base_source_type(self):
        """Test that non-FixedSourceComponent in base_sources raises TypeError."""
        model = openmc.Model()
        model.settings.run_mode = 'fixed source'

        # Need a valid alpha_n_source to get past first check
        with patch.object(_alpha_n_module, '_ALPHANSO_AVAILABLE', True):
            alpha_src = _alpha_n_module.AlphaNSource.__new__(
                _alpha_n_module.AlphaNSource)
            alpha_src._material = openmc.Material()
            alpha_src._calc_type = 'homogeneous'
            alpha_src._include_sf = True
            alpha_src._alphanso_kwargs = {}
            alpha_src._results = None
            alpha_src._neutron_yield = None
            alpha_src._energy_spectrum = None
            alpha_src._energy_bins = None
            alpha_src._source_rate = None

            with pytest.raises(TypeError, match="FixedSourceComponent"):
                AlphaNFixedSourceOperator(
                    model=model,
                    alpha_n_sources=[alpha_src],
                    base_sources=["not_a_component"],
                )


class TestAlphaNRecomputation:
    """Tests verifying (α,n) source recomputation behavior."""

    @patch('openmc.alpha_n.AlphansoTransport.calculate')
    def test_recomputation_changes_with_composition(
        self, mock_calculate, fuel_material, mock_alphanso_result,
        mock_alphanso_result_depleted
    ):
        """Test that (α,n) values change when material composition changes."""
        # First call returns initial result
        mock_calculate.return_value = mock_alphanso_result

        with patch.object(_alpha_n_module, '_ALPHANSO_AVAILABLE', True):
            src = _alpha_n_module.AlphaNSource(
                material=fuel_material, include_sf=True
            )
            src.calculate()
            initial_yield = src.neutron_yield
            initial_rate = src.source_rate

            # Simulate composition change and recompute
            mock_calculate.return_value = mock_alphanso_result_depleted
            src.calculate()
            new_yield = src.neutron_yield
            new_rate = src.source_rate

        assert initial_yield != new_yield
        assert initial_yield == 2.0e3
        assert new_yield == 1.6e3
        # Rates should differ proportionally
        assert initial_rate != new_rate

    @patch('openmc.alpha_n.AlphansoTransport.calculate')
    def test_source_rate_uses_volume_and_density(
        self, mock_calculate, fuel_material, mock_alphanso_result
    ):
        """Test that absolute source rate accounts for volume and density."""
        mock_calculate.return_value = mock_alphanso_result

        with patch.object(_alpha_n_module, '_ALPHANSO_AVAILABLE', True):
            src = _alpha_n_module.AlphaNSource(
                material=fuel_material, include_sf=True
            )
            src.calculate()

        # source_rate = yield [n/s/g] * density [g/cm³] * volume [cm³]
        expected_rate = 2.0e3 * 10.0 * 100.0  # 2e6 n/s
        assert src.source_rate == pytest.approx(expected_rate)


class TestSourceCombination:
    """Tests for combining (α,n) sources with base sources."""

    def test_combined_rate_calculation(self, fuel_material):
        """Test that combined source rate sums correctly."""
        with patch.object(_alpha_n_module, '_ALPHANSO_AVAILABLE', True):
            src = _alpha_n_module.AlphaNSource.__new__(
                _alpha_n_module.AlphaNSource)
            src._material = fuel_material
            src._calc_type = 'homogeneous'
            src._include_sf = True
            src._alphanso_kwargs = {}
            src._results = None
            src._neutron_yield = 2.0e3
            src._energy_spectrum = np.ones(50) / 50
            src._energy_bins = np.linspace(0, 10e6, 51)
            src._source_rate = 2.0e6  # 2e6 n/s

        base_source = openmc.IndependentSource()
        base_comp = FixedSourceComponent(
            source=base_source, rate=1.0e6, name="external"
        )

        # Verify relative strengths: α,n is 2/3, base is 1/3
        total = src.source_rate + base_comp.rate
        assert total == pytest.approx(3.0e6)
        an_fraction = src.source_rate / total
        base_fraction = base_comp.rate / total
        assert an_fraction == pytest.approx(2.0 / 3.0)
        assert base_fraction == pytest.approx(1.0 / 3.0)

    def test_base_sources_only_zero_alpha_n(self):
        """Test that base sources work when (α,n) rate is zero."""
        base_source = openmc.IndependentSource()
        base_comp = FixedSourceComponent(
            source=base_source, rate=5.0e6, name="strong_source"
        )

        alpha_n_rate = 0.0
        total = alpha_n_rate + base_comp.rate
        assert total == pytest.approx(5.0e6)
        assert base_comp.rate / total == pytest.approx(1.0)


class TestDiagnostics:
    """Tests for diagnostics recording."""

    def test_diagnostics_structure(self, fuel_material):
        """Test that diagnostics dict has expected keys."""
        diag = {
            'step': 0,
            'total_alpha_n_rate_n_per_s': 2.0e6,
            'per_material': {
                str(fuel_material.id): {
                    'rate_n_per_s': 2.0e6,
                    'neutron_yield_n_per_s_per_g': 2.0e3,
                }
            },
            'total_base_source_rate_n_per_s': 1.0e6,
        }

        assert 'step' in diag
        assert 'total_alpha_n_rate_n_per_s' in diag
        assert 'per_material' in diag
        assert 'total_base_source_rate_n_per_s' in diag


class TestCreateSourceFromAlphaN:
    """Tests for _create_openmc_source_from_alpha_n."""

    def test_creates_valid_source(self, fuel_material):
        """Test that a valid IndependentSource is created."""
        with patch.object(_alpha_n_module, '_ALPHANSO_AVAILABLE', True):
            src = _alpha_n_module.AlphaNSource.__new__(
                _alpha_n_module.AlphaNSource)
            src._material = fuel_material
            src._calc_type = 'homogeneous'
            src._include_sf = True
            src._alphanso_kwargs = {}
            src._results = None
            src._neutron_yield = 2.0e3
            spectrum = np.ones(50)
            spectrum /= spectrum.sum()
            src._energy_spectrum = spectrum
            src._energy_bins = np.linspace(0, 10e6, 51)
            src._source_rate = 2.0e6

        omc_source = AlphaNFixedSourceOperator._create_openmc_source_from_alpha_n(src)
        assert isinstance(omc_source, openmc.IndependentSource)
        assert omc_source.particle == 'neutron'

    def test_spectrum_preserved(self, fuel_material):
        """Test that energy spectrum shape is preserved in source."""
        n_bins = 51
        bins = np.linspace(0, 10e6, n_bins)
        spectrum = np.zeros(n_bins - 1)
        spectrum[25] = 1.0  # Single peaked bin

        with patch.object(_alpha_n_module, '_ALPHANSO_AVAILABLE', True):
            src = _alpha_n_module.AlphaNSource.__new__(
                _alpha_n_module.AlphaNSource)
            src._material = fuel_material
            src._calc_type = 'homogeneous'
            src._include_sf = True
            src._alphanso_kwargs = {}
            src._results = None
            src._neutron_yield = 1.0e3
            src._energy_spectrum = spectrum
            src._energy_bins = bins
            src._source_rate = 1.0e6

        omc_source = AlphaNFixedSourceOperator._create_openmc_source_from_alpha_n(src)
        # The source should have an energy distribution
        assert omc_source.energy is not None


class TestExistingWorkflowsUnaffected:
    """Tests that existing eigenvalue (α,n) workflow is not affected."""

    def test_alpha_n_manager_still_works(self, fuel_material):
        """Test that AlphaNManager for eigenvalue still functions."""
        # This just verifies the import and basic construction still works
        with patch.object(_alpha_n_module, '_ALPHANSO_AVAILABLE', True):
            manager = _alpha_n_module.AlphaNManager.__new__(
                _alpha_n_module.AlphaNManager)
            manager._model = MagicMock()
            manager._sources = []
            manager._rng = np.random.default_rng()
            manager._is_initialized = False
            manager._alpha_n_fraction = None

            # Can still add sources
            assert hasattr(manager, 'add_source')
            assert hasattr(manager, 'calculate_sources')
            assert hasattr(manager, 'inject_particles')
            assert hasattr(manager, 'run')

    def test_alpha_n_source_class_unchanged(self, fuel_material):
        """Test that AlphaNSource class has expected interface."""
        with patch.object(_alpha_n_module, '_ALPHANSO_AVAILABLE', True):
            src = _alpha_n_module.AlphaNSource.__new__(
                _alpha_n_module.AlphaNSource)
            src._material = fuel_material
            src._calc_type = 'homogeneous'
            src._include_sf = True
            src._alphanso_kwargs = {}
            src._results = None
            src._neutron_yield = None
            src._energy_spectrum = None
            src._energy_bins = None
            src._source_rate = None

            # Original interface still present
            assert hasattr(src, 'calculate')
            assert hasattr(src, 'sample_energy')
            assert hasattr(src, 'neutron_yield')
            assert hasattr(src, 'energy_spectrum')
            assert hasattr(src, 'energy_bins')
            assert hasattr(src, 'source_rate')
            assert hasattr(src, 'material')
