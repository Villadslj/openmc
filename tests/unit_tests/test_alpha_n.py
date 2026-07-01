"""Tests for the openmc.alpha_n module."""

from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import pytest

import openmc
from openmc.alpha_n import (
    _material_to_matdef,
    _sample_isotropic_directions,
    AlphaNSource,
    AlphaNManager,
)

# Create a module-level mock for AlphansoTransport that can be patched
# even when alphanso is not installed
import openmc.alpha_n as _alpha_n_module
if not _alpha_n_module._ALPHANSO_AVAILABLE:
    _alpha_n_module.AlphansoTransport = MagicMock()


@pytest.fixture
def uo2_material():
    """Create a UO2 material with density and volume."""
    mat = openmc.Material()
    mat.add_nuclide('U235', 0.05, 'wo')
    mat.add_nuclide('U238', 0.85, 'wo')
    mat.add_nuclide('O16', 0.10, 'wo')
    mat.set_density('g/cm3', 10.0)
    mat.volume = 100.0  # cm³
    return mat


@pytest.fixture
def uo2_material_ao():
    """Create a UO2 material with atom fractions."""
    mat = openmc.Material()
    mat.add_nuclide('U235', 0.02, 'ao')
    mat.add_nuclide('U238', 0.31, 'ao')
    mat.add_nuclide('O16', 0.67, 'ao')
    mat.set_density('g/cm3', 10.0)
    mat.volume = 50.0
    return mat


@pytest.fixture
def mock_alphanso_result():
    """Mock result from alphanso Transport.calculate()."""
    n_bins = 101
    bins = np.linspace(0, 15, n_bins)  # MeV
    spectrum = np.zeros(n_bins - 1)
    # Create a simple peaked spectrum around 2 MeV
    for i in range(n_bins - 1):
        mid = (bins[i] + bins[i + 1]) / 2
        spectrum[i] = np.exp(-((mid - 2.0) ** 2) / 0.5)
    spectrum /= np.sum(spectrum)

    return {
        'an_yield': 1.5e3,         # n/s/g
        'sf_yield': 5.0e2,         # n/s/g
        'combined_yield': 2.0e3,   # n/s/g
        'an_spectrum': spectrum * 0.75,
        'sf_spectrum': spectrum * 0.25,
        'combined_spectrum': spectrum,
        'neutron_energy_bins': bins,
    }


class TestMaterialToMatdef:
    """Tests for _material_to_matdef conversion."""

    def test_weight_fractions(self, uo2_material):
        """Test conversion with weight fractions."""
        matdef = _material_to_matdef(uo2_material)

        # Check that ZAIDs are correct
        assert 92235 in matdef  # U-235
        assert 92238 in matdef  # U-238
        assert 8016 in matdef   # O-16

        # Check normalization
        assert pytest.approx(sum(matdef.values()), rel=1e-10) == 1.0

        # Check relative fractions match input
        assert pytest.approx(matdef[92235], rel=1e-10) == 0.05
        assert pytest.approx(matdef[92238], rel=1e-10) == 0.85
        assert pytest.approx(matdef[8016], rel=1e-10) == 0.10

    def test_atom_fractions(self, uo2_material_ao):
        """Test conversion with atom fractions (should convert to mass)."""
        matdef = _material_to_matdef(uo2_material_ao)

        assert 92235 in matdef
        assert 92238 in matdef
        assert 8016 in matdef

        # Should sum to 1.0 after normalization
        assert pytest.approx(sum(matdef.values()), rel=1e-10) == 1.0

        # U-238 should dominate by mass (heavier and more abundant)
        assert matdef[92238] > matdef[92235]
        assert matdef[92238] > matdef[8016]

    def test_empty_material(self):
        """Test that empty material raises ValueError."""
        mat = openmc.Material()
        mat.set_density('g/cm3', 1.0)
        with pytest.raises(ValueError, match="no nuclides"):
            _material_to_matdef(mat)


class TestSampleIsotropicDirections:
    """Tests for isotropic direction sampling."""

    def test_unit_vectors(self):
        """Test that sampled directions are unit vectors."""
        rng = np.random.default_rng(42)
        directions = _sample_isotropic_directions(1000, rng)

        norms = np.linalg.norm(directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-14)

    def test_isotropic_distribution(self):
        """Test that the distribution is approximately isotropic."""
        rng = np.random.default_rng(42)
        directions = _sample_isotropic_directions(100000, rng)

        # Mean of each component should be close to 0
        for i in range(3):
            assert abs(np.mean(directions[:, i])) < 0.01

    def test_shape(self):
        """Test output shape."""
        rng = np.random.default_rng(42)
        directions = _sample_isotropic_directions(50, rng)
        assert directions.shape == (50, 3)


class TestAlphaNSource:
    """Tests for AlphaNSource class."""

    def test_calculate(self, uo2_material, mock_alphanso_result):
        """Test that calculate() properly calls alphanso and stores results."""
        mock_transport = MagicMock()
        mock_transport.calculate.return_value = mock_alphanso_result

        with patch.object(_alpha_n_module, '_ALPHANSO_AVAILABLE', True), \
             patch.object(_alpha_n_module, 'AlphansoTransport', mock_transport):
            src = AlphaNSource(material=uo2_material)
            src.calculate()

        # Check alphanso was called
        mock_transport.calculate.assert_called_once()

        # Check results are stored
        assert src.neutron_yield == 2.0e3
        assert src.energy_spectrum is not None
        assert src.energy_bins is not None

        # Check energy bins converted to eV
        np.testing.assert_allclose(
            src.energy_bins,
            mock_alphanso_result['neutron_energy_bins'] * 1e6
        )

        # Check source rate = yield × density × volume
        expected_rate = 2.0e3 * 10.0 * 100.0  # n/s/g × g/cm³ × cm³
        assert pytest.approx(src.source_rate) == expected_rate

    def test_calculate_without_sf(self, uo2_material, mock_alphanso_result):
        """Test calculation without spontaneous fission."""
        mock_transport = MagicMock()
        mock_transport.calculate.return_value = mock_alphanso_result

        with patch.object(_alpha_n_module, '_ALPHANSO_AVAILABLE', True), \
             patch.object(_alpha_n_module, 'AlphansoTransport', mock_transport):
            src = AlphaNSource(material=uo2_material, include_sf=False)
            src.calculate()

        assert src.neutron_yield == 1.5e3  # Only (α,n) yield

    def test_no_volume(self, mock_alphanso_result):
        """Test that source_rate is None when volume is not set."""
        mock_transport = MagicMock()
        mock_transport.calculate.return_value = mock_alphanso_result

        mat = openmc.Material()
        mat.add_nuclide('U238', 1.0, 'wo')
        mat.set_density('g/cm3', 10.0)
        # No volume set

        with patch.object(_alpha_n_module, '_ALPHANSO_AVAILABLE', True), \
             patch.object(_alpha_n_module, 'AlphansoTransport', mock_transport):
            src = AlphaNSource(material=mat)
            src.calculate()

        assert src.source_rate is None

    def test_sample_energy(self, uo2_material, mock_alphanso_result):
        """Test energy sampling from spectrum."""
        mock_transport = MagicMock()
        mock_transport.calculate.return_value = mock_alphanso_result

        with patch.object(_alpha_n_module, '_ALPHANSO_AVAILABLE', True), \
             patch.object(_alpha_n_module, 'AlphansoTransport', mock_transport):
            src = AlphaNSource(material=uo2_material)
            src.calculate()

        rng = np.random.default_rng(42)
        energies = src.sample_energy(1000, rng)

        assert len(energies) == 1000
        # Energies should be in eV (spectrum peaks around 2 MeV = 2e6 eV)
        assert np.all(energies >= 0)
        assert np.all(energies <= 15e6)
        # Mean should be around the peak (~2 MeV = 2e6 eV)
        assert 1e6 < np.mean(energies) < 3e6

    def test_sample_energy_before_calculate(self, uo2_material):
        """Test that sampling before calculate raises RuntimeError."""
        with patch.object(_alpha_n_module, '_ALPHANSO_AVAILABLE', True):
            src = AlphaNSource(material=uo2_material)
        with pytest.raises(RuntimeError, match="No energy spectrum"):
            src.sample_energy(10)

    def test_invalid_calc_type(self, uo2_material):
        """Test that invalid calc_type raises ValueError."""
        with patch.object(_alpha_n_module, '_ALPHANSO_AVAILABLE', True):
            with pytest.raises(ValueError):
                AlphaNSource(material=uo2_material, calc_type='beam')

    def test_alphanso_not_installed(self, uo2_material):
        """Test that missing alphanso raises ImportError."""
        with patch.object(_alpha_n_module, '_ALPHANSO_AVAILABLE', False):
            with pytest.raises(ImportError, match="alphanso"):
                AlphaNSource(material=uo2_material)


class TestAlphaNManager:
    """Tests for AlphaNManager class."""

    def test_add_source(self, uo2_material):
        """Test adding a source to the manager."""
        model = MagicMock(spec=openmc.Model)
        with patch.object(_alpha_n_module, '_ALPHANSO_AVAILABLE', True):
            manager = AlphaNManager(model)
            manager.add_source(uo2_material)

        assert len(manager.sources) == 1
        assert manager.sources[0].material is uo2_material

    def test_calculate_sources(self, uo2_material, mock_alphanso_result):
        """Test calculating all sources."""
        mock_transport = MagicMock()
        mock_transport.calculate.return_value = mock_alphanso_result

        model = MagicMock(spec=openmc.Model)
        with patch.object(_alpha_n_module, '_ALPHANSO_AVAILABLE', True), \
             patch.object(_alpha_n_module, 'AlphansoTransport', mock_transport):
            manager = AlphaNManager(model)
            manager.add_source(uo2_material)
            manager.calculate_sources()

        assert manager.sources[0].neutron_yield == 2.0e3
        assert manager.total_source_rate > 0

    def test_set_alpha_n_fraction(self, uo2_material):
        """Test setting the (α,n) fraction."""
        model = MagicMock(spec=openmc.Model)
        with patch.object(_alpha_n_module, '_ALPHANSO_AVAILABLE', True):
            manager = AlphaNManager(model)
        manager.set_alpha_n_fraction(0.05)

        assert manager._alpha_n_fraction == 0.05

    def test_invalid_fraction(self, uo2_material):
        """Test that invalid fraction values raise errors."""
        model = MagicMock(spec=openmc.Model)
        with patch.object(_alpha_n_module, '_ALPHANSO_AVAILABLE', True):
            manager = AlphaNManager(model)

        with pytest.raises(ValueError):
            manager.set_alpha_n_fraction(-0.1)

        with pytest.raises(ValueError):
            manager.set_alpha_n_fraction(1.5)

    def test_total_source_rate(self, uo2_material):
        """Test total source rate computation."""
        model = MagicMock(spec=openmc.Model)
        with patch.object(_alpha_n_module, '_ALPHANSO_AVAILABLE', True):
            manager = AlphaNManager(model)

        # No sources => rate is 0
        assert manager.total_source_rate == 0.0

    def test_alphanso_not_installed(self):
        """Test that missing alphanso raises ImportError."""
        model = MagicMock(spec=openmc.Model)
        with patch.object(_alpha_n_module, '_ALPHANSO_AVAILABLE', False):
            with pytest.raises(ImportError, match="alphanso"):
                AlphaNManager(model)


class TestMaterialConversionEdgeCases:
    """Test edge cases in material-to-matdef conversion."""

    def test_single_nuclide(self):
        """Test material with a single nuclide."""
        mat = openmc.Material()
        mat.add_nuclide('Be9', 1.0, 'wo')
        mat.set_density('g/cm3', 1.85)

        matdef = _material_to_matdef(mat)
        assert 4009 in matdef
        assert pytest.approx(matdef[4009]) == 1.0

    def test_heavy_nuclides(self):
        """Test material with heavy nuclides (Pu, Am)."""
        mat = openmc.Material()
        mat.add_nuclide('Pu238', 0.3, 'wo')
        mat.add_nuclide('Pu239', 0.5, 'wo')
        mat.add_nuclide('Am241', 0.2, 'wo')
        mat.set_density('g/cm3', 15.0)

        matdef = _material_to_matdef(mat)
        assert 94238 in matdef
        assert 94239 in matdef
        assert 95241 in matdef
        assert pytest.approx(sum(matdef.values()), rel=1e-10) == 1.0
