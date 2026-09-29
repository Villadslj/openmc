from contextlib import nullcontext
from pathlib import Path
import re

import numpy as np
import pytest
import openmc
from openmc.deplete import Chain, R2SManager
from openmc.deplete.r2s import enumerate_regions, get_activation_materials


@pytest.fixture
def simple_model_and_mesh():
    # Define two materials: water and Ni
    h2o = openmc.Material()
    h2o.add_nuclide("H1", 2.0)
    h2o.add_nuclide("O16", 1.0)
    h2o.set_density("g/cm3", 1.0)
    nickel = openmc.Material()
    nickel.add_element("Ni", 1.0)
    nickel.set_density("g/cm3", 4.0)

    # Geometry: two half-spaces split by x=0 plane
    left = openmc.XPlane(0.0)
    x_min = openmc.XPlane(-10.0, boundary_type='vacuum')
    x_max = openmc.XPlane(10.0, boundary_type='vacuum')
    y_min = openmc.YPlane(-10.0, boundary_type='vacuum')
    y_max = openmc.YPlane(10.0, boundary_type='vacuum')
    z_min = openmc.ZPlane(-10.0, boundary_type='vacuum')
    z_max = openmc.ZPlane(10.0, boundary_type='vacuum')

    c1 = openmc.Cell(fill=h2o, region=+x_min & -left & +y_min & -y_max & +z_min & -z_max)
    c2 = openmc.Cell(fill=nickel, region=+left & -x_max & +y_min & -y_max & +z_min & -z_max)
    c1.volume = 4000.0
    c2.volume = 4000.0
    geometry = openmc.Geometry([c1, c2])

    # Simple settings with a point source
    settings = openmc.Settings()
    settings.batches = 2
    settings.particles = 250
    settings.run_mode = 'fixed source'
    settings.source = openmc.IndependentSource()
    model = openmc.Model(geometry, settings=settings)

    mesh = openmc.RegularMesh()
    mesh.lower_left = (-10.0, -10.0, -10.0)
    mesh.upper_right = (10.0, 10.0, 10.0)
    mesh.dimension = (1, 1, 1)
    return model, (c1, c2), mesh


@pytest.fixture
def source_stage_manager(simple_model_and_mesh):
    model, (c1, c2), _ = simple_model_and_mesh
    r2s = R2SManager(model, [c1, c2])
    r2s.results['depletion_results'] = [None, None]
    r2s.results['activation_materials'] = [c1.fill, c2.fill]
    bounding_boxes = {c1.id: c1.bounding_box, c2.id: c2.bounding_box}
    return r2s, bounding_boxes


@pytest.fixture
def activation_regions():
    openmc.reset_auto_ids()
    materials = [openmc.Material(material_id=i) for i in (10, 20, 30)]
    model = openmc.Model(geometry=openmc.Geometry([
        openmc.Cell(fill=material) for material in materials
    ]))
    mesh = openmc.RegularMesh()
    mesh.dimension = (3,)
    mesh.lower_left = (0.0,)
    mesh.upper_right = (3.0,)
    mmv = openmc.MeshMaterialVolumes(
        np.array([[10, 20], [-1, -1], [30, -1]]),
        np.array([[0.4, 0.6], [0.0, 0.0], [1.0, 0.0]]),
    )
    return model, mesh, mmv


def test_enumerate_regions(activation_regions):
    model, mesh, mmv = activation_regions
    regions = enumerate_regions([mmv])

    assert regions == [(0, 0, 10), (0, 0, 20), (0, 2, 30)]

    mesh_filter = openmc.MeshMaterialFilter.from_volumes(mesh, mmv)
    filter_regions = [
        (0, int(elem), int(mat_id)) for elem, mat_id in mesh_filter.bins
    ]
    assert filter_regions == regions

    activation_materials = get_activation_materials(model, [mmv])
    name_pattern = re.compile(r'Mesh (\d+), Element (\d+), Material (\d+)')
    material_regions = [
        tuple(map(int, name_pattern.fullmatch(mat.name).groups()))
        for mat in activation_materials
    ]
    assert material_regions == regions


def test_step2_rejects_inconsistent_counts(activation_regions, tmp_path):
    model, mesh, mmv = activation_regions
    r2s = R2SManager(model, mesh)
    r2s.results = {
        'mesh_material_volumes': [mmv],
        'regions': enumerate_regions([mmv]),
        'fluxes': [object(), object()],
        'micros': [object(), object(), object()],
    }

    with pytest.raises(
        RuntimeError,
        match=(
            'Inconsistent activation data: 3 regions, 3 materials, 2 fluxes, '
            '3 micros'
        ),
    ):
        r2s.step2_activation([1.0], 1.0, output_dir=tmp_path)


@pytest.mark.parametrize('mat_ids', [(100, 100), (101, 100)])
def test_step2_rejects_nonmonotonic_material_ids(
    activation_regions, tmp_path, monkeypatch, mat_ids
):
    model, mesh, mmv = activation_regions
    activation_materials = openmc.Materials()
    for mat_id in mat_ids:
        with pytest.warns(openmc.IDWarning) if mat_id in {
            mat.id for mat in activation_materials
        } else nullcontext():
            activation_materials.append(openmc.Material(material_id=mat_id))

    monkeypatch.setattr(
        'openmc.deplete.r2s.get_activation_materials',
        lambda model, mmv_list: activation_materials,
    )
    r2s = R2SManager(model, mesh)
    r2s.results = {
        'mesh_material_volumes': [mmv],
        'regions': [(0, 0, 10), (0, 0, 20)],
        'fluxes': [object(), object()],
        'micros': [object(), object()],
    }

    with pytest.raises(RuntimeError, match='not unique and increasing'):
        r2s.step2_activation([1.0], 1.0, output_dir=tmp_path)


def test_r2s_mesh_expected_output(simple_model_and_mesh, tmp_path):
    model, (c1, c2), mesh = simple_model_and_mesh

    # Use mesh-based domains
    r2s = R2SManager(model, mesh)

    # Use custom reduced chain file for Ni
    chain = Chain.from_xml(Path(__file__).parents[1] / "chain_ni.xml")

    # Run R2S calculation
    outdir = r2s.run(
        timesteps=[(1.0, 'd')],
        source_rates=[1.0],
        output_dir=tmp_path,
        chain_file=chain,
        micro_kwargs={'nuclides': ['Ni58'], 'reactions': ['(n,p)']},
    )

    # Check directories and files exist
    nt = Path(outdir) / 'neutron_transport'
    assert (nt / 'fluxes.npy').exists()
    assert (nt / 'micros.h5').exists()
    assert (nt / 'regions.json').exists()
    assert (nt / 'tally_ids.json').exists()
    assert (nt / 'mesh_material_volumes_0.npz').exists()
    act = Path(outdir) / 'activation'
    assert (act / 'depletion_results.h5').exists()
    pt = Path(outdir) / 'photon_transport'
    assert (pt / 'tally_ids.json').exists()
    assert not (pt / 'time_0').exists()
    assert (pt / 'time_1' / 'statepoint.2.h5').exists()

    # Basic results structure checks
    assert len(r2s.results['fluxes']) == 2
    assert len(r2s.results['micros']) == 2
    assert r2s.results['neutron_tallies'] == []
    assert len(r2s.results['mesh_material_volumes']) == 1
    assert len(r2s.results['mesh_material_volumes'][0]) == 2
    assert len(r2s.results['activation_materials']) == 2
    assert len(r2s.results['depletion_results']) == 2
    assert list(r2s.results['photon_sources']) == [1]
    assert r2s.results['photon_sources'][1]

    # Check activation materials
    amats = r2s.results['activation_materials']
    assert all(m.depletable for m in amats)
    # Volumes preserved
    assert {m.volume for m in amats} == {c1.volume, c2.volume}

    # Check loading results
    r2s_loaded = R2SManager(model, mesh)
    r2s_loaded.load_results(outdir)
    assert len(r2s_loaded.results['fluxes']) == 2
    assert len(r2s_loaded.results['micros']) == 2
    assert len(r2s_loaded.results['regions']) == 2
    assert r2s_loaded.results['neutron_tallies'] == []
    assert len(r2s_loaded.results['mesh_material_volumes']) == 1
    assert len(r2s_loaded.results['mesh_material_volumes'][0]) == 2
    assert len(r2s_loaded.results['activation_materials']) == 2
    assert len(r2s_loaded.results['depletion_results']) == 2


def test_r2s_multi_mesh(simple_model_and_mesh, tmp_path):
    model, _, _ = simple_model_and_mesh

    # Two 1x1x1 meshes that together cover the full domain, split along y.
    # Each mesh element spans the full x range [-10, 10], crossing the x=0
    # material boundary, so both meshes contain both materials within their
    # single element.
    mesh1 = openmc.RegularMesh()
    mesh1.lower_left = (-10.0, -10.0, -10.0)
    mesh1.upper_right = (10.0, 0.0, 10.0)
    mesh1.dimension = (1, 1, 1)
    mesh2 = openmc.RegularMesh()
    mesh2.lower_left = (-10.0, 0.0, -10.0)
    mesh2.upper_right = (10.0, 10.0, 10.0)
    mesh2.dimension = (1, 1, 1)

    r2s = R2SManager(model, [mesh1, mesh2])
    chain = Chain.from_xml(Path(__file__).parents[1] / "chain_ni.xml")

    outdir = r2s.run(
        timesteps=[(1.0, 'd')],
        source_rates=[1.0],
        photon_time_indices=[1],
        output_dir=tmp_path,
        chain_file=chain,
        micro_kwargs={'nuclides': ['Ni58'], 'reactions': ['(n,p)']},
    )

    # Check that per-mesh MMV files were written
    nt = Path(outdir) / 'neutron_transport'
    assert (nt / 'fluxes.npy').exists()
    assert (nt / 'micros.h5').exists()
    assert (nt / 'regions.json').exists()
    assert (nt / 'mesh_material_volumes_0.npz').exists()
    assert (nt / 'mesh_material_volumes_1.npz').exists()
    act = Path(outdir) / 'activation'
    assert (act / 'depletion_results.h5').exists()
    pt = Path(outdir) / 'photon_transport'
    assert (pt / 'tally_ids.json').exists()
    assert (pt / 'time_1' / 'statepoint.2.h5').exists()

    # Two meshes, each with 1 element containing both materials →
    # 2 element-material combinations per mesh, 4 total
    assert len(r2s.results['mesh_material_volumes']) == 2
    assert len(r2s.results['mesh_material_volumes'][0]) == 2
    assert len(r2s.results['mesh_material_volumes'][1]) == 2
    assert len(r2s.results['fluxes']) == 4
    assert len(r2s.results['micros']) == 4
    assert len(r2s.results['activation_materials']) == 4
    assert len(r2s.results['depletion_results']) == 2

    # Activation material names encode mesh index
    amats = r2s.results['activation_materials']
    assert all(m.depletable for m in amats)
    assert any('Mesh 0' in m.name for m in amats)
    assert any('Mesh 1' in m.name for m in amats)

    # Check loading results
    r2s_loaded = R2SManager(model, [mesh1, mesh2])
    r2s_loaded.load_results(outdir)
    assert len(r2s_loaded.results['mesh_material_volumes']) == 2
    assert len(r2s_loaded.results['mesh_material_volumes'][0]) == 2
    assert len(r2s_loaded.results['mesh_material_volumes'][1]) == 2
    assert len(r2s_loaded.results['regions']) == 4
    assert len(r2s_loaded.results['activation_materials']) == 4
    assert len(r2s_loaded.results['depletion_results']) == 2


def test_r2s_cell_expected_output(simple_model_and_mesh, tmp_path):
    model, (c1, c2), _ = simple_model_and_mesh
    tally = openmc.Tally(name='neutron flux')
    tally.scores = ['flux']
    model.tallies = [tally]

    # Use cell-based domains
    r2s = R2SManager(model, [c1, c2])

    # Use custom reduced chain file for Ni
    chain = Chain.from_xml(Path(__file__).parents[1] / "chain_ni.xml")

    # Run R2S calculation
    bounding_boxes = {c1.id: c1.bounding_box, c2.id: c2.bounding_box}
    outdir = r2s.run(
        timesteps=[(1.0, 'd')],
        source_rates=[1.0],
        photon_time_indices=[1],
        by_parent_nuclide=True,
        output_dir=tmp_path,
        bounding_boxes=bounding_boxes,
        chain_file=chain,
        micro_kwargs={'nuclides': ['Ni58'], 'reactions': ['(n,p)']},
    )

    # Check directories and files exist
    nt = Path(outdir) / 'neutron_transport'
    assert (nt / 'fluxes.npy').exists()
    assert (nt / 'micros.h5').exists()
    assert (nt / 'tally_ids.json').exists()
    act = Path(outdir) / 'activation'
    assert (act / 'depletion_results.h5').exists()
    pt = Path(outdir) / 'photon_transport'
    assert (pt / 'tally_ids.json').exists()
    assert (pt / 'time_1' / 'statepoint.2.h5').exists()

    # Basic results structure checks
    assert len(r2s.results['fluxes']) == 2
    assert len(r2s.results['micros']) == 2
    assert len(r2s.results['neutron_tallies']) == 1
    assert r2s.results['neutron_tallies'][0].id == tally.id
    assert r2s.results['neutron_tallies'][0].mean[0] > 0.0
    assert len(r2s.results['activation_materials']) == 2
    assert len(r2s.results['depletion_results']) == 2
    assert r2s.photon_model.tallies[0].contains_filter(
        openmc.ParentNuclideFilter)

    # Check activation materials
    amats = r2s.results['activation_materials']
    assert all(m.depletable for m in amats)
    # Names include cell IDs
    assert any(f"Cell {c1.id}" in m.name for m in amats)
    assert any(f"Cell {c2.id}" in m.name for m in amats)
    # Volumes preserved
    assert {m.volume for m in amats} == {c1.volume, c2.volume}

    # Check loading results
    r2s_loaded = R2SManager(model, [c1, c2])
    r2s_loaded.load_results(outdir)
    assert len(r2s_loaded.results['fluxes']) == 2
    assert len(r2s_loaded.results['micros']) == 2
    assert len(r2s_loaded.results['neutron_tallies']) == 1
    assert r2s_loaded.results['neutron_tallies'][0].id == tally.id
    assert r2s_loaded.results['neutron_tallies'][0].mean[0] > 0.0
    assert len(r2s_loaded.results['activation_materials']) == 2
    assert len(r2s_loaded.results['depletion_results']) == 2


def test_step4_requires_photon_sources(simple_model_and_mesh, tmp_path):
    model, (c1, c2), _ = simple_model_and_mesh
    r2s = R2SManager(model, [c1, c2])
    output_dir = tmp_path / 'photon'

    with pytest.raises(RuntimeError, match='step3_photon_source'):
        r2s.step4_photon_transport(output_dir)

    r2s.results['photon_sources'] = {}
    with pytest.raises(RuntimeError, match='No decay photon sources'):
        r2s.step4_photon_transport(output_dir)

    assert not output_dir.exists()


def test_default_photon_times_skip_empty_sources(
    source_stage_manager, tmp_path, monkeypatch
):
    r2s, bounding_boxes = source_stage_manager
    source = object()
    sources_by_time = {0: [], 1: [source]}
    monkeypatch.setattr(
        r2s, '_create_photon_sources',
        lambda time_index, work_items: sources_by_time[time_index])

    r2s.step3_photon_source(
        bounding_boxes=bounding_boxes, output_dir=tmp_path)

    assert r2s.results['photon_sources'] == {1: [source]}


def test_explicit_empty_photon_source_fails(
    source_stage_manager, tmp_path, monkeypatch
):
    r2s, bounding_boxes = source_stage_manager
    source = object()
    sources_by_time = {0: [], 1: [source]}
    monkeypatch.setattr(
        r2s, '_create_photon_sources',
        lambda time_index, work_items: sources_by_time[time_index])
    r2s.results['photon_sources'] = {99: [source]}

    with pytest.raises(RuntimeError, match='requested time indices: 0'):
        r2s.step3_photon_source(
            [0, 1], bounding_boxes, output_dir=tmp_path)

    assert 'photon_sources' not in r2s.results


def test_default_photon_times_require_a_source(
    source_stage_manager, tmp_path, monkeypatch
):
    r2s, bounding_boxes = source_stage_manager
    monkeypatch.setattr(
        r2s, '_create_photon_sources',
        lambda time_index, work_items: [])

    with pytest.raises(RuntimeError, match='at any depletion time'):
        r2s.step3_photon_source(
            bounding_boxes=bounding_boxes, output_dir=tmp_path)

    assert 'photon_sources' not in r2s.results


@pytest.mark.parametrize(
    ('time_indices', 'exception'),
    [
        ([], ValueError),
        ([2], IndexError),
        ([-3], IndexError),
        ([1.0], TypeError),
    ],
)
def test_photon_time_index_validation(
    source_stage_manager, tmp_path, time_indices, exception
):
    r2s, bounding_boxes = source_stage_manager

    with pytest.raises(exception):
        r2s.step3_photon_source(
            time_indices, bounding_boxes, output_dir=tmp_path)

    assert 'photon_sources' not in r2s.results
