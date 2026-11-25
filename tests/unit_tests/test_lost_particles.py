from pathlib import Path

import numpy as np
import openmc
import openmc.lib
import pytest

from tests.testing_harness import config


@pytest.fixture
def model():
    mat = openmc.Material()
    mat.add_nuclide('N14', 1.0)
    mat.set_density('g/cm3', 1e-5)

    s1 = openmc.Sphere(r=80.0)
    s2 = openmc.Sphere(r=90.0)
    s3 = openmc.Sphere(r=100.0, boundary_type='vacuum')
    cell1 = openmc.Cell(fill=mat, region=-s1)
    cell2 = openmc.Cell(fill=mat, region=+s2 & -s3)
    model = openmc.Model()
    model.geometry = openmc.Geometry([cell1, cell2])

    model.settings.run_mode = 'fixed source'
    model.settings.batches = 10
    model.settings.inactive = 5
    model.settings.particles = 50
    model.settings.max_lost_particles = 1000
    model.settings.source = openmc.IndependentSource(space=openmc.stats.Point())

    return model


def test_max_write_lost_particles(model: openmc.Model, run_in_tmpdir):
    # Set maximum number of lost particle restart files
    model.settings.max_write_lost_particles = 5

    # Run OpenMC to generate lost particle files. Use one thread so that we know
    # exactly how much will be produced. If running in MPI mode, setup proper
    # keyword arguments for run()
    kwargs = {'openmc_exec': config['exe']}
    if config['mpi']:
        kwargs['mpi_args'] = [config['mpiexec'], '-n', config['mpi_np']]
    model.run(threads=1, **kwargs)

    # Make sure number of lost particle files is as expected
    lost_particle_files = list(Path.cwd().glob('particle*.h5'))
    n_procs = int(config['mpi_np']) if config['mpi'] else 1
    assert len(lost_particle_files) == model.settings.max_write_lost_particles * n_procs



def test_raytrace_lost_rays(model: openmc.Model, run_in_tmpdir, mpi_intracomm):
    # Rays that pass through the undefined region between the spheres are
    # flagged in the ID buffer of a solid ray-traced plot instead of stopping
    # OpenMC
    model.init_lib(output=False, intracomm=mpi_intracomm)
    try:
        inner, outer = model.geometry.root_universe.cells.values()
        plot = openmc.lib.SolidRayTracePlot()
        plot.pixels = (20, 20)
        plot.color_by = openmc.lib.SolidRayTracePlot.COLOR_BY_CELL
        plot.set_default_colors()
        plot.set_all_opaque()
        plot.camera_position = (300.0, 0.0, 0.0)
        plot.look_at = (0.0, 0.0, 0.0)
        plot.up = (0.0, 0.0, 1.0)
        plot.fov = 60.0

        # Looking through the outer shell, every ray that reaches the inner
        # sphere crosses the gap
        plot.set_domain_styles([outer.id], alphas=[0.0])
        _, depth, ids, _ = plot.render(normals=False)
        assert ids[10, 10] == -2
        assert depth[10, 10] == pytest.approx(300.0 - 80.0, rel=1e-6)
        assert np.all(ids[np.isfinite(depth)] == -2)

        # Rays stop at the outer shell when it is opaque
        plot.set_domain_styles([outer.id], alphas=[1.0])
        _, depth, ids, _ = plot.render(normals=False)
        assert ids[10, 10] == outer.id
        assert depth[10, 10] == pytest.approx(300.0 - 100.0, rel=1e-6)
    finally:
        model.finalize_lib()
