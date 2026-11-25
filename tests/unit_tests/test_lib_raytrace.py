"""Tests for rendering solid ray-traced plots through openmc.lib

The model is a sphere of radius R (cell 11, material 1) inside a box of
half-width L (cell 12, material 2) that is seen by a camera on the x-axis at a
distance D from the origin, so that depths and normals can be checked against
analytic values.
"""

import numpy as np
import pytest

import openmc
import openmc.exceptions as exc
import openmc.lib

from tests import cdtemp


R = 5.0
L = 10.0
D = 20.0
WIDTH, HEIGHT = 40, 30
FOV = 60.0

# With an even number of pixels, the ray of this pixel follows the axis of the
# camera
CENTER = (HEIGHT // 2, WIDTH // 2)

SPHERE_CELL, BOX_CELL = 11, 12
SPHERE_MAT, BOX_MAT = 1, 2


@pytest.fixture(scope='module')
def sphere_model(mpi_intracomm):
    openmc.reset_auto_ids()
    sphere_mat = openmc.Material(material_id=SPHERE_MAT)
    sphere_mat.add_nuclide('H1', 1.0)
    sphere_mat.set_density('g/cm3', 1.0)
    box_mat = openmc.Material(material_id=BOX_MAT)
    box_mat.add_nuclide('O16', 1.0)
    box_mat.set_density('g/cm3', 1.0)

    sphere = openmc.Sphere(r=R)
    box = openmc.model.RectangularParallelepiped(
        -L, L, -L, L, -L, L, boundary_type='vacuum')
    inner = openmc.Cell(cell_id=SPHERE_CELL, fill=sphere_mat, region=-sphere)
    outer = openmc.Cell(cell_id=BOX_CELL, fill=box_mat,
                        region=+sphere & -box)

    model = openmc.Model()
    model.materials = openmc.Materials([sphere_mat, box_mat])
    model.geometry = openmc.Geometry([inner, outer])
    model.settings.run_mode = 'fixed source'
    model.settings.particles = 10
    model.settings.batches = 1
    model.settings.verbosity = 1
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point())

    with cdtemp():
        model.export_to_xml()
        openmc.lib.init(intracomm=mpi_intracomm)
        yield
        openmc.lib.finalize()


@pytest.fixture
def plot(sphere_model):
    plot = openmc.lib.SolidRayTracePlot()
    plot.pixels = (WIDTH, HEIGHT)
    plot.color_by = openmc.lib.SolidRayTracePlot.COLOR_BY_CELL
    plot.set_default_colors()
    plot.set_all_opaque()
    plot.camera_position = (D, 0.0, 0.0)
    plot.look_at = (0.0, 0.0, 0.0)
    plot.up = (0.0, 0.0, 1.0)
    plot.light_position = (D, 0.0, 0.0)
    plot.fov = FOV
    return plot


def perspective_rays():
    """Directions of the pixel rays of the camera used by the plot fixture"""
    # Follows RayTracePlot::get_pixel_ray with a focal plane distance of 10 cm
    dx = 2.0 * 10.0 * np.tan(0.5 * np.radians(FOV))
    dy = HEIGHT / WIDTH * dx
    horiz, vert = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    right = -0.5 * dx + horiz * dx / WIDTH
    up = 0.5 * dy - vert * dy / HEIGHT
    # The camera looks along -x with +y to its right and +z up
    u = np.stack([-10.0 * np.ones_like(right), right, up], axis=-1)
    return u / np.linalg.norm(u, axis=-1, keepdims=True)


def test_render_buffers(plot):
    # Hide the box so that the sphere is the first visible surface
    plot.set_domain_styles([BOX_CELL], alphas=[0.0])
    image, depth, ids, normals = plot.render()

    assert image.shape == (HEIGHT, WIDTH, 3)
    assert image.dtype == np.uint8
    assert depth.shape == (HEIGHT, WIDTH)
    assert depth.dtype == np.float32
    assert ids.shape == (HEIGHT, WIDTH)
    assert ids.dtype == np.int32
    assert normals.shape == (HEIGHT, WIDTH, 3)
    assert normals.dtype == np.float32

    assert depth[CENTER] == pytest.approx(D - R, rel=1e-6)
    assert ids[CENTER] == SPHERE_CELL
    assert normals[CENTER] == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)

    # Compare with the analytic intersection of every pixel ray with the sphere
    u = perspective_rays()
    origin = np.array([D, 0.0, 0.0])
    b = u @ origin
    disc = b**2 - (origin @ origin - R**2)
    hit = ids == SPHERE_CELL
    # Some rays are tangent to the sphere
    assert np.all(disc[hit] >= -1e-9)
    assert np.all(ids[~hit] == -1)
    # Pixel rays that clearly hit or miss the sphere
    assert np.all(hit[disc > 0.5])
    assert not np.any(hit[disc < -0.5])

    # Tangent rays are excluded since the side that their normal faces is
    # determined by roundoff
    hit &= disc > 1e-6
    t = -b[hit] - np.sqrt(disc[hit])
    assert depth[hit] == pytest.approx(t, rel=1e-5)
    position = origin + t[:, np.newaxis] * u[hit]
    assert normals[hit] == pytest.approx(position / R, abs=1e-5)

    # Background pixels have no depth, ID or normal
    hit = ids == SPHERE_CELL
    assert np.all(np.isinf(depth[~hit]))
    assert np.all(normals[~hit] == 0.0)
    assert np.all(image[~hit] == plot.background)


def test_render_optional_buffers(plot):
    result = plot.render(depth=False, ids=False, normals=False)
    assert isinstance(result, openmc.lib.RayTraceResult)
    assert result.depth is None
    assert result.ids is None
    assert result.normals is None

    # The image matches the one created through the original interface
    plot.update_view()
    assert np.array_equal(result.image, plot.create_image())


def test_render_color_by_material(plot):
    plot.color_by = openmc.lib.SolidRayTracePlot.COLOR_BY_MATERIAL
    plot.set_default_colors()
    plot.set_all_opaque()
    _, depth, ids, _ = plot.render(normals=False)
    assert ids[CENTER] == BOX_MAT
    assert depth[CENTER] == pytest.approx(D - L, rel=1e-6)

    plot.set_domain_styles([BOX_MAT], alphas=[0.0])
    _, depth, ids, _ = plot.render(normals=False)
    assert ids[CENTER] == SPHERE_MAT
    assert depth[CENTER] == pytest.approx(D - R, rel=1e-6)


def test_background(plot):
    plot.set_domain_styles([BOX_CELL], alphas=[0.0])
    plot.background = (10, 20, 30)
    assert plot.background == (10, 20, 30)
    image, _, ids, _ = plot.render(depth=False, normals=False)
    assert np.all(image[ids == -1] == (10, 20, 30))


def test_clip_box(plot):
    assert plot.clip_box is None
    assert plot.show_cut_faces

    # Cut away the half of the model that faces the camera
    box = ((-np.inf, -np.inf, -np.inf), (0.0, np.inf, np.inf))
    plot.clip_box = box
    assert plot.clip_box == box
    _, depth, ids, normals = plot.render()

    # The center ray starts on the cut face of the sphere
    assert ids[CENTER] == SPHERE_CELL
    assert depth[CENTER] == pytest.approx(D, rel=1e-6)
    assert normals[CENTER] == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)

    # Rays that pass beside the sphere start on the cut face of the box
    u = perspective_rays()
    t_cut = D / -u[..., 0]
    lateral = t_cut[..., np.newaxis] * u[..., 1:]
    on_box = np.all(np.abs(lateral) < L - 0.5, axis=-1) & \
        (np.linalg.norm(lateral, axis=-1) > R + 0.5)
    assert np.any(on_box)
    assert np.all(ids[on_box] == BOX_CELL)
    assert depth[on_box] == pytest.approx(t_cut[on_box], rel=1e-5)
    assert np.all(normals[on_box] == (1.0, 0.0, 0.0))

    # Without cut faces, the center ray passes through the sphere and shows
    # the box behind it
    plot.show_cut_faces = False
    assert not plot.show_cut_faces
    _, depth, ids, _ = plot.render(normals=False)
    assert ids[CENTER] == BOX_CELL
    assert depth[CENTER] == pytest.approx(D + R, rel=1e-6)

    # Geometry beyond the clip box is not drawn
    plot.show_cut_faces = True
    plot.set_domain_styles([BOX_CELL], alphas=[0.0])
    plot.clip_box = ((R + 1.0, -np.inf, -np.inf), (np.inf, np.inf, np.inf))
    _, _, ids, _ = plot.render(depth=False, normals=False)
    assert np.all(ids == -1)

    plot.clip_box = None
    assert plot.clip_box is None
    _, depth, ids, _ = plot.render(normals=False)
    assert ids[CENTER] == SPHERE_CELL
    assert depth[CENTER] == pytest.approx(D - R, rel=1e-6)


def test_transparency(plot):
    # Without specular shading, colors are composited exactly
    plot.diffuse_fraction = 1.0
    plot.set_domain_styles([SPHERE_CELL, BOX_CELL],
                           [(255, 0, 0), (0, 0, 255)], [1.0, 0.5])
    assert plot.get_alpha(SPHERE_CELL) == 1.0
    assert plot.get_alpha(BOX_CELL) == pytest.approx(0.5)

    image, depth, ids, _ = plot.render(normals=False)

    # The first visible surface is the box, through which the sphere shows
    assert ids[CENTER] == BOX_CELL
    assert depth[CENTER] == pytest.approx(D - L, rel=1e-6)
    assert image[CENTER] == pytest.approx((127, 0, 127), abs=1)

    # Beside the sphere, the background shows through the box
    side = (HEIGHT // 2, 4)
    assert ids[side] == BOX_CELL
    assert image[side] == pytest.approx((127, 127, 255), abs=1)

    # Hiding the box reveals the sphere
    plot.set_domain_styles([BOX_CELL], alphas=[0.0])
    assert plot.get_alpha(BOX_CELL) == 0.0
    image, _, ids, _ = plot.render(depth=False, normals=False)
    assert ids[CENTER] == SPHERE_CELL
    assert tuple(image[CENTER]) == (255, 0, 0)

    # Showing the box again makes it opaque
    plot.set_visibility(BOX_CELL, True)
    assert plot.get_alpha(BOX_CELL) == 1.0
    image, _, ids, _ = plot.render(depth=False, normals=False)
    assert ids[CENTER] == BOX_CELL
    assert tuple(image[CENTER]) == (0, 0, 255)


def test_orthographic(plot):
    assert plot.orthographic_width is None
    plot.set_domain_styles([BOX_CELL], alphas=[0.0])

    # Aim above the sphere so that it appears below the center of the image
    plot.camera_position = (D, 0.0, 1.5)
    plot.look_at = (0.0, 0.0, 1.5)
    plot.pixels = (40, 20)
    _, _, ids, _ = plot.render(depth=False, normals=False)
    rows, _ = np.nonzero(ids == SPHERE_CELL)
    assert rows.mean() > 10.0

    # With a width of 30 cm, pixels are 0.75 cm wide and high, so the center of
    # the sphere is 2 pixels below the center of the image
    plot.orthographic_width = 30.0
    assert plot.orthographic_width == pytest.approx(30.0)
    _, depth, ids, normals = plot.render()
    rows, cols = np.nonzero(ids == SPHERE_CELL)
    assert rows.mean() == pytest.approx(12.0)
    assert cols.mean() == pytest.approx(20.0)

    # The sphere appears as a circle
    assert np.ptp(rows) == np.ptp(cols) == 12

    # Rays are parallel and start on the plane of the camera
    assert depth[12, 20] == pytest.approx(D - R, rel=1e-6)
    assert normals[12, 20] == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)
    assert depth[12, 24] == pytest.approx(D - np.sqrt(R**2 - 3.0**2), rel=1e-6)

    plot.orthographic_width = None
    assert plot.orthographic_width is None


def test_invalid_input(plot):
    plot.set_domain_styles([SPHERE_CELL], [(1, 2, 3)], [1.0])

    # Styles are only changed if all IDs are valid
    with pytest.raises(exc.InvalidIDError):
        plot.set_domain_styles([SPHERE_CELL, 9999], [(4, 5, 6), (7, 8, 9)])
    assert plot.get_color(SPHERE_CELL) == (1, 2, 3)
    with pytest.raises(exc.InvalidIDError):
        plot.get_alpha(9999)

    with pytest.raises(ValueError):
        plot.set_domain_styles([SPHERE_CELL, BOX_CELL], [(1, 2, 3)])
    with pytest.raises(ValueError):
        plot.set_domain_styles([SPHERE_CELL], alphas=[0.5, 0.5])
    with pytest.raises(ValueError):
        plot.set_domain_styles([SPHERE_CELL], [(1, 2, 300)])
    with pytest.raises(exc.InvalidArgumentError):
        plot.set_domain_styles([SPHERE_CELL], alphas=[np.nan])
    with pytest.raises(exc.InvalidArgumentError):
        plot.clip_box = ((1.0, 0.0, 0.0), (0.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        plot.orthographic_width = -1.0
    with pytest.raises(ValueError):
        plot.background = (0, 0, 256)

    # An invalid view is reported as an error instead of stopping OpenMC
    plot.up = (1.0, 0.0, 0.0)
    with pytest.raises(exc.InvalidArgumentError, match='Up vector'):
        plot.render()
    with pytest.raises(exc.InvalidArgumentError, match='Up vector'):
        plot.update_view()
    plot.up = (0.0, 0.0, 1.0)
    plot.look_at = plot.camera_position
    with pytest.raises(exc.InvalidArgumentError):
        plot.render()
