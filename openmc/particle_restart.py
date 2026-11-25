import h5py

import openmc.checkvalue as cv
from .particle_type import ParticleType
from .tracks import ParticleTrack

_VERSION_PARTICLE_RESTART = 2


class Particle:
    """Information used to restart a specific particle that caused a simulation to
    fail.

    Parameters
    ----------
    filename : str
        Path to the particle restart file

    Attributes
    ----------
    current_batch : int
        The batch containing the particle
    generations_per_batch : int
        Number of generations per batch
    current_generation : int
        The generation containing the particle
    n_particles : int
        Number of particles per generation
    run_mode : int
        Type of simulation (criticality or fixed source)
    id : long
        Identifier of the particle
    type : openmc.ParticleType
        Particle type
    weight : float
        Weight of the particle
    energy : float
        Energy of the particle in eV
    xyz : list of float
        Position of the particle
    uvw : list of float
        Directional cosines of the particle
    lost_message : str or None
        Message describing why the particle was lost, or None if the file does
        not contain the state in which the particle was lost. This state is
        written by OpenMC 0.16.1 and later.

        .. versionadded:: 0.16.1
    lost_position : numpy.ndarray or None
        Position of the particle when it was lost in [cm]

        .. versionadded:: 0.16.1
    lost_direction : numpy.ndarray or None
        Direction of the particle when it was lost

        .. versionadded:: 0.16.1
    lost_energy : float or None
        Energy of the particle when it was lost in [eV], or its energy group in
        multigroup mode

        .. versionadded:: 0.16.1
    lost_time : float or None
        Time of the particle when it was lost in [s]

        .. versionadded:: 0.16.1
    lost_weight : float or None
        Weight of the particle when it was lost

        .. versionadded:: 0.16.1
    lost_last_position : numpy.ndarray or None
        Position of the particle at the start of its last flight in [cm]

        .. versionadded:: 0.16.1
    lost_last_direction : numpy.ndarray or None
        Direction of the particle at the start of its last flight

        .. versionadded:: 0.16.1
    lost_n_event : int or None
        Number of events the particle underwent before it was lost

        .. versionadded:: 0.16.1
    lost_cell_ids : numpy.ndarray or None
        IDs of the cells at each level of the geometry that the particle was
        last located in, where -1 indicates no cell

        .. versionadded:: 0.16.1
    lost_universe_ids : numpy.ndarray or None
        IDs of the universes at each level of the geometry that the particle
        was last located in, where -1 indicates no universe

        .. versionadded:: 0.16.1
    lost_cell_instance : int or None
        Instance of the lowest-level cell the particle was last located in, or
        -1 if it is unknown

        .. versionadded:: 0.16.1
    lost_last_cell_ids : numpy.ndarray or None
        IDs of the cells at each level of the geometry that the particle was in
        before it last crossed a surface

        .. versionadded:: 0.16.1
    lost_material_id : int or None
        ID of the material of the cell the particle was last located in, where
        -1 indicates void

        .. versionadded:: 0.16.1
    lost_surface_id : int or None
        Signed ID of the surface the particle was crossing when it was lost,
        where the sign indicates the side of the surface it was moving to, or 0
        if it was not on a surface

        .. versionadded:: 0.16.1
    history : list of openmc.ParticleTrack or None
        Most recent states of the particle history, ending with the state in
        which the particle was lost. Each primary or secondary particle has its
        own track. Only present if the simulation was run with
        :attr:`openmc.Settings.lost_particle_history` greater than zero.

        .. versionadded:: 0.16.1

    """

    def __init__(self, filename):
        with h5py.File(filename, 'r') as f:

            # Ensure filetype and version are correct
            cv.check_filetype_version(f, 'particle restart', _VERSION_PARTICLE_RESTART)

            self.current_batch = f['current_batch'][()]
            self.current_generation = f['current_generation'][()]
            self.energy = f['energy'][()]
            self.generations_per_batch = f['generations_per_batch'][()]
            self.id = f['id'][()]
            self.type = ParticleType(f['type'][()])
            self.n_particles = f['n_particles'][()]
            self.run_mode = f['run_mode'][()].decode()
            self.uvw = f['uvw'][()]
            self.weight = f['weight'][()]
            self.xyz = f['xyz'][()]

            # State in which the particle was lost
            lost = f.get('lost')
            self.lost_message = None
            self.lost_position = None
            self.lost_direction = None
            self.lost_energy = None
            self.lost_time = None
            self.lost_weight = None
            self.lost_last_position = None
            self.lost_last_direction = None
            self.lost_n_event = None
            self.lost_cell_ids = None
            self.lost_universe_ids = None
            self.lost_cell_instance = None
            self.lost_last_cell_ids = None
            self.lost_material_id = None
            self.lost_surface_id = None
            self.history = None
            if lost is None:
                return

            self.lost_message = lost['message'][()].decode()
            self.lost_position = lost['xyz'][()]
            self.lost_direction = lost['uvw'][()]
            self.lost_energy = lost['energy'][()]
            self.lost_time = lost['time'][()]
            self.lost_weight = lost['weight'][()]
            self.lost_last_position = lost['xyz_last'][()]
            self.lost_last_direction = lost['uvw_last'][()]
            self.lost_n_event = int(lost['n_event'][()])
            self.lost_cell_ids = lost['cell_id'][()]
            self.lost_universe_ids = lost['universe_id'][()]
            self.lost_cell_instance = int(lost['cell_instance'][()])
            self.lost_last_cell_ids = lost['cell_last_id'][()]
            self.lost_material_id = int(lost['material_id'][()])
            self.lost_surface_id = int(lost['surface_id'][()])

            if 'history' in lost:
                dset = lost['history']
                states = dset[()]
                offsets = dset.attrs['offsets']
                particles = dset.attrs['particles']
                self.history = [
                    ParticleTrack(ParticleType(particle), states[start:end])
                    for particle, start, end in
                    zip(particles, offsets[:-1], offsets[1:])
                ]
