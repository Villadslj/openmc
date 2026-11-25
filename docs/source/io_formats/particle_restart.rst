.. _io_particle_restart:

============================
Particle Restart File Format
============================

The current version of the particle restart file format is 2.2.

**/**

:Attributes: - **filetype** (*char[]*) -- String indicating the type of file.
             - **version** (*int[2]*) -- Major and minor version of the particle
               restart file format.
             - **openmc_version** (*int[3]*) -- Major, minor, and release
               version number for OpenMC.
             - **git_sha1** (*char[40]*) -- Git commit SHA-1 hash.

:Datasets: - **current_batch** (*int*) -- The number of batches already
             simulated.
           - **generations_per_batch** (*int*) -- Number of generations per
             batch.
           - **current_generation** (*int*) -- The number of generations already
             simulated.
           - **n_particles** (*int8_t*) -- Number of particles used per
             generation.
           - **run_mode** (*char[]*) -- Run mode used, either 'fixed source',
             'eigenvalue', or 'particle restart'.
           - **id** (*int8_t*) -- Unique identifier of the particle.
           - **type** (*int32_t*) -- Particle type (PDG number)
           - **weight** (*double*) -- Weight of the particle.
           - **energy** (*double*) -- Energy of the particle in eV for
             continuous-energy mode, or the energy group of the particle for
             multi-group mode.
           - **xyz** (*double[3]*) -- Position of the particle.
           - **uvw** (*double[3]*) -- Direction of the particle.
           - **time** (*double*) -- Time of the particle in [s].

**/lost/**

This group is only present if the particle was lost. It describes the state of
the particle at the moment it was lost.

:Datasets: - **message** (*char[]*) -- Message describing why the particle was
             lost.
           - **xyz** (*double[3]*) -- Position of the particle in [cm].
           - **uvw** (*double[3]*) -- Direction of the particle.
           - **energy** (*double*) -- Energy of the particle in eV for
             continuous-energy mode, or the energy group of the particle for
             multi-group mode.
           - **time** (*double*) -- Time of the particle in [s].
           - **weight** (*double*) -- Weight of the particle.
           - **xyz_last** (*double[3]*) -- Position of the particle at the start
             of its last flight in [cm].
           - **uvw_last** (*double[3]*) -- Direction of the particle at the
             start of its last flight.
           - **n_event** (*int*) -- Number of collisions and surface crossings
             of the particle before the event in which it was lost.
           - **cell_id** (*int[]*) -- ID of the cell at each level of the
             geometry that the particle was last located in, where -1 indicates
             no cell.
           - **universe_id** (*int[]*) -- ID of the universe at each level of
             the geometry that the particle was last located in, where -1
             indicates no universe.
           - **cell_instance** (*int*) -- Instance of the lowest-level cell the
             particle was last located in, or -1 if it is unknown.
           - **cell_last_id** (*int[]*) -- ID of the cell at each level of the
             geometry that the particle was in before it last crossed a surface.
           - **material_id** (*int*) -- ID of the material of the cell the
             particle was last located in, or -1 for void.
           - **surface_id** (*int*) -- ID of the surface the particle was
             crossing when it was lost, or 0 if it was not on a surface. The ID
             is negative if the particle was moving to the negative side of the
             surface.
           - **history** (Compound type) -- Most recent states of the particle
             history, ending with the state in which the particle was lost.
             This dataset is only present if the :ref:`lost_particle_history
             <lost_particle_history>` setting is greater than zero. It has the
             same compound type and attributes as the datasets in a
             :ref:`track file <io_track>`. A state is recorded at the start of
             each flight of a primary or secondary particle, and at the end of
             each primary or secondary particle that was not lost.
