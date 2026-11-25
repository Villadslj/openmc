#ifndef OPENMC_RAY_H
#define OPENMC_RAY_H

#include "openmc/particle_data.h"
#include "openmc/position.h"

namespace openmc {

// Base class that implements ray tracing logic, not necessarily through
// defined regions of the geometry but also outside of it.
class Ray : public GeometryState {

public:
  // Initialize from location and direction
  Ray(Position r, Direction u) { init_from_r_u(r, u); }

  // Initialize from known geometry state
  Ray(const GeometryState& p) : GeometryState(p) {}

  // Called at every surface intersection within the model
  virtual void on_intersection() = 0;

  // Called when the ray leaves the region where cells are defined. Returning
  // true continues the trace through the void to look for a point where the
  // ray enters the model again, which is needed for non-convex models. By
  // default the trace ends, which assumes a convex model.
  virtual bool on_exit_model() { return false; }

  /*
   * Traces the ray through the geometry, calling on_intersection
   * at every surface boundary.
   */
  void trace();

  // Stops the ray and exits tracing when called from on_intersection
  void stop() { stop_ = true; }

  // Whether the ray has been stopped
  bool stopped() const { return stop_; }

  // Whether the geometry could not be resolved somewhere along the ray
  bool lost() const { return lost_; }

  // Unlike particles, rays never abort the program when they get lost. The
  // ray is stopped and flagged as lost instead, so that interactive rendering
  // of broken geometry keeps working.
  void mark_as_lost(const char* message) override;
  using GeometryState::mark_as_lost;

  // Sets the dist_ variable
  void compute_distance();

protected:
  // Records how far the ray has traveled
  double traversal_distance_ {0.0};

  // Whether the current position of the ray was found in a cell. After a
  // failed cell search, the cell and material data are stale.
  bool inside_cell_ {false};

  bool lost_ {false};

private:
  // Max intersections before we assume ray tracing is caught in an infinite
  // loop:
  static const int MAX_INTERSECTIONS = 1000000;

  bool stop_ {false};

  unsigned event_counter_ {0};
};

} // namespace openmc
#endif // OPENMC_RAY_H
