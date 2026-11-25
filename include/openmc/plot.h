#ifndef OPENMC_PLOT_H
#define OPENMC_PLOT_H

#include <cmath>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "openmc/tensor.h"
#include "pugixml.hpp"

#include "hdf5.h"
#include "openmc/array.h"
#include "openmc/cell.h"
#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/geometry.h"
#include "openmc/particle.h"
#include "openmc/position.h"
#include "openmc/random_lcg.h"
#include "openmc/ray.h"
#include "openmc/tallies/filter.h"
#include "openmc/tallies/filter_match.h"
#include "openmc/xml_interface.h"

namespace openmc {

//===============================================================================
// Global variables
//===============================================================================

class PlottableInterface;

namespace model {

extern std::unordered_map<int, int> plot_map; //!< map of plot ids to index
extern vector<std::unique_ptr<PlottableInterface>>
  plots; //!< Plot instance container

extern uint64_t plotter_seed; // Stream index used by the plotter

} // namespace model

//===============================================================================
// RGBColor holds color information for plotted objects
//===============================================================================

struct RGBColor {
  // Constructors
  RGBColor() : red(0), green(0), blue(0) {};
  RGBColor(const int v[3]) : red(v[0]), green(v[1]), blue(v[2]) {};
  RGBColor(int r, int g, int b) : red(r), green(g), blue(b) {};

  RGBColor(const vector<int>& v)
  {
    if (v.size() != 3) {
      throw std::out_of_range("Incorrect vector size for RGBColor.");
    }
    red = v[0];
    green = v[1];
    blue = v[2];
  }

  bool operator==(const RGBColor& other)
  {
    return red == other.red && green == other.green && blue == other.blue;
  }

  RGBColor& operator*=(const double x)
  {
    red *= x;
    green *= x;
    blue *= x;
    return *this;
  }

  // Members
  uint8_t red, green, blue;
};

// some default colors
const RGBColor WHITE {255, 255, 255};
const RGBColor RED {255, 0, 0};
const RGBColor BLACK {0, 0, 0};

/**
 * \class PlottableInterface
 * \brief Interface for plottable objects.
 *
 * PlottableInterface classes must have unique IDs. If no ID (or -1) is
 * provided, the next available ID is assigned automatically. They guarantee
 * the ability to create output in some form. This interface is designed to be
 * implemented by classes that produce plot-relevant data which can be
 * visualized.
 */

typedef tensor::Tensor<RGBColor> ImageData;
class PlottableInterface {
public:
  PlottableInterface() = default;

  void set_default_colors();

private:
  void set_id(pugi::xml_node plot_node);
  int id_ {C_NONE}; // unique plot ID

  void set_bg_color(pugi::xml_node plot_node);
  void set_universe(pugi::xml_node plot_node);
  void set_color_by(pugi::xml_node plot_node);
  void set_user_colors(pugi::xml_node plot_node);
  void set_overlap_color(pugi::xml_node plot_node);
  void set_mask(pugi::xml_node plot_node);

protected:
  // Plot output filename, derived classes have logic to set it
  std::string path_plot_;

public:
  enum class PlotColorBy { cells = 0, mats = 1 };

  // Generates image data based on plot parameters and returns it
  virtual ImageData create_image() const = 0;

  // Creates the output image named path_plot_
  virtual void create_output() const = 0;

  // Write populated image data to file
  void write_image(const ImageData& data) const;

  // Print useful info to the terminal
  virtual void print_info() const = 0;

  const std::string& path_plot() const { return path_plot_; }
  std::string& path_plot() { return path_plot_; }
  int id() const { return id_; }
  void set_id(int id = C_NONE);
  int level() const { return level_; }
  PlotColorBy color_by() const { return color_by_; }

  // Public color-related data
  PlottableInterface(pugi::xml_node plot_node);
  virtual ~PlottableInterface() = default;
  int level_ {-1};                           // Universe level to plot
  bool color_overlaps_ {false};              // Show overlapping cells?
  PlotColorBy color_by_ {PlotColorBy::mats}; // Plot coloring (cell/material)
  RGBColor not_found_ {WHITE};               // Plot background color
  RGBColor overlap_color_ {RED};             // Plot overlap color
  vector<RGBColor> colors_;                  // Plot colors
};

struct IdData {
  // Constructor
  IdData(size_t h_res, size_t v_res, bool include_filter = false);

  // Methods
  void set_value(size_t y, size_t x, const Particle& p, int level,
    Filter* filter = nullptr, FilterMatch* match = nullptr);
  void set_overlap(size_t y, size_t x, int overlap_idx);

  // Members
  tensor::Tensor<int32_t> data_; //!< 2D array of cell & material ids
};

struct PropertyData {
  // Constructor
  PropertyData(size_t h_res, size_t v_res, bool include_filter = false);

  // Methods
  void set_value(size_t y, size_t x, const Particle& p, int level,
    Filter* filter = nullptr, FilterMatch* match = nullptr);
  void set_overlap(size_t y, size_t x, int overlap_idx);

  // Members
  tensor::Tensor<double> data_; //!< 2D array of temperature & density data
};

struct RasterData {
  // Constructor
  RasterData(size_t h_res, size_t v_res, bool include_filter = false);

  // Methods
  void set_value(size_t y, size_t x, const Particle& p, int level,
    Filter* filter = nullptr, FilterMatch* match = nullptr);
  void set_overlap(size_t y, size_t x, int overlap_idx);

  // Members
  tensor::Tensor<int32_t>
    id_data_; //!< [v_res, h_res, 3 or 4]: cell, instance, mat, [filter_bin]
  tensor::Tensor<double>
    property_data_;     //!< [v_res, h_res, 2]: temperature, density
  bool include_filter_; //!< Whether filter bin index is included
};

//===============================================================================
// Plot class
//===============================================================================

class SlicePlotBase {
public:
  template<class T>
  T get_map(int32_t filter_index = -1) const;

  enum class PlotBasis { xy = 1, xz = 2, yz = 3 };

  // Accessors

  const std::array<size_t, 3>& pixels() const { return pixels_; }
  std::array<size_t, 3>& pixels() { return pixels_; }

  // Members
public:
  Position origin_;         //!< Plot origin in geometry
  Direction u_span_;        //!< Full-width span vector in geometry
  Direction v_span_;        //!< Full-height span vector in geometry
  array<size_t, 3> pixels_; //!< Plot size in pixels
  bool show_overlaps_;      //!< Show overlapping cells?
  int slice_level_ {-1};    //!< Plot universe level
private:
};

template<class T>
T SlicePlotBase::get_map(int32_t filter_index) const
{

  size_t width = pixels_[0];
  size_t height = pixels_[1];

  // Determine if filter is being used
  bool include_filter = (filter_index >= 0);
  Filter* filter = nullptr;
  if (include_filter) {
    filter = model::tally_filters[filter_index].get();
  }

  // size data array
  T data(width, height, include_filter);

  // compute pixel steps and top-left pixel center
  Direction u_step = u_span_ / static_cast<double>(width);
  Direction v_step = v_span_ / static_cast<double>(height);

  Position start =
    origin_ - 0.5 * u_span_ + 0.5 * v_span_ + 0.5 * u_step - 0.5 * v_step;

  // Validate that span vectors define a valid plane
  Position cross = u_span_.cross(v_span_);
  if (cross.norm() == 0.0) {
    fatal_error("Slice span vectors are invalid (zero area).");
  }

  // Use an arbitrary direction that is not aligned with any coordinate axis.
  // The direction has no physical meaning for plotting but is used by
  // Surface::sense() to break ties when a pixel is coincident with a surface.
  Direction dir = {1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0), 0.0};

#pragma omp parallel
  {
    Particle p;
    p.r() = start;
    p.u() = dir;
    p.coord(0).universe() = model::root_universe;
    int level = slice_level_;
    int j {};
    FilterMatch match;

#pragma omp for
    for (int y = 0; y < height; y++) {
      Position row = start - v_step * static_cast<double>(y);
      for (int x = 0; x < width; x++) {
        p.r() = row + u_step * static_cast<double>(x);
        p.n_coord() = 1;
        // local variables
        bool found_cell = exhaustive_find_cell(p);
        j = p.n_coord() - 1;
        if (level >= 0) {
          j = level;
        }
        if (found_cell) {
          data.set_value(y, x, p, j, filter, &match);
        }
        if (show_overlaps_) {
          int overlap_idx = check_cell_overlap(p, false);
          if (overlap_idx >= 0) {
            data.set_overlap(y, x, overlap_idx);
          }
        }
      } // inner for
    }
  }

  return data;
}

// Represents either a voxel or pixel plot
class Plot : public PlottableInterface, public SlicePlotBase {

public:
  enum class PlotType { slice = 1, voxel = 2 };

  Plot(pugi::xml_node plot, PlotType type);

private:
  void set_output_path(pugi::xml_node plot_node);
  void set_basis(pugi::xml_node plot_node);
  void set_origin(pugi::xml_node plot_node);
  void set_width(pugi::xml_node plot_node);
  void set_meshlines(pugi::xml_node plot_node);

public:
  // Add mesh lines to ImageData
  void draw_mesh_lines(ImageData& data) const;
  ImageData create_image() const override;
  void create_voxel() const;

  void create_output() const override;
  void print_info() const override;

  PlotType type_;                 //!< Plot type (Slice/Voxel)
  Position width_;                //!< Axis-aligned width from plot.xml
  PlotBasis basis_;               //!< Basis from plot.xml for slice plots
  int meshlines_width_;           //!< Width of lines added to the plot
  int index_meshlines_mesh_ {-1}; //!< Index of the mesh to draw on the plot
  RGBColor meshlines_color_;      //!< Color of meshlines on the plot
};

/**
 * \class RaytracePlot
 * \brief Base class for plots that generate images through ray tracing.
 *
 * This class serves as a base for plots that create their visuals by tracing
 * rays from a camera through the problem geometry. It inherits from
 * PlottableInterface, ensuring that it provides an implementation for
 * generating output specific to ray-traced visualization. WireframeRayTracePlot
 * and SolidRayTracePlot provide concrete implementations of this class.
 */
class RayTracePlot : public PlottableInterface {
public:
  RayTracePlot() = default;
  RayTracePlot(pugi::xml_node plot);

  // Standard getters. No setting since it's done from XML.
  const Position& camera_position() const { return camera_position_; }
  Position& camera_position() { return camera_position_; }
  const Position& look_at() const { return look_at_; }
  Position& look_at() { return look_at_; }

  const double& horizontal_field_of_view() const
  {
    return horizontal_field_of_view_;
  }
  double& horizontal_field_of_view() { return horizontal_field_of_view_; }

  void print_info() const override;

  const std::array<int, 2>& pixels() const { return pixels_; }
  std::array<int, 2>& pixels() { return pixels_; }

  const Direction& up() const { return up_; }
  Direction& up() { return up_; }

  //! Width of the view for an orthographic projection. A value of C_NONE
  //! selects a perspective projection.
  double orthographic_width() const { return orthographic_width_; }
  double& orthographic_width() { return orthographic_width_; }

  //! Checks whether the camera parameters define a valid view
  //! \return Description of the problem, or an empty string if valid
  std::string check_view() const;

  //! brief Updates the cached camera-to-model matrix after changes to
  //! camera parameters.
  void update_view();

protected:
  Direction camera_x_axis() const
  {
    return {camera_to_model_[0], camera_to_model_[3], camera_to_model_[6]};
  }

  Direction camera_y_axis() const
  {
    return {camera_to_model_[1], camera_to_model_[4], camera_to_model_[7]};
  }

  Direction camera_z_axis() const
  {
    return {camera_to_model_[2], camera_to_model_[5], camera_to_model_[8]};
  }

  void set_output_path(pugi::xml_node plot_node);

  /*
   * Gets the starting position and direction for the pixel corresponding
   * to this horizontal and vertical position.
   */
  std::pair<Position, Direction> get_pixel_ray(int horiz, int vert) const;

private:
  void set_look_at(pugi::xml_node node);
  void set_camera_position(pugi::xml_node node);
  void set_field_of_view(pugi::xml_node node);
  void set_pixels(pugi::xml_node node);
  void set_orthographic_width(pugi::xml_node node);

  double horizontal_field_of_view_ {70.0}; // horiz. f.o.v. in degrees
  Position camera_position_;               // where camera is
  Position look_at_;                     // point camera is centered looking at
  std::array<int, 2> pixels_ {100, 100}; // pixel dimension of resulting image
  Direction up_ {0.0, 0.0, 1.0};         // which way is up

  /* The horizontal thickness, if using an orthographic projection.
   * If set to zero, we assume using a perspective projection.
   */
  double orthographic_width_ {C_NONE};

  /*
   * Cached camera-to-model matrix with column vectors of axes. The x-axis is
   * the vector between the camera_position_ and look_at_; the y-axis is the
   * cross product of the x-axis with the up_ vector, and the z-axis is the
   * cross product of the x and y axes.
   */
  std::array<double, 9> camera_to_model_;
};

class ProjectionRay;

/**
 * \class WireframeRayTracePlot
 * \brief Creates plots that are like colorful x-ray imaging
 *
 * WireframeRayTracePlot is a specialized form of RayTracePlot designed for
 * creating projection plots. This involves tracing rays from a camera through
 * the problem geometry and rendering the results based on depth of penetration
 * through materials or cells and their colors.
 */
class WireframeRayTracePlot : public RayTracePlot {

  friend class ProjectionRay;

public:
  WireframeRayTracePlot(pugi::xml_node plot);

  ImageData create_image() const override;
  void create_output() const override;
  void print_info() const override;

private:
  void set_opacities(pugi::xml_node node);
  void set_wireframe_thickness(pugi::xml_node node);
  void set_wireframe_ids(pugi::xml_node node);
  void set_wireframe_color(pugi::xml_node node);

  /* Checks if a vector of two TrackSegments is equivalent. We define this
   * to mean not having matching intersection lengths, but rather having
   * a matching sequence of surface/cell/material intersections.
   */
  struct TrackSegment;
  bool trackstack_equivalent(const vector<TrackSegment>& track1,
    const vector<TrackSegment>& track2) const;

  /* Used for drawing wireframe and colors. We record the list of
   * surface/cell/material intersections and the corresponding lengths as a ray
   * traverses the geometry, then color by iterating in reverse.
   */
  struct TrackSegment {
    int id;        // material or cell ID (which is being colored)
    double length; // length of this track intersection

    /* Recording this allows us to draw edges on the wireframe. For instance
     * if two surfaces bound a single cell, it allows drawing that sharp edge
     * where the surfaces intersect.
     */
    int surface_index {-1}; // last surface index intersected in this segment
    TrackSegment(int id_a, double length_a, int surface_a)
      : id(id_a), length(length_a), surface_index(surface_a)
    {}
  };

  // which color IDs should be wireframed. If empty, all cells are wireframed.
  vector<int> wireframe_ids_;

  // Thickness of the wireframe lines. Can set to zero for no wireframe.
  int wireframe_thickness_ {1};

  RGBColor wireframe_color_ {BLACK}; // wireframe color
  vector<double> xs_; // macro cross section values for cell volume rendering
};

/**
 * \class SolidRayTracePlot
 * \brief Plots 3D objects as the eye might see them.
 *
 * Plots a geometry with single-scattered Phong lighting plus a diffuse lighting
 * contribution. The result is a physically reasonable, aesthetic 3D view of a
 * geometry.
 */
class SolidRayTracePlot : public RayTracePlot {
  friend class PhongRay;

public:
  SolidRayTracePlot() = default;

  SolidRayTracePlot(pugi::xml_node plot);

  ImageData create_image() const override;
  void create_output() const override;
  void print_info() const override;

  const std::unordered_set<int>& opaque_ids() const { return opaque_ids_; }
  std::unordered_set<int>& opaque_ids() { return opaque_ids_; }

  const Position& light_location() const { return light_location_; }
  Position& light_location() { return light_location_; }

  const double& diffuse_fraction() const { return diffuse_fraction_; }
  double& diffuse_fraction() { return diffuse_fraction_; }

  //! Opacity of each domain, indexed like colors_. Domains without an entry
  //! are fully opaque.
  const vector<double>& alphas() const { return alphas_; }
  vector<double>& alphas() { return alphas_; }

  //! Opacity of the domain with the given index
  double domain_alpha(int32_t index) const
  {
    return index >= 0 && static_cast<size_t>(index) < alphas_.size()
             ? alphas_[index]
             : 1.0;
  }

  //! Color of the domain with the given index
  RGBColor domain_color(int32_t index) const;

  //! Whether the domain with the given index is drawn
  bool domain_visible(int32_t index) const
  {
    return index >= 0 && opaque_ids_.count(index) > 0 &&
           domain_alpha(index) > 0.0;
  }

  //! User ID of the cell or material with the given domain index
  int32_t domain_id(int32_t index) const;

  //! Corners of the axis-aligned box outside of which geometry is cut away
  const Position& clip_lower() const { return clip_lower_; }
  Position& clip_lower() { return clip_lower_; }
  const Position& clip_upper() const { return clip_upper_; }
  Position& clip_upper() { return clip_upper_; }

  //! Whether surfaces where the clip box cuts through visible domains are
  //! drawn
  bool show_cut_faces() const { return show_cut_faces_; }
  bool& show_cut_faces() { return show_cut_faces_; }

  //! Intersects a ray with the clip box
  //! \param[in] r Origin of the ray
  //! \param[in] u Direction of the ray
  //! \param[out] t_enter Distance where the ray enters the box, at least 0
  //! \param[out] t_exit Distance where the ray leaves the box
  //! \param[out] axis Axis of the face through which the ray enters the box,
  //!   or -1 if the origin is inside the box
  //! \return Whether the ray passes through the box
  bool clip_ray(const Position& r, const Direction& u, double& t_enter,
    double& t_exit, int& axis) const;

  //! Renders the image along with auxiliary per-pixel data. All buffers are
  //! row-major with the top row first and may be null.
  //! \param[out] rgb Colors, 3 values per pixel
  //! \param[out] depth Distance from the origin of the pixel's ray to the
  //!   first visible surface, infinite where nothing visible is hit
  //! \param[out] ids ID of the first visible cell or material, -1 where
  //!   nothing visible is hit and -2 where the ray found undefined geometry
  //! \param[out] normals Normal of the first visible surface, 3 values per
  //!   pixel, zero where nothing visible is hit
  //! \return Number of rays that found undefined geometry
  int64_t render(
    uint8_t* rgb, float* depth, int32_t* ids, float* normals) const;

private:
  void set_opaque_ids(pugi::xml_node node);
  void set_light_position(pugi::xml_node node);
  void set_diffuse_fraction(pugi::xml_node node);

  // Renders the image and counts the rays that found undefined geometry
  ImageData render_image(int64_t& n_undefined) const;

  std::unordered_set<int> opaque_ids_;

  vector<double> alphas_;

  double diffuse_fraction_ {0.1};

  // By default, the light is at the camera unless otherwise specified.
  Position light_location_;

  Position clip_lower_ {-INFINITY, -INFINITY, -INFINITY};
  Position clip_upper_ {INFINITY, INFINITY, INFINITY};
  bool show_cut_faces_ {true};
};

class ProjectionRay : public Ray {
public:
  ProjectionRay(Position r, Direction u, const WireframeRayTracePlot& plot,
    vector<WireframeRayTracePlot::TrackSegment>& line_segments)
    : Ray(r, u), plot_(plot), line_segments_(line_segments)
  {}

  void on_intersection() override;

private:
  /* Store a reference to the plot object which is running this ray, in order
   * to access some of the plot settings which influence the behavior where
   * intersections are.
   */
  const WireframeRayTracePlot& plot_;

  /* The ray runs through the geometry, and records the lengths of ray segments
   * and cells they lie in along the way.
   */
  vector<WireframeRayTracePlot::TrackSegment>& line_segments_;
};

class PhongRay : public Ray {
public:
  PhongRay(Position r, Direction u, const SolidRayTracePlot& plot)
    : Ray(r, u), plot_(plot), origin_(r), direction_(u)
  {}

  //! Traces the ray through the clip box of the plot and shades it
  void render();

  void on_intersection() override;
  bool on_exit_model() override;
  void mark_as_lost(const char* message) override;
  using Ray::mark_as_lost;

  //! Color of the pixel
  RGBColor result_color() const;

  //! Distance from the origin of the ray to the first visible surface, or
  //! infinity if no visible surface was hit
  double depth() const { return depth_; }

  //! Domain index of the first visible surface, or C_NONE
  int32_t hit_domain() const { return hit_domain_; }

  //! Normal of the first visible surface, facing the camera
  const Direction& normal() const { return normal_; }

  //! Whether the ray found undefined geometry before leaving the clip box
  bool found_undefined() const { return lost() || gap_; }

private:
  // Index of the material or cell, depending on the color_by setting of the
  // plot, at the current position
  int32_t domain_index() const;

  // Whether the camera ray has passed the far side of the clip box
  bool beyond_clip_box() const
  {
    return (r() - origin_).dot(direction_) - TINY_BIT > t_max_;
  }

  // Normal of the surface that was just crossed, facing against the ray
  Direction surface_normal();

  // Adds a surface with the given opacity behind what has been composited so
  // far
  void composite(const RGBColor& color, double modulation, double alpha);

  // Shades a surface with the given normal at the current position
  void shade(int32_t domain, const Direction& normal, double t);

  // Handles intersections of the ray that checks for shadows
  void shadow_intersection();

  // Finishes the shading of an opaque surface once the shadow ray is done
  void finish_shadow(bool lit);

  // Checks for undefined regions when the ray enters the model from the void
  void entered_model();

  const SolidRayTracePlot& plot_;

  // Origin and direction of the camera ray, used to measure depth
  Position origin_;
  Direction direction_;

  // Distance along the camera ray at which it leaves the clip box
  double t_max_ {INFINITY};

  // Composited color and opacity, accumulated from front to back
  array<double, 3> color_ {0.0, 0.0, 0.0};
  double opacity_ {0.0};

  // Domain the ray is currently in. Boundaries between cells of the same
  // domain are not drawn.
  int32_t current_domain_ {C_NONE};

  // Data of the first visible surface
  double depth_ {INFINITY};
  int32_t hit_domain_ {C_NONE};
  Direction normal_ {0.0, 0.0, 0.0};

  // State used to detect undefined regions which are enclosed by the model.
  // A ray that leaves the model through a surface without a boundary
  // condition and enters it again through such a surface has passed through
  // a region where particles would get lost.
  bool in_void_ {false};
  bool exited_open_ {false};
  bool start_in_void_ {false};
  Position start_;
  bool gap_ {false};

  // State of the ray that checks whether an opaque surface is shadowed. The
  // ray is pointed from the surface to the light and stops at the first
  // opaque surface, at the light, or where it leaves the clip box.
  bool shadow_ {false};
  Position shadow_origin_;
  double shadow_t_max_ {INFINITY};
  RGBColor pending_color_;
  double pending_modulation_ {1.0};
};

//===============================================================================
// Non-member functions
//===============================================================================

/* Write a PPM image
 * filename - name of output file
 * data - image data to write
 */
void output_ppm(const std::string& filename, const ImageData& data);

#ifdef USE_LIBPNG
/* Write a PNG image
 * filename - name of output file
 * data - image data to write
 */
void output_png(const std::string& filename, const ImageData& data);
#endif

//! Initialize a voxel file
//! \param[in] id of an open hdf5 file
//! \param[in] dimensions of the voxel file (dx, dy, dz)
//! \param[out] dataspace pointer to voxel data
//! \param[out] dataset pointer to voxesl data
//! \param[out] pointer to memory space of voxel data
void voxel_init(hid_t file_id, const hsize_t* dims, hid_t* dspace, hid_t* dset,
  hid_t* memspace);

//! Write a section of the voxel data to hdf5
//! \param[in] voxel slice
//! \param[out] dataspace pointer to voxel data
//! \param[out] dataset pointer to voxesl data
//! \param[out] pointer to data to write
void voxel_write_slice(
  int x, hid_t dspace, hid_t dset, hid_t memspace, void* buf);

//! Close voxel file entities
//! \param[in] data space to close
//! \param[in] dataset to close
//! \param[in] memory space to close
void voxel_finalize(hid_t dspace, hid_t dset, hid_t memspace);

//===============================================================================
// External functions
//===============================================================================

//! Read plot specifications from a plots.xml file
void read_plots_xml();

//! Read plot specifications from an XML Node
//! \param[in] XML node containing plot info
void read_plots_xml(pugi::xml_node root);

//! Clear memory
void free_memory_plot();

//! Create a randomly generated RGB color
//! \return RGBColor with random value
RGBColor random_color();

} // namespace openmc
#endif // OPENMC_PLOT_H
