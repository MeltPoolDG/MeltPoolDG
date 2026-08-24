#include <meltpooldg/cut/util.hpp>
//

#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/non_matching/immersed_surface_quadrature.h>
#include <deal.II/non_matching/quadrature_generator.h>

namespace MeltPoolDG::CutUtil
{
  template <int dim>
  CutPhaseType
  get_cut_type(const dealii::DoFHandler<dim> &dof_handler)
  {
    // Detect whether the DoFHandler is set up for CutFEM by checking if it is in hp-mode.
    if (not dof_handler.has_hp_capabilities())
      return CutPhaseType::not_cut;
    // If so, detect weather it is in one phase or on two phase cut mode. To that, we  check the
    // number of components in the intersected fe collection. If it's 2, the cut mode is two phase.
    const unsigned int n_components_intersected =
      dof_handler.get_fe_collection()[CellCategory::intersected].n_components();
    if (n_components_intersected == 1)
      return CutUtil::CutPhaseType::one_phase_cut;
    else if (n_components_intersected == 2)
      return CutUtil::CutPhaseType::two_phase_cut;

    DEAL_II_NOT_IMPLEMENTED();
    return CutPhaseType::not_cut;
  }

  FaceType
  get_face_type(const std::pair<unsigned int, unsigned int> &adjacent_cell_categories)
  {
    // inside_face_liquid: both adjacent cells to the face are completely inside the liquid phase
    if (adjacent_cell_categories.first == CellCategory::liquid and
        adjacent_cell_categories.second == CellCategory::liquid)
      {
        return inside_face_liquid;
      }
    // inside_face_gas: both adjacent cells to the face are completely inside the gas phase
    else if (adjacent_cell_categories.first == CellCategory::gas and
             adjacent_cell_categories.second == CellCategory::gas)
      {
        return inside_face_gas;
      }
    // intersected_face: both adjacent cells to the face are intersected
    else if (adjacent_cell_categories.first == CellCategory::intersected and
             adjacent_cell_categories.second == CellCategory::intersected)
      {
        return intersected_face;
      }
    // mixed_face_liquid_intersected: the adjacent cell of the face in "inside" direction
    // (indicated by a negative signed distance to the face) is completely inside the liquid phase
    // and the adjacent cell of the face in "outside" direction is intersected
    else if ((adjacent_cell_categories.first == CellCategory::liquid and
              adjacent_cell_categories.second == CellCategory::intersected))
      {
        return mixed_face_liquid_intersected;
      }
    // mixed_face_intersected_liquid: the adjacent cell of the face in "outside" direction
    // (indicated by a positive signed distance to the face) is completely inside the liquid phase
    // and the adjacent cell of the face in "inside" direction is intersected
    else if ((adjacent_cell_categories.first == CellCategory::intersected and
              adjacent_cell_categories.second == CellCategory::liquid))
      {
        return mixed_face_intersected_liquid;
      }
    // mixed_face_gas_intersected: the adjacent cell of the face in "inside" direction
    // (indicated by a negative signed distance to the face) is completely inside the gas phase
    // and the adjacent cell of the face in "outside" direction is intersected
    else if ((adjacent_cell_categories.first == CellCategory::gas and
              adjacent_cell_categories.second == CellCategory::intersected))
      {
        return mixed_face_gas_intersected;
      }
    // mixed_face_intersected_gas: the adjacent cell of the face in "outside" direction
    // (indicated by a positive signed distance to the face) is completely inside the gas phase
    // and the adjacent cell of the face in "inside" direction
    // is intersected
    else
      {
        return mixed_face_intersected_gas;
      }
  }

  template <int dim>
  void
  set_fe_index(const dealii::DoFHandler<dim>                  &dof_handler,
               const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier,
               const bool                                      set_future)
  {
    const auto set_fe_index_lambda = [set_future](const auto &cell, const auto active_fe_index) {
      if (set_future)
        cell->set_future_fe_index(active_fe_index);
      else
        cell->set_active_fe_index(active_fe_index);
    };

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned())
          continue;

        // dealii::NonMatching::MeshClassifier classifies positive level set values as "outside".
        // With our definition of the level set, "outside" correlates with the liquid phase.
        const auto cell_location = mesh_classifier.location_to_level_set(cell);
        if (cell_location == dealii::NonMatching::LocationToLevelSet::outside)
          set_fe_index_lambda(cell, CellCategory::liquid);
        else if (cell_location == dealii::NonMatching::LocationToLevelSet::intersected)
          set_fe_index_lambda(cell, CellCategory::intersected);
        else if (cell_location == dealii::NonMatching::LocationToLevelSet::inside)
          set_fe_index_lambda(cell, CellCategory::gas);
        else
          AssertThrow(false, dealii::ExcMessage("Location not found."));
      }
  }

  template <int dim, typename number, typename VectorType>
  void
  compute_intersected_quadrature(
    MappingInfoVectorType<dim, number>                                      mapping_info_cells,
    MappingInfoType<dim, number>                                           &mapping_info_surface,
    const dealii::DoFHandler<dim>                                          &level_set_dof_handler,
    const VectorType                                                       &level_set,
    const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free,
    const int                                                               fe_degree,
    const bool                                                              is_two_phase,
    const bool                                                              is_dg,
    MappingInfoVectorType<dim, number>                                      mapping_info_faces)
  {
    AssertDimension(mapping_info_cells.size(), is_two_phase ? 2 : 1);

    dealii::hp::QCollection<1> q_collection(dealii::QGauss<1>(fe_degree + 1));

    const unsigned int n_lanes = dealii::VectorizedArray<number>::size();
    const unsigned int n_cell_batches =
      matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();

    // face quadrature is only required for DG and dim>1
    const bool has_face_quadrature = is_dg and dim > 1;

    dealii::NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(
      q_collection, level_set_dof_handler, level_set);

    dealii::NonMatching::DiscreteFaceQuadratureGenerator<dim> face_quadrature_generator(
      q_collection, level_set_dof_handler, level_set);

    std::vector<dealii::Quadrature<dim>>                             quad_vec_cells_liquid_domain;
    std::vector<dealii::Quadrature<dim>>                             quad_vec_cells_gas_domain;
    std::vector<dealii::NonMatching::ImmersedSurfaceQuadrature<dim>> quad_vec_surface;
    std::vector<std::vector<dealii::Quadrature<dim - 1>>>            quad_vec_faces_liquid_domain;
    std::vector<std::vector<dealii::Quadrature<dim - 1>>>            quad_vec_faces_gas_domain;
    std::vector<typename dealii::DoFHandler<dim>::cell_iterator>     vector_cell_iterators;

    const unsigned int reserve_size = n_cell_batches * n_lanes;
    quad_vec_cells_liquid_domain.reserve(reserve_size);
    if (is_two_phase)
      quad_vec_cells_gas_domain.reserve(reserve_size);
    quad_vec_surface.reserve(reserve_size);
    if (has_face_quadrature)
      {
        quad_vec_faces_liquid_domain.resize(reserve_size);
        if (is_two_phase)
          quad_vec_faces_gas_domain.resize(reserve_size);
      }
    vector_cell_iterators.reserve(reserve_size);

    for (unsigned int cell_batch = 0; cell_batch < n_cell_batches; ++cell_batch)
      for (unsigned int lane = 0; lane < n_lanes; ++lane)
        {
          // fill empty lanes with dummy data (first lane is used in this case)
          const unsigned int lane_used_for_quad_generation =
            lane < matrix_free.n_active_entries_per_cell_batch(cell_batch) ? lane : 0;

          const auto cell =
            matrix_free.get_cell_iterator(cell_batch, lane_used_for_quad_generation);
          vector_cell_iterators.push_back(cell);
          quadrature_generator.generate(cell);

          if (has_face_quadrature)
            {
              for (const auto face : dealii::GeometryInfo<dim>::face_indices())
                {
                  face_quadrature_generator.generate(cell, face);
                  // Currently, the single-phase region is assigned to the positive level-set
                  // region.
                  quad_vec_faces_liquid_domain[cell_batch * n_lanes + lane].push_back(
                    face_quadrature_generator.get_outside_quadrature());
                  if (is_two_phase)
                    quad_vec_faces_gas_domain[cell_batch * n_lanes + lane].push_back(
                      face_quadrature_generator.get_inside_quadrature());
                }
            }

          quad_vec_cells_liquid_domain.push_back(quadrature_generator.get_outside_quadrature());
          if (is_two_phase)
            quad_vec_cells_gas_domain.push_back(quadrature_generator.get_inside_quadrature());

          quad_vec_surface.push_back(quadrature_generator.get_surface_quadrature());
        }

    mapping_info_cells[0]->reinit_cells(vector_cell_iterators, quad_vec_cells_liquid_domain);
    if (is_two_phase)
      mapping_info_cells[1]->reinit_cells(vector_cell_iterators, quad_vec_cells_gas_domain);

    mapping_info_surface.reinit_surface(vector_cell_iterators, quad_vec_surface);

    if (has_face_quadrature)
      {
        mapping_info_faces[0]->reinit_faces(vector_cell_iterators, quad_vec_faces_liquid_domain);
        if (is_two_phase)
          mapping_info_faces[1]->reinit_faces(vector_cell_iterators, quad_vec_faces_gas_domain);
      }
  }

  template CutPhaseType
  get_cut_type(const dealii::DoFHandler<1> &dof_handler);
  template CutPhaseType
  get_cut_type(const dealii::DoFHandler<2> &dof_handler);
  template CutPhaseType
  get_cut_type(const dealii::DoFHandler<3> &dof_handler);

  template void
  set_fe_index<1>(const dealii::DoFHandler<1> &,
                  const dealii::NonMatching::MeshClassifier<1> &,
                  const bool);
  template void
  set_fe_index<2>(const dealii::DoFHandler<2> &,
                  const dealii::NonMatching::MeshClassifier<2> &,
                  const bool);
  template void
  set_fe_index<3>(const dealii::DoFHandler<3> &,
                  const dealii::NonMatching::MeshClassifier<3> &,
                  const bool);

  template <int dim, typename number, typename VectorType>
  void
  compute_immersed_surface_quadrature(
    MappingInfoType<dim, number>                                           &mapping_info_surface,
    const dealii::DoFHandler<dim>                                          &level_set_dof_handler,
    const VectorType                                                       &level_set,
    const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free,
    const int                                                               fe_degree)
  {
    dealii::hp::QCollection<1> q_collection(dealii::QGauss<1>(fe_degree + 1));

    const unsigned int n_lanes = dealii::VectorizedArray<number>::size();
    const unsigned int n_cell_batches =
      matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();

    dealii::NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(
      q_collection, level_set_dof_handler, level_set);

    std::vector<dealii::NonMatching::ImmersedSurfaceQuadrature<dim>> quad_vec_surface;
    std::vector<typename dealii::DoFHandler<dim>::cell_iterator>     vector_cell_iterators;
    {
      const unsigned int reserve_size = n_cell_batches * n_lanes;
      vector_cell_iterators.reserve(reserve_size);
    }

    for (unsigned int cell_batch = 0; cell_batch < n_cell_batches; ++cell_batch)
      for (unsigned int lane = 0; lane < n_lanes; ++lane)
        {
          // fill empty lanes with dummy data (first lane is used in this case)
          const unsigned int lane_used_for_quad_generation =
            lane < matrix_free.n_active_entries_per_cell_batch(cell_batch) ? lane : 0;

          const auto cell =
            matrix_free.get_cell_iterator(cell_batch, lane_used_for_quad_generation);

          vector_cell_iterators.push_back(cell);
          quadrature_generator.generate(cell);

          quad_vec_surface.push_back(quadrature_generator.get_surface_quadrature());
        }

    mapping_info_surface.reinit_surface(vector_cell_iterators, quad_vec_surface);
  }

  template void
  compute_intersected_quadrature(
    std::vector<
      std::shared_ptr<dealii::NonMatching::MappingInfo<1, 1, dealii::VectorizedArray<double>>>>,
    dealii::NonMatching::MappingInfo<1, 1, dealii::VectorizedArray<double>> &,
    const dealii::DoFHandler<1> &,
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    const dealii::MatrixFree<1, double, dealii::VectorizedArray<double>> &,
    const int,
    const bool,
    const bool,
    std::vector<
      std::shared_ptr<dealii::NonMatching::MappingInfo<1, 1, dealii::VectorizedArray<double>>>>);
  template void
  compute_intersected_quadrature(
    std::vector<
      std::shared_ptr<dealii::NonMatching::MappingInfo<2, 2, dealii::VectorizedArray<double>>>>,
    dealii::NonMatching::MappingInfo<2, 2, dealii::VectorizedArray<double>> &,
    const dealii::DoFHandler<2> &,
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    const dealii::MatrixFree<2, double, dealii::VectorizedArray<double>> &,
    const int,
    const bool,
    const bool,
    std::vector<
      std::shared_ptr<dealii::NonMatching::MappingInfo<2, 2, dealii::VectorizedArray<double>>>>);
  template void
  compute_intersected_quadrature(
    std::vector<
      std::shared_ptr<dealii::NonMatching::MappingInfo<3, 3, dealii::VectorizedArray<double>>>>,
    dealii::NonMatching::MappingInfo<3, 3, dealii::VectorizedArray<double>> &,
    const dealii::DoFHandler<3> &,
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    const dealii::MatrixFree<3, double, dealii::VectorizedArray<double>> &,
    const int,
    const bool,
    const bool,
    std::vector<
      std::shared_ptr<dealii::NonMatching::MappingInfo<3, 3, dealii::VectorizedArray<double>>>>);


  template void
  compute_immersed_surface_quadrature(
    dealii::NonMatching::MappingInfo<1, 1, dealii::VectorizedArray<double>> &,
    const dealii::DoFHandler<1> &,
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    const dealii::MatrixFree<1, double, dealii::VectorizedArray<double>> &,
    const int);
  template void
  compute_immersed_surface_quadrature(
    dealii::NonMatching::MappingInfo<2, 2, dealii::VectorizedArray<double>> &,
    const dealii::DoFHandler<2> &,
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    const dealii::MatrixFree<2, double, dealii::VectorizedArray<double>> &,
    const int);
  template void
  compute_immersed_surface_quadrature(
    dealii::NonMatching::MappingInfo<3, 3, dealii::VectorizedArray<double>> &,
    const dealii::DoFHandler<3> &,
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    const dealii::MatrixFree<3, double, dealii::VectorizedArray<double>> &,
    const int);
} // namespace MeltPoolDG::CutUtil
