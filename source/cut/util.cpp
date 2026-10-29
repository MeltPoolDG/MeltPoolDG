#include <meltpooldg/cut/util.hpp>
//

#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q_base.h>

#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/non_matching/immersed_surface_quadrature.h>
#include <deal.II/non_matching/mesh_classifier.h>
#include <deal.II/non_matching/quadrature_generator.h>

#include <optional>

namespace MeltPoolDG::CutUtil
{
  template <int dim>
  CutPhaseType
  get_cut_type(const dealii::DoFHandler<dim> &dof_handler)
  {
    // Detect whether the DoFHandler is set up for CutFEM by checking if it is in hp-mode.
    if (not dof_handler.has_hp_capabilities())
      return CutPhaseType::not_cut;
    // If so, detect whether it is in one phase or on two phase cut mode. To that, we  check the
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

  template <int dim>
  bool
  cell_classifications_changed(const dealii::Triangulation<dim>               &tria,
                               const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier_old,
                               const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier,
                               const MPI_Comm                                  mpi_comm)
  {
    bool changed = false;
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned() and mesh_classifier.location_to_level_set(cell) !=
                                         mesh_classifier_old.location_to_level_set(cell))
        {
          changed = true;
          break;
        }

    return dealii::Utilities::MPI::logical_or(changed, mpi_comm);
  }

  template <int dim>
  dealii::NonMatching::LocationToLevelSet
  location_from_quadratures(
    const dealii::NonMatching::QuadratureGenerator<dim> &quadrature_generator)
  {
    const bool has_inside  = not quadrature_generator.get_inside_quadrature().empty();
    const bool has_outside = not quadrature_generator.get_outside_quadrature().empty();
    const bool has_surface = not quadrature_generator.get_surface_quadrature().empty();

    // Consistent with the MeshClassifier, a cell, of which the interface only
    // touches a face, is only considered intersected if it lies on the side of
    // negative level-set values. Thus, an interface coinciding with a cell face is
    // only counted once.
    if (has_inside and (has_outside or has_surface))
      return dealii::NonMatching::LocationToLevelSet::intersected;
    else if (has_inside)
      return dealii::NonMatching::LocationToLevelSet::inside;
    else
      return dealii::NonMatching::LocationToLevelSet::outside;
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

    const bool update_ghosts = not level_set.has_ghost_elements();
    if (update_ghosts)
      level_set.update_ghost_values();

    dealii::hp::QCollection<1> q_collection(dealii::QGauss<1>(fe_degree + 1));

    const unsigned int n_lanes = dealii::VectorizedArray<number>::size();
    const unsigned int n_cell_batches =
      matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();

    // face quadrature is only required for DG and dim>1
    const bool has_face_quadrature = is_dg and dim > 1;

    dealii::NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(
      q_collection, level_set_dof_handler, level_set);

    std::vector<dealii::Quadrature<dim>>                             quad_vec_cells_liquid_domain;
    std::vector<dealii::Quadrature<dim>>                             quad_vec_cells_gas_domain;
    std::vector<dealii::NonMatching::ImmersedSurfaceQuadrature<dim>> quad_vec_surface;
    std::vector<typename dealii::DoFHandler<dim>::cell_iterator>     vector_cell_iterators;

    const unsigned int reserve_size = n_cell_batches * n_lanes;
    quad_vec_cells_liquid_domain.reserve(reserve_size);
    if (is_two_phase)
      quad_vec_cells_gas_domain.reserve(reserve_size);
    quad_vec_surface.reserve(reserve_size);
    vector_cell_iterators.reserve(reserve_size);

    // cell and surface quadrature
    for (unsigned int cell_batch = 0; cell_batch < n_cell_batches; ++cell_batch)
      {
        // check if the cell batch is intersected, all cells in a batch share the same category
        const bool is_intersected_batch =
          matrix_free.get_cell_category(cell_batch) == CellCategory::intersected;

        for (unsigned int lane = 0; lane < n_lanes; ++lane)
          {
            const bool is_active_lane =
              lane < matrix_free.n_active_entries_per_cell_batch(cell_batch);

            // empty lanes of a partially filled cell batch use the cell of the first lane as dummy
            const auto cell = matrix_free.get_cell_iterator(cell_batch, is_active_lane ? lane : 0);
            vector_cell_iterators.push_back(cell);

            // only compute quadrature for intersected cell batches and active lanes of partially
            // filled batches
            if (is_intersected_batch and is_active_lane)
              {
                quadrature_generator.generate(cell);

                // Currently, the single-phase region is assigned to the positive level-set region.
                quad_vec_cells_liquid_domain.push_back(
                  quadrature_generator.get_outside_quadrature());
                if (is_two_phase)
                  quad_vec_cells_gas_domain.push_back(quadrature_generator.get_inside_quadrature());

                quad_vec_surface.push_back(quadrature_generator.get_surface_quadrature());
              }
            else
              {
                // add empty quadratures for non-intersected cell batches and empty lanes of
                // partially filled cell batches
                quad_vec_cells_liquid_domain.emplace_back();
                if (is_two_phase)
                  quad_vec_cells_gas_domain.emplace_back();

                quad_vec_surface.emplace_back();
              }
          }
      }

    mapping_info_cells[0]->reinit_cells(vector_cell_iterators, quad_vec_cells_liquid_domain);
    if (is_two_phase)
      mapping_info_cells[1]->reinit_cells(vector_cell_iterators, quad_vec_cells_gas_domain);

    mapping_info_surface.reinit_surface(vector_cell_iterators, quad_vec_surface);

    // face quadrature
    if (not has_face_quadrature)
      return;

    dealii::NonMatching::DiscreteFaceQuadratureGenerator<dim> face_quadrature_generator(
      q_collection, level_set_dof_handler, level_set);

    std::vector<std::vector<dealii::Quadrature<dim - 1>>> quad_vec_faces_liquid_domain;
    std::vector<std::vector<dealii::Quadrature<dim - 1>>> quad_vec_faces_gas_domain;

    // fill the face quadrature vectors with empty quadratures for all cell batches and lanes
    const std::vector<dealii::Quadrature<dim - 1>> empty_face_quadratures(
      dealii::GeometryInfo<dim>::faces_per_cell);
    quad_vec_faces_liquid_domain.assign(vector_cell_iterators.size(), empty_face_quadratures);
    if (is_two_phase)
      quad_vec_faces_gas_domain.assign(vector_cell_iterators.size(), empty_face_quadratures);

    const auto generate_face_quadrature = [&](const unsigned int cell_index,
                                              const unsigned int face_no) {
      face_quadrature_generator.generate(vector_cell_iterators[cell_index], face_no);

      // Currently, the single-phase region is assigned to the positive level-set region.
      quad_vec_faces_liquid_domain[cell_index][face_no] =
        face_quadrature_generator.get_outside_quadrature();
      if (is_two_phase)
        quad_vec_faces_gas_domain[cell_index][face_no] =
          face_quadrature_generator.get_inside_quadrature();
    };

    const unsigned int n_inner_face_batches    = matrix_free.n_inner_face_batches();
    const unsigned int n_boundary_face_batches = matrix_free.n_boundary_face_batches();
    const unsigned int n_face_batches =
      n_inner_face_batches + n_boundary_face_batches + matrix_free.n_ghost_inner_face_batches();

    for (unsigned int face_batch = 0; face_batch < n_face_batches; ++face_batch)
      {
        const bool is_boundary_face = face_batch >= n_inner_face_batches and
                                      face_batch < n_inner_face_batches + n_boundary_face_batches;

        // all lanes in a face batch share the same face category, which is determined by the
        // categories of the adjacent cells
        const auto face_category = matrix_free.get_face_category(face_batch);

        // Note: For boundary faces, face_category.second is numbers::invalid_unsigned_int.
        // Therefore, boundary faces are checked via the category of the interior cell.
        const bool needs_face_quadrature =
          is_boundary_face ? face_category.first == CellCategory::intersected :
                             get_face_type(face_category) == FaceType::intersected_face;

        if (not needs_face_quadrature)
          continue;

        const auto &face_info = matrix_free.get_face_info(face_batch);

        for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_face_batch(face_batch);
             ++lane)
          {
            generate_face_quadrature(face_info.cells_interior[lane], face_info.interior_face_no);
            if (not is_boundary_face)
              generate_face_quadrature(face_info.cells_exterior[lane], face_info.exterior_face_no);
          }
      }

    mapping_info_faces[0]->reinit_faces(vector_cell_iterators, quad_vec_faces_liquid_domain);
    if (is_two_phase)
      mapping_info_faces[1]->reinit_faces(vector_cell_iterators, quad_vec_faces_gas_domain);

    if (update_ghosts)
      level_set.zero_out_ghost_values();
  }

  template <int dim>
  bool
  cell_has_ghost_penalty_face(const typename dealii::DoFHandler<dim>::cell_iterator &cell,
                              const Phase                                            phase)
  {
    const auto is_ghost_penalty_face = [&](const unsigned int neighbor_category) {
      return category_has_phase(neighbor_category, phase) &&
             (cell->active_fe_index() == CutUtil::CellCategory::intersected or
              neighbor_category == CutUtil::CellCategory::intersected);
    };

    for (const unsigned int f : cell->face_indices())
      {
        const bool is_periodic = cell->has_periodic_neighbor(f);

        // skip boundary faces if the boundary is not periodic, since the ghost penalty is only
        // applied to interior faces
        if (cell->at_boundary(f) and !is_periodic)
          continue;

        const auto neighbor = cell->neighbor_or_periodic_neighbor(f);

        if (!neighbor->has_children())
          {
            // neighbor on the same level or coarser
            if (is_ghost_penalty_face(neighbor->active_fe_index()))
              return true;
          }
        else if constexpr (dim == 1)
          {
            // a face is a point: only the category of the adjacent active child matters
            auto child = neighbor;
            while (child->has_children())
              {
                // Child numbering in 1D: a cell is an interval with face 0 on the left and face 1
                // on the right. When it is refined, child(0) is the left half and child(1) is the
                // right half.
                // The child element touching the shared point is always the one on the opposite
                // side of the face index, so it is child(1 - f).
                child = child->child(1 - f);
              }

            if (is_ghost_penalty_face(child->active_fe_index()))
              return true;
          }
        else
          {
            // finer neighbor: check the subfaces
            // (in 2D and 3D deal.II allows neighbours across a face to differ by at most one
            // refinement level)
            const unsigned int n_subfaces = is_periodic ?
                                              dealii::GeometryInfo<dim>::max_children_per_face
                                              /*2 in 2D and 4 in 3D*/ :
                                              cell->face(f)->n_children();
            for (unsigned int subface = 0; subface < n_subfaces; ++subface)
              {
                const auto child =
                  is_periodic ?
                    cell->periodic_neighbor_child_on_subface(
                      f, subface) /*looks up the periodic neighbor and then the child*/ :
                    cell->neighbor_child_on_subface(f, subface);

                if (is_ghost_penalty_face(child->active_fe_index()))
                  return true;
              }
          }
      }

    return false;
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

  template bool
  cell_classifications_changed(const dealii::Triangulation<1> &,
                               const dealii::NonMatching::MeshClassifier<1> &,
                               const dealii::NonMatching::MeshClassifier<1> &,
                               const MPI_Comm);
  template bool
  cell_classifications_changed(const dealii::Triangulation<2> &,
                               const dealii::NonMatching::MeshClassifier<2> &,
                               const dealii::NonMatching::MeshClassifier<2> &,
                               const MPI_Comm);
  template bool
  cell_classifications_changed(const dealii::Triangulation<3> &,
                               const dealii::NonMatching::MeshClassifier<3> &,
                               const dealii::NonMatching::MeshClassifier<3> &,
                               const MPI_Comm);

  template dealii::NonMatching::LocationToLevelSet
  location_from_quadratures(const dealii::NonMatching::QuadratureGenerator<1> &);
  template dealii::NonMatching::LocationToLevelSet
  location_from_quadratures(const dealii::NonMatching::QuadratureGenerator<2> &);
  template dealii::NonMatching::LocationToLevelSet
  location_from_quadratures(const dealii::NonMatching::QuadratureGenerator<3> &);

  template bool
  cell_has_ghost_penalty_face<1>(const dealii::DoFHandler<1>::cell_iterator &, const Phase);
  template bool
  cell_has_ghost_penalty_face<2>(const dealii::DoFHandler<2>::cell_iterator &, const Phase);
  template bool
  cell_has_ghost_penalty_face<3>(const dealii::DoFHandler<3>::cell_iterator &, const Phase);

  template <int dim, typename number, typename VectorType>
  void
  compute_immersed_surface_quadrature(
    MappingInfoType<dim, number>                                           &mapping_info_surface,
    const dealii::DoFHandler<dim>                                          &level_set_dof_handler,
    const VectorType                                                       &level_set_in,
    const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free,
    const int                                                               fe_degree)
  {
    dealii::hp::QCollection<1> q_collection(dealii::QGauss<1>(fe_degree + 1));

    const unsigned int n_lanes = dealii::VectorizedArray<number>::size();
    const unsigned int n_cell_batches =
      matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();

    // The MeshClassifier is only implemented for elements derived from FE_Q_Base.
    const bool use_mesh_classifier =
      dynamic_cast<const dealii::FE_Q_Base<dim> *>(&level_set_dof_handler.get_fe()) != nullptr;

    // The MeshClassifier reads the level-set values on all faces of the ghost cells, which
    // requires the locally relevant DoFs as ghost entries. However, the ghost set of a vector
    // set up by MatrixFree might only contain the DoFs of locally owned cells. In this case,
    // the level-set field is copied to a vector with a sufficient ghost set.
    VectorType level_set_relevant;
    bool       use_copy = false;
    if (use_mesh_classifier)
      {
        const auto locally_relevant_dofs =
          dealii::DoFTools::extract_locally_relevant_dofs(level_set_dof_handler);

        auto required_ghost_dofs = locally_relevant_dofs;
        required_ghost_dofs.subtract_set(level_set_in.locally_owned_elements());

        const bool has_sufficient_ghosts =
          required_ghost_dofs.is_subset_of(level_set_in.get_partitioner()->ghost_indices());
        // All processes have to take the same decision. Otherwise, a process that does not need
        // the copy (e.g. one without cells) would skip it, and the others would wait for it
        // forever.
        use_copy = dealii::Utilities::MPI::logical_or(not has_sufficient_ghosts,
                                                      level_set_in.get_mpi_communicator());

        if (use_copy)
          {
            level_set_relevant.reinit(level_set_dof_handler.locally_owned_dofs(),
                                      locally_relevant_dofs,
                                      level_set_in.get_mpi_communicator());
            level_set_relevant.copy_locally_owned_data_from(level_set_in);
            level_set_relevant.update_ghost_values();
          }
      }

    const VectorType &level_set = use_copy ? level_set_relevant : level_set_in;

    const bool update_ghosts = not level_set.has_ghost_elements();
    if (update_ghosts)
      level_set.update_ghost_values();

    dealii::NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(
      q_collection, level_set_dof_handler, level_set);

    std::optional<dealii::NonMatching::MeshClassifier<dim>> mesh_classifier;
    if (use_mesh_classifier)
      {
        mesh_classifier.emplace(level_set_dof_handler, level_set);
        mesh_classifier->reclassify();
      }

    std::vector<dealii::NonMatching::ImmersedSurfaceQuadrature<dim>> quad_vec_surface;
    std::vector<typename dealii::DoFHandler<dim>::cell_iterator>     vector_cell_iterators;
    {
      const unsigned int reserve_size = n_cell_batches * n_lanes;
      quad_vec_surface.reserve(reserve_size);
      vector_cell_iterators.reserve(reserve_size);
    }

    for (unsigned int cell_batch = 0; cell_batch < n_cell_batches; ++cell_batch)
      for (unsigned int lane = 0; lane < n_lanes; ++lane)
        {
          const bool is_active_lane =
            lane < matrix_free.n_active_entries_per_cell_batch(cell_batch);

          // Empty lanes of a partially filled cell batch use the cell of the first lane as dummy.
          const auto cell = matrix_free.get_cell_iterator(cell_batch, is_active_lane ? lane : 0);

          vector_cell_iterators.push_back(cell);

          // Early exit for empty lanes of partially filled cell batches.
          if (not is_active_lane)
            {
              quad_vec_surface.emplace_back();
              continue;
            }

          // Determine the location of the cell with respect to the level set. If a MeshClassifier
          // is used, it is queried. Otherwise, the location is determined from the quadratures
          // generated by the QuadratureGenerator.
          dealii::NonMatching::LocationToLevelSet location;
          if (mesh_classifier)
            location = mesh_classifier->location_to_level_set(cell);
          else
            {
              quadrature_generator.generate(cell);
              location = location_from_quadratures(quadrature_generator);
            }

          // Skip the quadrature generation for cells, which are not intersected.
          if (location != dealii::NonMatching::LocationToLevelSet::intersected)
            {
              quad_vec_surface.emplace_back();
              continue;
            }

          if (mesh_classifier)
            quadrature_generator.generate(cell);

          quad_vec_surface.push_back(quadrature_generator.get_surface_quadrature());
        }

    mapping_info_surface.reinit_surface(vector_cell_iterators, quad_vec_surface);

    if (update_ghosts)
      level_set.zero_out_ghost_values();
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
