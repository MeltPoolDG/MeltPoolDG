#pragma once

#include <deal.II/base/array_view.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/non_matching/mapping_info.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <meltpooldg/utilities/enum.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace MeltPoolDG::CutUtil
{
  /**
   * @brief Enum that names a type of CutFEM in use.
   * - not_cut: no CutFEM in use, the field in continuous
   * - one_phase_cut: The domain of the field is restricted to the liquid domain, i.e., positive
   *                  level set values. CutFEM is used for cells that contain the interface.
   * - two_phase_cut: The domain is cut at the interface with the liquid domain
   */
  BETTER_ENUM(CutPhaseType, char, not_cut, one_phase_cut, two_phase_cut)

  /**
   * @brief Determine the CutPhaseType of the @p dof_handler.
   *
   * @tparam dim Dimension in which this function is to be used.
   *
   * @param dof_handler Given DoF-Handler.
   *
   * @return CutPhaseType of the considered @p dof_handler.
   */
  template <int dim>
  CutPhaseType
  get_cut_type(const dealii::DoFHandler<dim> &dof_handler);

  /**
   * Definition of aliases for dealii::MappingInfo types.
   */
  template <int dim, typename number>
  using MappingInfoType =
    dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>>;

  template <int dim, typename number>
  using MappingInfoVectorType = std::vector<std::shared_ptr<MappingInfoType<dim, number>>>;

  /**
   * @brief Definition of the cell category numbering (active FE index).
   */
  enum CellCategory
  {
    liquid      = 0,
    intersected = 1,
    gas         = 2
  };

  /**
   * @brief Enumeration for the face type.
   */
  enum FaceType
  {
    inside_face_liquid,
    inside_face_gas,
    intersected_face,
    mixed_face_liquid_intersected, // "inside" cell in liquid phase; "outside" cell intersected
    mixed_face_intersected_liquid, // "inside" cell intersected; "outside" cell in liquid phase
    mixed_face_gas_intersected,    // "inside" cell in gas phase; "outside" cell intersected
    mixed_face_intersected_gas     // "inside" cell intersected; "outside" cell in gas phase
  };

  /**
   * @brief This function categorizes the FaceType of the current face range.
   *
   * @param adjacent_cell_categories Pair which contains the category of the cells
   * on the two sides of the current range of faces.
   *
   * @retrun FaceType of the shared face between the provided cell pair.
   */
  FaceType
  get_face_type(const std::pair<unsigned int, unsigned int> &adjacent_cell_categories);

  /**
   * @brief This function is setting the FE index for every cell.
   *
   * @tparam dim Dimension in which this function is to be used.
   *
   * @param dof_handler DoFHandler object, provides the cell iterator.
   * @param mesh_classifier The dealii::NonMatching::MeshClassifier object which contains the
   * information, how the active cells and faces of the triangulation are related to the
   * sign of a level set function. Note that the reclassify() function has to be called
   * before, so that the cell and face locations are categorized as one of the values of
   * dealii::NonMatching::LocationToLevelSet: inside, outside or intersected.
   * @param set_future Criteria, whether the active FE index should be set (false) or
   * the future FE index should be set (true) in this function.
   */
  template <int dim>
  void
  set_fe_index(const dealii::DoFHandler<dim>                  &dof_handler,
               const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier,
               const bool                                      set_future);

  /**
   * Return true if at least one cell (on any MPI rank) changed its location with respect to the
   * level set between @p mesh_classifier_old and @p mesh_classifier.
   *
   * Since the FE index of a cell (liquid / intersected / gas) is determined solely by its
   * location (see CutUtil::set_fe_index), an unchanged classification on a fixed mesh implies an
   * unchanged DoF layout, i.e. the same number of DoFs, the same DoF numbering and parallel
   * partitioning, and the same coupling (sparsity) structure.
   *
   * @param tria The triangulation whose cells are checked for changes in their classification.
   * @param mesh_classifier_old The mesh classifier from the previous state, used as the reference
   *   classification.
   * @param mesh_classifier The current mesh classifier against which @p mesh_classifier_old is
   *   compared.
   * @param mpi_comm The MPI communicator over which the result is reduced, so that the function
   *   returns @p true if a change occurred on any MPI rank.
   *
   * @return @p true if at least one cell changed its location with respect to the level set;
   *   @p false otherwise.
   */
  template <int dim>
  bool
  cell_classifications_changed(const dealii::Triangulation<dim>               &tria,
                               const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier_old,
                               const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier,
                               const MPI_Comm                                  mpi_comm);

  /**
   * @brief This function generates the immersed quadrature rules in the case that the
   * domain is described by a discrete level-set function.
   *
   * The class NonMatching::DiscreteQuadratureGenerator<dim>
   * is used to generate the immersed
   * quadrature rules, for the immersed geometry described by discrete level-set function
   * via a (@p level_set_dof_handler, @p level_set) pair.
   *
   * @tparam dim Dimension in which this function is to be used.
   * @tparam number Floating-point value type used for this function.
   * @tparam VectorType Vector type used for this function.
   *
   * @param level_set_dof_handler DoFHandler object of the discrete level-set.
   * @param level_set Discrete level-set vector.
   * @param matrix_free Matrix free object, provides the iterators.
   * @param mapping_info_cells Vector of dealii::NonMatching::MappingInfo objects, provides
   * the mapping information computation and mapping data storage of the cells on the
   * inner subdomain and the outer subdomain, respectively.
   * @param mapping_info_surface dealii::NonMatching::MappingInfo object, provides the mapping
   * information computation and mapping data storage of the surface.
   * @param fe_degree
   * @param is_two_phase
   * @param is_dg Consider FE_DGQ elements?
   * @param mapping_info_faces Vector of dealii::NonMatching::MappingInfo objects, provides
   * the mapping information computation and mapping data storage of the faces on the
   * inner subdomain and the outer subdomain, respectively.
   *
   * @note Currently, the @p fe_degree has to be the same for both subdomains in
   * the case of a two-domain problem.
   * @note Keep attention of the definition of the level-set field orientation. Currently, the single-phase region
   * is assigned to the positive level-set region.
   *
   */
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
    const bool                                                              is_dg = false,
    MappingInfoVectorType<dim, number> mapping_info_faces                         = {});



  /**
   * @brief This function generates the immersed quadrature rules in the case that the
   * domain is described by a discrete level-set function.
   *
   * It is an overload of the above function, which is used when the cut quadrature rules are only
   * needed for the surface integrals.
   *
   * The class NonMatching::DiscreteQuadratureGenerator<dim> is used to generate the immersed
   * quadrature rules, for the immersed geometry described by discrete level-set function
   * via a (@p level_set_dof_handler, @p level_set) pair.
   *
   * @tparam dim Dimension in which this function is to be used.
   * @tparam number Floating-point value type used for this function.
   * @tparam VectorType Vector type used for this function.
   *
   * @param mapping_info_surface dealii::NonMatching::MappingInfo object, provides the mapping
   * information computation and mapping data storage of the surface.
   * @param level_set_dof_handler DoFHandler object of the discrete level-set.
   * @param level_set Discrete level-set vector.
   * @param matrix_free Matrix free object, provides the iterators.
   * @param fe_degree
   *
   * @note Currently, the @p fe_degree has to be the same for both subdomains in
   * the case of a two-domain problem.
   * @note Keep attention of the definition of the level-set field orientation. Currently, the single-phase region
   * is assigned to the positive level-set region.
   *
   */
  template <int dim, typename number, typename VectorType>
  void
  compute_immersed_surface_quadrature(
    MappingInfoType<dim, number>                                           &mapping_info_surface,
    const dealii::DoFHandler<dim>                                          &level_set_dof_handler,
    const VectorType                                                       &level_set,
    const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free,
    const int                                                               fe_degree);


  /**
   * @brief Evaluates solution values in the intersected domain for a specific SIMD lane within a cell batch.
   *
   * This function performs finite element point evaluation on a single lane (i.e., SIMD vector
   * entry) of a cell batch in the context of vectorized matrix-free computations.
   *
   * It uses the given `FECellIntegrator` to access DoF values and performs an evaluation
   * at the point corresponding to the provided lane index.
   *
   * @tparam dim Dimension in which this function is to be used.
   * @tparam number Floating-point value type used for this function.
   * @tparam n_components Number of solution components (per phase). The default value is 1, which corresponds to
   * scalar problems.
   *
   * @param point_eval FEPointEvaluation object in which the evaluations are stored.
   * @param cell_eval FECellIntegrator providing the DoF values.
   * @param evaluation_flags Flags indicating what kind of evaluation (values, gradients, etc.) to perform.
   * @param cell_batch Cell batch index
   * @param cell_lane The SIMD lane within the batch to evaluate.
   * @param n_dofs_per_cell Number of DoFs per cell in the evaluated system.
   */
  template <int dim, typename number, int n_components = 1>
  inline void
  evaluate_intersected_domain(
    dealii::FEPointEvaluation<n_components, dim, dim, dealii::VectorizedArray<number>> &point_eval,
    const FECellIntegrator<dim, n_components, number>                                  &cell_eval,
    const dealii::EvaluationFlags::EvaluationFlags evaluation_flags,
    const unsigned int                             cell_batch,
    const unsigned int                             cell_lane,
    const unsigned int                             n_dofs_per_cell)
  {
    static constexpr unsigned int n_lanes = dealii::VectorizedArray<number>::size();

    point_eval.reinit(cell_batch * n_lanes + cell_lane);
    point_eval.evaluate(dealii::StridedArrayView<const number, n_lanes>(
                          &cell_eval.begin_dof_values()[0][cell_lane], n_dofs_per_cell),
                        evaluation_flags);
  }

  /**
   * @brief This function checks whether the currently considered face requires the application
   * of ghost-penalty stabilization or not.
   *
   * A face is a ghost-penalty face if one of the two adjacent cells is an intersected cell and
   * the other cell is either inside the active domain or is also an intersected cell.
   *
   * @tparam dim Dimension in which this function is to be used.
   *
   * @param mesh_classifier The NonMatching::MeshClassifier object which contains the
   * information, how the active cells and faces of the triangulation are related to the
   * sign of a level set function. Note that the reclassify() function has to be called
   * before, so that the cell and face locations are categorized as one of the values of
   * LocationToLevelSet: inside, outside or intersected.
   * @param cell The considered cell for the function evaluation.
   * @param face_index The index of the considered face, for which the check is done,
   * whether this face is a ghost-penalty-face or not.
   * @param inactive_location Location of the domain, which is not active (location for
   * which a FE_Nothing element is set).
   *
   * @note For multiple component problems, the @p inactive_location has to be chosen
   * carefully at the relevant code section, at which this function is called, as the
   * @p inactive_location depends on the currently considered component. For single
   * component problems, the default value of the inactive location is outside.
   *
   */
  template <int dim>
  bool
  face_has_ghost_penalty(const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier,
                         const typename dealii::Triangulation<dim>::active_cell_iterator &cell,
                         const unsigned int                             face_index,
                         const dealii::NonMatching::LocationToLevelSet &inactive_location =
                           dealii::NonMatching::LocationToLevelSet::inside)
  {
    if (cell->at_boundary(face_index))
      return false;

    const dealii::NonMatching::LocationToLevelSet cell_location =
      mesh_classifier.location_to_level_set(cell);
    const dealii::NonMatching::LocationToLevelSet neighbor_location =
      mesh_classifier.location_to_level_set(cell->neighbor(face_index));

    if (cell_location == dealii::NonMatching::LocationToLevelSet::intersected and
        neighbor_location != inactive_location)
      return true;

    if (neighbor_location == dealii::NonMatching::LocationToLevelSet::intersected and
        cell_location != inactive_location)
      return true;

    return false;
  }

  /**
   * @brief Checks whether the ghost penalty stabilization applies on a face.
   *
   * The ghost penalty of a phase is applied on faces where that phase is
   * present in the cells on both sides and at least one of them is cut:
   * - intersected faces (both cells cut): `true` for both phases
   * - liquid phase: also `true` on liquid–intersected faces
   *   (mixed_face_liquid_intersected, mixed_face_intersected_liquid)
   * - gas phase: also `true` on gas–intersected faces
   *   (mixed_face_gas_intersected, mixed_face_intersected_gas)
   *
   * For all other face types the function returns `false`.
   *
   * @param face_type        Face type, as returned by CutUtil::get_face_type().
   * @param is_liquid_phase  `true` to query the liquid phase, `false` for the gas phase.
   *
   * @return                 `true` if the ghost penalty must be applied for the given phase on this face.
   */
  constexpr bool
  face_type_has_ghost_penalty(const FaceType face_type, const bool is_liquid_phase)
  {
    if (face_type == FaceType::intersected_face)
      return true;

    if (is_liquid_phase)
      return face_type == FaceType::mixed_face_liquid_intersected or
             face_type == FaceType::mixed_face_intersected_liquid;

    return face_type == FaceType::mixed_face_gas_intersected or
           face_type == FaceType::mixed_face_intersected_gas;
  }

  /**
   * @brief This function checks whether the considered face is a newly created intersected face.
   *
   * This occurs when both adjacent cells of the face are currently intersected cells and
   * were in the inactive location in the old state. This function is created for handling
   * moving interface problems.
   *
   * @tparam dim Dimension in which this function is to be used.
   *
   * @param mesh_classifier_old The NonMatching::MeshClassifier object which contains the
   * information, how the active cells and faces of the triangulation are related to the
   * sign of a level set function at the old state. The cell and face locations are
   * categorized as one of the values of LocationToLevelSet: inside, outside or intersected.
   * @param mesh_classifier The NonMatching::MeshClassifier object which contains the
   * information, how the active cells and faces of the triangulation are related to the
   * sign of a level set function at the current state. The cell and face locations are
   * categorized as one of the values of LocationToLevelSet: inside, outside or intersected.
   * @param cell The considered cell for the function evaluation.
   * @param face_index The index of the considered face, for which the check is done,
   * whether this face is a newly created intersected face or not.
   * @param inactive_location Location of the domain, which is not active (location for
   * which a FE_Nothing element is set).
   *
   * @note For multiple component problems, the @p inactive_location has to be chosen
   * carefully at the relevant code section, at which this function is called, as the
   * @p inactive_location depends on the currently considered component. For single
   * component problems, the default value of the inactive location is outside.
   * It has to be ensured that the @p mesh_classifier_old and @p mesh_classifier are
   * reclassified according to the correct states.
   */
  template <int dim>
  bool
  is_new_intersected_face(const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier,
                          const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier_old,
                          const typename dealii::Triangulation<dim>::active_cell_iterator &cell,
                          const unsigned int                             face_index,
                          const dealii::NonMatching::LocationToLevelSet &inactive_location =
                            dealii::NonMatching::LocationToLevelSet::inside)
  {
    if (cell->at_boundary(face_index))
      return false;

    const dealii::NonMatching::LocationToLevelSet cell_location =
      mesh_classifier.location_to_level_set(cell);
    const dealii::NonMatching::LocationToLevelSet neighbor_location =
      mesh_classifier.location_to_level_set(cell->neighbor(face_index));

    const dealii::NonMatching::LocationToLevelSet cell_location_old =
      mesh_classifier_old.location_to_level_set(cell);
    const dealii::NonMatching::LocationToLevelSet neighbor_location_old =
      mesh_classifier_old.location_to_level_set(cell->neighbor(face_index));

    if (cell_location == dealii::NonMatching::LocationToLevelSet::intersected and
        neighbor_location == dealii::NonMatching::LocationToLevelSet::intersected and
        cell_location_old == inactive_location and neighbor_location_old == inactive_location)
      return true;

    return false;
  }
} // namespace MeltPoolDG::CutUtil
