#pragma once

#include <deal.II/base/parameter_handler.h>

namespace MeltPoolDG
{
  /**
   * @brief Collection of parameters for the ghost-penalty stabilization of cutFEM and cutDG
   * applications.
   */
  template <typename number>
  struct GhostPenaltyData
  {
    // mass matrix stabilization

    /// Ghost-penalty parameter for penalization of jumps in the values on the ghost-faces
    number gamma_M_degree_0 = 1.;
    /// Ghost-penalty parameter for penalization of jumps in the normal gradients on the ghost-faces
    number gamma_M_degree_1 = 1.;
    /// Ghost-penalty parameter for penalization of jumps in the normal hessians on the ghost-faces
    /// (only relevant for polynomial degree = 2)
    number gamma_M_degree_2 = 1.;

    // stiffness matrix stabilization

    /// Ghost-penalty parameter for penalization of jumps in the values on the ghost-faces
    number gamma_A_degree_0 = 1.;
    /// Ghost-penalty parameter for penalization of jumps in the normal gradients on the ghost-faces
    number gamma_A_degree_1 = 1.;
    /// Ghost-penalty parameter for penalization of jumps in the normal hessians on the ghost-faces
    /// (only relevant for polynomial degree = 2)
    number gamma_A_degree_2 = 1.;

    /**
     * @brief Add ghost-penalty parameters in the parameter handler.
     *
     * @param prm The parameter handler to which the parameters are added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm);
  };

  /**
   * @brief Collection of parameters for the stabilization of cutFEM and cutDG applications.
   */
  template <typename number>
  struct CutStabilizationData
  {
    /// Nitsche stabilization parameter
    number nitsche_parameter = 1.;

    /// Parameters for ghost-penalty stabilization
    GhostPenaltyData<number> ghost_penalty;

    /**
     * @brief Add cut-related stabilization parameters in the parameter handler.
     *
     * @param prm The parameter handler to which the parameters are added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm);
  };

  /**
   * @brief Collection of parameters for the weighted repartitioning of the triangulation in cut applications.
   *
   * The weight of a cell is chosen according to its location with respect to the zero level-set
   * isosurface, as determined by dealii::NonMatching::MeshClassifier:
   * - inside: the level set is negative on the whole cell,
   * - outside: the level set is positive on the whole cell,
   * - intersected: the zero level-set isosurface intersects the cell.
   */
  struct CutRepartitionData
  {
    /// If true, the triangulation is repartitioned with cell weights according to the location of
    /// the cells with respect to the level set
    bool enable = false;

    /// The triangulation is repartitioned every n-th time step
    unsigned int every_n_step = 100;

    /// Partitioning weights
    /// The default weights are chosen according to the reference:
    /// Bergbauer, Maximilian, et al. "High-performance matrix-free unfitted finite element operator
    /// evaluation." SIAM Journal on Scientific Computing 47.3 (2025): B665-B689. Note that we used
    /// 20 instead of 10 for the weight of intersected cells, since we have a two-phase flow
    /// problem.
    struct Weights
    {
      /// Weight of cells located completely inside (negative level set)
      unsigned int inside = 1;

      /// Weight of cells located completely outside (positive level set)
      unsigned int outside = 1;

      /// Weight of cells intersected by the zero level-set isosurface
      unsigned int intersected = 20;
    } weights;

    /**
     * @brief Add the repartitioning parameters in the parameter handler.
     *
     * @param prm The parameter handler to which the parameters are added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm);
  };
} // namespace MeltPoolDG
