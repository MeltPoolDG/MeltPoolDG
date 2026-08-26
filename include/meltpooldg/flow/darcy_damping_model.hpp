#pragma once

#include <deal.II/base/vectorization.h>

#include <meltpooldg/flow/darcy_damping_data.hpp>


namespace MeltPoolDG::Flow
{
  /**
   * @brief This class computes the Darcy damping coefficient \f$ K \f$ given by the Kozeny-Carman equation:
   *
   * \f$ K= -C \frac{f_s^2}{(1-f_s)^3 + b} \f$
   *
   * with the solid fraction \f$ f_s \in [0, 1] \f$, the morphology of the mushy zone \f$ C \f$ and
   * the parameter \f$ b \f$ to avoid division by zero.
   *
   * @note The regularization constant \f$ b \f$ must be greater than zero.
   *
   * Voller, V. R., & Prakash, C. (1987). A fixed grid numerical modelling methodology for
   * convection-diffusion mushy region phase-change problems. International Journal of Heat and Mass
   * Transfer, 30(8), 1709–1719. https://doi.org/10.1016/0017-9310(87)90317-6
   *
   * @tparam number Floating-point type
   */
  template <typename number>
  class DarcyDampingModel
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param data_in Input parameters related to darcy damping.
     */
    explicit DarcyDampingModel(const DarcyDampingData<number> &data_in);

    /**
     * @brief Compute the Darcy damping coefficient based on a given @p solid_fraction.
     *
     * @tparam number2 Enables both scalar and vectorized computations.
     *
     * @param solid_fraction Volume fraction of solid phase between 0 and 1.
     *
     * @return Darcy damping coefficient.
     */
    template <typename number2>
    [[nodiscard]] number2
    compute_darcy_damping_coefficient(const number2 &solid_fraction) const;

  private:
    /// Morphological constant C.
    const number mushy_zone_morphology;

    /// Small constant b to avoid division by zero.
    const number avoid_div_zero_constant;
  };
} // namespace MeltPoolDG::Flow
