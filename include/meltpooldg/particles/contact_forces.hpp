#pragma once

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>

#include <meltpooldg/particles/contact_forces_data.hpp>
#include <meltpooldg/particles/dem_util.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/particle_accessor.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>

#include <memory>


namespace MeltPoolDG
{
  template <int dim, typename number, typename ObstacleType>
  class SphericalParticleContactForce
  {
  public:
    /**
     * Constructor for the spherical particle contact force model.
     *
     * @param contact_data Contact data defining the material and contact properties.
     * @param time_iterator Time iterator used in the DEM simulation for which the contact force is applied.
     */
    explicit SphericalParticleContactForce(
      const SphericalParticleContactData<number>              &contact_data,
      const MeltPoolDG::TimeIntegration::TimeIterator<number> &time_iterator);


    /**
     * This function adds a wall defined by a distance function to the contact force
     * computation. The wall is represented by a function that returns the distance from
     * any point in space to the wall surface. The contact normal is obtained from the gradient of
     * this function.
     *
     * @param wall_distance_function A unique pointer to a function representing the
     * distance to the wall.
     *
     * @note The value of the wall function is the distance to the closest point on the wall, and
     * its gradient points from that point towards the evaluation point. Hence, a single wall
     * function acts on both sides of the wall.
     */
    void
    attach_wall(std::unique_ptr<dealii::Function<dim, number>> &&wall_distance_function);

    /**
     * Compute the contact forces and add them to the obstacles in the given obstacle field. This
     * algorithm consists of two steps, first finding contacting particles, then computing and
     * adding the contact forces.
     *
     * For finding contacting particles, a brute-force approach is used where each particle is
     * checked against all other particles in the global particle data structure.
     *
     * For the contact force computation, a nonlinear spring-dashpot model (Hertz contact theory) is
     * used to compute normal contact forces between spherical particles. Currently only normal
     * contact forces are computed; tangential forces and friction are not yet implemented. For
     * details on the implemented model, see e.g. Gaboriault et al. (DOI:10.48550/arXiv.2509.26402).
     *
     * @param obstacle_field The obstacle field containing the particles for which contact forces are to be
     * computed. The resulting forces are added directly to the particles in this field.
     */
    void
    add_load_to_obstacles(ObstacleField<dim, number, ObstacleType> &obstacle_field) const;

  private:
    /// Contact data for the spherical particle contact force model.
    const SphericalParticleContactData<number> &contact_data;

    /// Damping prefactor computed from the restitution coefficient. This is cached for efficiency.
    const number damping_prefactor;

    /// Tangential gap vectors between particles. The key of the outer map identifies the particle
    /// of consideration (self), the inner map the other particle in contact.
    mutable std::map<int, std::map<int, dealii::Tensor<1, dim, number>>> tangential_gaps;

    /// Same as above, but for particle–wall contacts. The outer key corresponds to the particle id,
    /// while the inner key corresponds to the wall id. The stored tangential gap represents the
    /// tangential displacement between the particle and the wall associated with the respective
    /// ids.
    mutable std::map<int, std::map<int, dealii::Tensor<1, dim, number>>> tangential_gaps_with_walls;

    /// Time iterator which is used in the DEM simulation. This is needed to get the current time
    /// step size.
    const MeltPoolDG::TimeIntegration::TimeIterator<number> &time_iterator;

    /// Map of wall distance functions added to the contact model. The key is a unique wall
    /// id to identify each wall.
    std::map<int, std::unique_ptr<dealii::Function<dim, number>>> wall_distance_functions;

    /**
     * Struct describing the contact configuration between two particles, i.e., it computes and
     * caches all relevant contact parameters such as contact velocity, overlaps etc. required in
     * the computation of contact forces.
     */
    struct ContactConfiguration
    {
      /**
       * Constructor which computes all relevant input for the contact model between two particles.
       * Note that the computed normal vector points from @p self to @p other.
       *
       * @param self The first particle in the contact.
       * @param other The second particle in the contact.
       * @param youngs_modulus The Young's modulus of the particle material.
       * @param poisson_ratio The Poisson's ratio of the particle material.
       */
      ContactConfiguration(const DEMParticleAccessor<dim, number> &self,
                           const DEMParticleAccessor<dim, number> &other,
                           const number                            youngs_modulus,
                           const number                            poisson_ratio);

      /**
       * Constructor which computes all relevant input for the particle–wall contact model. The wall
       * is assumed to have infinite mass and stiffness.
       *
       * @param particle The particle involved in the contact.
       * @param wall Pointer to the distance function representing the wall.
       * @param youngs_modulus The Young's modulus of the particle material.
       * @param poisson_ratio The Poisson's ratio of the particle material.
       */
      ContactConfiguration(const DEMParticleAccessor<dim, number> &particle,
                           const dealii::Function<dim>            *wall,
                           const number                            youngs_modulus,
                           const number                            poisson_ratio);

      /// Effective mass of the two contacting particles, i.e., m1*m2/(m1+m2).
      number effective_mass;

      /// Effective radius of the two contacting particles, i.e., R1*R2/(R1+R2).
      number effective_radius;

      /// Effective youngs modulus of the two contacting particles, i.e., E/(2*(1-nu^2)).
      number effective_youngs_modulus;

      /// Effective shear modulus of the two contacting particles, i.e., G/(4*(2-nu)*(1+nu)).
      number effective_shear_modulus;

      /// Normal vector pointing from self to other.
      dealii::Tensor<1, dim, number> normal_vector;

      /// Normal overlap between the two particles (positive if in contact).
      number normal_overlap;

      /// Relative angular velocity between the two particles or the particle and the wall.
      dealii::Tensor<1, axial_dim<dim>, number> relative_angular_velocity;

      /**
       * Struct for describing the relative velocity between two particles at the contact point.
       */
      struct
      {
        /// Full relative velocity vector at the contact point.
        dealii::Tensor<1, dim, number> value;

        /// Normal component of the relative velocity.
        dealii::Tensor<1, dim, number> normal_component;

        /// Tangential component of the relative velocity.
        dealii::Tensor<1, dim, number> tangential_component;
      } relative_velocity;
    };

    /**
     * Computes the normal contact force for a given particle–particle contact. The formulation
     * follows Gaboriault et al. (DOI:10.48550/arXiv.2509.26402).
     *
     * The normal contact force is defined as
     * \f[
     * \boldsymbol{F}_n = k_n\delta_n\boldsymbol{n} + \eta_n\boldsymbol{v}_n,
     * \f]
     * where \f$k_n\f$ is the normal stiffness, \f$\delta_n\f$ the normal overlap,
     * \f$\boldsymbol{n}\f$ the unit normal vector at the contact, \f$\eta_n\f$ the normal damping
     * coefficient, and \f$v_n\f$ the normal component of the relative velocity.
     *
     * The normal stiffness is computed as
     * \f[
     * k_n = \frac{4}{3} Y_e \sqrt{R_e \delta_n},
     * \f]
     * where \f$Y_e\f$ is the effective Young's modulus and \f$R_e\f$ the effective radius. The
     * normal damping coefficient is given by
     * \f[
     * \eta_n = c_d \sqrt{\frac{3}{2} k_n m_e},
     * \f]
     * with \f$c_d\f$ being a damping prefactor derived from the coefficient of restitution (see
     * compute_damping_prefactor()) and \f$m_e\f$ the effective mass.
     *
     * @param contact_configuration  Configuration of the contact between two particles.
     * @return The computed normal contact force vector.
     */
    dealii::Tensor<1, dim, number>
    normal_contact_force(const ContactConfiguration &contact_configuration) const;

    /**
     * Computes the tangential contact force for a given particle–particle contact. The formulation
     * follows Golshan et al. (https://doi.org/10.1007/s40571-022-00478-6).
     *
     * The tangential contact force is defined as
     * \f[
     * \boldsymbol{F}_t = -k_t \boldsymbol{\delta}_t - \eta_t \boldsymbol{v}_{rt},
     * \f]
     * where \f$k_t\f$ is the tangential stiffness, \f$\boldsymbol{\delta}_t\f$ the tangential gap
     * vector, \f$\eta_t\f$ the tangential damping coefficient, and \f$\boldsymbol{v}_{rt}\f$ the
     * tangential component of the relative velocity.
     *
     * The tangential stiffness is computed as
     * \f[
     * k_t = 8 G_e \sqrt{R_e \delta_n},
     * \f]
     * with \f$G_e\f$ denoting the effective shear modulus, \f$R_e\f$ the effective radius, and
     * \f$\delta_n\f$ the normal gap.
     *
     * The tangential damping coefficient is given by
     * \f[
     * \eta_t = -2 \sqrt{\frac{5}{6}}\frac{\ln(e_r)}{\sqrt{\ln^2(e_r) + \pi^2}} \sqrt{k_t m_e},
     * \f]
     * where \f$e_r\f$ is the coefficient of restitution and \f$m_e\f$ the effective mass.
     *
     * If the magnitude of the tangential force exceeds the Coulomb friction limit, i.e.,
     * \f[
     * |\boldsymbol{F}_t| > \mu |\boldsymbol{F}_n|,
     *  \f]
     * with \f$\mu\f$ being the sliding friction coefficient and \f$\boldsymbol{F}_n\f$ the normal
     * contact force, the tangential gap is adjusted and the tangential force is recomputed such
     * that the Coulomb limit is exactly satisfied.
     *
     * @param contact_configuration  Contact configuration between the two particles.
     * @param normal_force Normal contact force vector for the contact.
     * @param tangential_gap Tangential gap vector from the previous time step; updated by this
     * function.
     *
     * @return Tangential contact force vector.
     */
    dealii::Tensor<1, dim, number>
    tangential_contact_force(const ContactConfiguration           &contact_configuration,
                             const dealii::Tensor<1, dim, number> &normal_force,
                             dealii::Tensor<1, dim, number>       &tangential_gap) const;

    /**
     * Given the tangential contact force at a particle–particle contact, this function
     * calculates the resulting torque about the particle center as the cross product
     * of the contact lever arm and the tangential force (M = (r*N) × F) with r being the particle
     * radius, N is the contact normal vector pointing from the particle center to the point of
     * contact, and F the tangential contact force acting at the contact point.
     *
     * @param contact_configuration Geometric contact configuration between the two particles.
     * @param tangential_force Tangential contact force vector at the contact point.
     * @param particle_radius Radius of the particle for which the torque is computed.
     * @return Tangential contact torque vector acting on the particle.
     */
    dealii::Tensor<1, axial_dim<dim>, number>
    tangential_contact_torque(const ContactConfiguration           &contact_configuration,
                              const dealii::Tensor<1, dim, number> &tangential_force,
                              const number                          particle_radius) const;

    /**
     * Computes the rolling resistance torque at a particle–particle or particle-wall contact based
     * on a viscous rolling resistance model. The model used in this function follows the
     * formulation presented by Meier et al. (DOI:10.1016/j.powtec.2018.11.072).
     *
     * The rolling resistance torque is defined as
     * \f[
     * \boldsymbol{M}_r = \mu_r |\boldsymbol{F}_n| R_e \boldsymbol{\omega}_c,
     * \f]
     * where \f$\mu_r\f$ is the rolling resistance coefficient, \f$\boldsymbol{F}_n\f$ the normal
     * contact force, \f$R_e\f$ the effective radius, and \f$\boldsymbol{\omega}_c\f$ the relative
     * angular velocity projected onto the contact plane.
     *
     * The relative angular velocity projected onto the contact plane is computed as
     * \f[
     * \boldsymbol{\omega}_c = \boldsymbol{\omega}_r - (\boldsymbol{\omega}_r \cdot \boldsymbol{n})
     * \boldsymbol{n},
     *  \f]
     * where \f$\boldsymbol{\omega}_r\f$ is the relative angular velocity
     * between the two particles (or particle and wall) and \f$\boldsymbol{n}\f$ the contact normal
     * vector.
     */
    dealii::Tensor<1, axial_dim<dim>, number>
    rolling_resistance_torque(const ContactConfiguration           &contact_configuration,
                              const dealii::Tensor<1, dim, number> &contact_normal_force) const;

    /**
     * Function which computes the damping prefactor from the restitution coefficient
     * \f[
     * \eta_t = 2 \sqrt{\frac{5}{6}}\frac{\ln(e_r)}{\sqrt{\ln^2(e_r) + \pi^2}},
     * \f]
     * with \f$e_r\f$ being the coefficient of restitution.
     *
     * @param restitution_coefficient The coefficient of restitution.
     * @return The computed damping prefactor.
     */
    number
    compute_damping_prefactor(const number restitution_coefficient) const;
  };
} // namespace MeltPoolDG
