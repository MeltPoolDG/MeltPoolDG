#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <meltpooldg/particles/contact_forces.hpp>
#include <meltpooldg/particles/dem_util.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/particle_accessor.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>

#include <cmath>
#include <numbers>

namespace MeltPoolDG
{
  template <int dim, typename number, typename ObstacleType>
  SphericalParticleContactForce<dim, number, ObstacleType>::SphericalParticleContactForce(
    const SphericalParticleContactData<number>              &contact_data,
    const MeltPoolDG::TimeIntegration::TimeIterator<number> &time_iterator)
    : contact_data(contact_data)
    , damping_prefactor(compute_damping_prefactor(contact_data.restitution_coefficient))
    , time_iterator(time_iterator)
  {}

  template <int dim, typename number, typename ObstacleType>
  void
  SphericalParticleContactForce<dim, number, ObstacleType>::attach_wall(
    std::unique_ptr<dealii::Function<dim, number>> &&wall_distance_function)
  {
    static int next_wall_id                 = 0;
    wall_distance_functions[next_wall_id++] = std::move(wall_distance_function);
  }


  template <int dim, typename number, typename ObstacleType>
  void
  SphericalParticleContactForce<dim, number, ObstacleType>::add_load_to_obstacles(
    ObstacleField<dim, number, ObstacleType> &obstacle_field) const
  {
    auto compute_and_add_contact_loads = [&](const ContactConfiguration &contact_configuration,
                                             DEMParticleAccessor<dim, number> &particle,
                                             dealii::Tensor<1, dim, number>   &tangential_gap) {
      dealii::Tensor<1, dim, number> normal_force = normal_contact_force(contact_configuration);
      dealii::Tensor<1, dim, number> tangential_force =
        tangential_contact_force(contact_configuration, normal_force, tangential_gap);

      particle.add_force(normal_force + tangential_force);

      dealii::Tensor<1, axial_dim<dim>, number> tangential_torque =
        tangential_contact_torque(contact_configuration,
                                  tangential_force,
                                  particle.get_property(ObstacleType::Properties::radius));

      dealii::Tensor<1, axial_dim<dim>, number> rolling_torque =
        rolling_resistance_torque(contact_configuration, normal_force);

      particle.add_torque(tangential_torque + rolling_torque);
    };


    for (auto &particle : obstacle_field.locally_owned_particle_range())
      {
        std::map<int, dealii::Tensor<1, dim, number>> &self_tangential_gaps =
          tangential_gaps[particle.id()];


        for (auto &other :
             obstacle_field.find_particles_in_neighborhood(particle,
                                                           0.0 /* direct contact required */))
          {
            if (particle.id() == other.id())
              continue;

            ContactConfiguration particle_particle_contact_configuration(
              particle,
              other,
              contact_data.particle.youngs_modulus,
              contact_data.particle.poisson_ratio);

            if (particle_particle_contact_configuration.normal_overlap > 0)
              {
                compute_and_add_contact_loads(particle_particle_contact_configuration,
                                              particle,
                                              self_tangential_gaps[other.id()]);
              }
            else
              {
                self_tangential_gaps.erase(other.id());
              }
          }

        for (auto &[key, wall_function] : wall_distance_functions)
          {
            ContactConfiguration particle_wall_contact_configuration(
              particle,
              wall_function.get(),
              contact_data.particle.youngs_modulus,
              contact_data.particle.poisson_ratio);
            if (particle_wall_contact_configuration.normal_overlap > 0)
              {
                compute_and_add_contact_loads(particle_wall_contact_configuration,
                                              particle,
                                              tangential_gaps_with_walls[particle.id()][key]);
              }
            else
              {
                tangential_gaps_with_walls[particle.id()].erase(key);
              }
          }
      }
  }

  template <int dim, typename number, typename ObstacleType>
  dealii::Tensor<1, dim, number>
  SphericalParticleContactForce<dim, number, ObstacleType>::normal_contact_force(
    const ContactConfiguration &contact_configuration) const
  {
    const number normal_stiffness =
      4. / 3. * contact_configuration.effective_youngs_modulus *
      std::sqrt(contact_configuration.effective_radius * contact_configuration.normal_overlap);

    const number normal_damping =
      damping_prefactor * std::sqrt(1.5 * normal_stiffness * contact_configuration.effective_mass);

    auto normal_force = -normal_stiffness * contact_configuration.normal_overlap *
                          contact_configuration.normal_vector -
                        normal_damping * contact_configuration.relative_velocity.normal_component;
    return normal_force;
  }

  template <int dim, typename number, typename ObstacleType>
  dealii::Tensor<1, dim, number>
  SphericalParticleContactForce<dim, number, ObstacleType>::tangential_contact_force(
    const ContactConfiguration           &contact_configuration,
    const dealii::Tensor<1, dim, number> &normal_force,
    dealii::Tensor<1, dim, number>       &tangential_gap) const
  {
    const number tangential_stiffness =
      8. * contact_configuration.effective_shear_modulus *
      std::sqrt(contact_configuration.effective_radius * contact_configuration.normal_overlap);

    const number tangential_damping =
      damping_prefactor * std::sqrt(tangential_stiffness * contact_configuration.effective_mass);

    const auto compute_tangential_force = [&]() -> dealii::Tensor<1, dim, number> {
      return -tangential_stiffness * tangential_gap -
             tangential_damping * contact_configuration.relative_velocity.tangential_component;
    };

    tangential_gap += contact_configuration.relative_velocity.tangential_component *
                      time_iterator.get_current_time_increment();

    dealii::Tensor<1, dim, number> tangential_force = compute_tangential_force();

    // Ensure that the tangential force does not exceed the Coulomb limit
    const number coulomb_friction_force_norm =
      contact_data.sliding_friction_coefficient * normal_force.norm();
    const number tangential_force_norm = tangential_force.norm();
    if (coulomb_friction_force_norm < tangential_force_norm)
      {
        dealii::Tensor<1, dim, number> limited_elastic_tangential_force =
          coulomb_friction_force_norm * tangential_force / tangential_force_norm +
          tangential_damping * contact_configuration.relative_velocity.tangential_component;

        tangential_gap = -limited_elastic_tangential_force / tangential_stiffness;

        tangential_force = compute_tangential_force();
      }

    return tangential_force;
  }

  template <int dim, typename number, typename ObstacleType>
  dealii::Tensor<1, axial_dim<dim>, number>
  SphericalParticleContactForce<dim, number, ObstacleType>::tangential_contact_torque(
    const ContactConfiguration           &contact_configuration,
    const dealii::Tensor<1, dim, number> &tangential_force,
    const number                          particle_radius) const
  {
    dealii::Tensor<1, axial_dim<dim>, number> tangential_torque;
    if constexpr (dim == 3)
      {
        tangential_torque =
          dealii::cross_product_3d(contact_configuration.normal_vector, tangential_force) *
          particle_radius;
      }
    else if constexpr (dim == 2)
      {
        tangential_torque[0] = contact_configuration.normal_vector[0] * tangential_force[1] -
                               contact_configuration.normal_vector[1] * tangential_force[0];
        tangential_torque[0] *= particle_radius;
      }
    else
      {
        AssertThrow(
          false,
          dealii::ExcMessage(
            "Only dimensions 2 and 3 are supported by the contact force implementation."));
      }
    return tangential_torque;
  }

  template <int dim, typename number, typename ObstacleType>
  dealii::Tensor<1, axial_dim<dim>, number>
  SphericalParticleContactForce<dim, number, ObstacleType>::rolling_resistance_torque(
    const ContactConfiguration           &contact_configuration,
    const dealii::Tensor<1, dim, number> &contact_normal_force) const
  {
    dealii::Tensor<1, axial_dim<dim>, number> contact_plane_relative_angular_velocity;

    if constexpr (dim == 3)
      {
        contact_plane_relative_angular_velocity =
          contact_configuration.relative_angular_velocity -
          (contact_configuration.relative_angular_velocity * contact_configuration.normal_vector) *
            contact_configuration.normal_vector;
      }
    else if constexpr (dim == 2)
      {
        contact_plane_relative_angular_velocity = contact_configuration.relative_angular_velocity;
      }
    else
      {
        AssertThrow(false,
                    dealii::ExcMessage("Only dimensions 2 and 3 are supported by the "
                                       "rolling resistance torque computation."));
      }

    return contact_data.rolling_resistance_coefficient * contact_normal_force.norm() *
           contact_configuration.effective_radius * contact_plane_relative_angular_velocity;
  }

  template <int dim, typename number, typename ObstacleType>
  number
  SphericalParticleContactForce<dim, number, ObstacleType>::compute_damping_prefactor(
    const number restitution_coefficient) const
  {
    return -2. * std::sqrt(5. / 6.) * std::log(restitution_coefficient) /
           std::sqrt(dealii::Utilities::fixed_power<2>(std::log(restitution_coefficient)) +
                     dealii::Utilities::fixed_power<2>(std::numbers::pi_v<number>));
  }

  template <int dim, typename number, typename ObstacleType>
  SphericalParticleContactForce<dim, number, ObstacleType>::ContactConfiguration::
    ContactConfiguration(const DEMParticleAccessor<dim, number> &self,
                         const DEMParticleAccessor<dim, number> &other,
                         const number                            youngs_modulus,
                         const number                            poisson_ratio)
  {
    auto compute_effective_quantity = [](number t1, number t2) -> number {
      return t1 * t2 / (t1 + t2);
    };

    effective_mass = compute_effective_quantity(self.get_property(ObstacleType::Properties::mass),
                                                other.get_property(ObstacleType::Properties::mass));
    effective_radius =
      compute_effective_quantity(self.get_property(ObstacleType::Properties::radius),
                                 other.get_property(ObstacleType::Properties::radius));

    effective_youngs_modulus = 0.5 * youngs_modulus / (1 - poisson_ratio * poisson_ratio);

    effective_shear_modulus = 0.25 * youngs_modulus / ((2 - poisson_ratio) * (1 + poisson_ratio));

    const number distance = self.get_location().distance(other.get_location());

    normal_vector  = (other.get_location() - self.get_location()) / distance;
    normal_overlap = (self.get_property(ObstacleType::Properties::radius) +
                      other.get_property(ObstacleType::Properties::radius)) -
                     distance;

    relative_angular_velocity = other.get_angular_velocity() - self.get_angular_velocity();

    if constexpr (dim == 3)
      {
        relative_velocity.value =
          self.linear_velocity() - other.linear_velocity() +
          dealii::cross_product_3d(self.get_property(ObstacleType::Properties::radius) *
                                       self.get_angular_velocity() +
                                     other.get_property(ObstacleType::Properties::radius) *
                                       other.get_angular_velocity(),
                                   normal_vector);
      }
    else if constexpr (dim == 2)
      {
        relative_velocity.value =
          self.linear_velocity() - other.linear_velocity() +
          (self.get_property(ObstacleType::Properties::radius) * self.get_angular_velocity()[0] +
           other.get_property(ObstacleType::Properties::radius) * other.get_angular_velocity()[0]) *
            dealii::cross_product_2d(normal_vector);
      }
    else
      {
        AssertThrow(false, dealii::ExcMessage("Dimension not supported!"));
      }

    relative_velocity.normal_component = (normal_vector * relative_velocity.value) * normal_vector;
    relative_velocity.tangential_component =
      relative_velocity.value - relative_velocity.normal_component;
  }

  template <int dim, typename number, typename ObstacleType>
  SphericalParticleContactForce<dim, number, ObstacleType>::ContactConfiguration::
    ContactConfiguration(const DEMParticleAccessor<dim, number> &particle,
                         const dealii::Function<dim>            *wall,
                         const number                            youngs_modulus,
                         const number                            poisson_ratio)
  {
    effective_mass   = particle.get_property(ObstacleType::Properties::mass);
    effective_radius = particle.get_property(ObstacleType::Properties::radius);

    effective_youngs_modulus = youngs_modulus / (1 - poisson_ratio * poisson_ratio);

    effective_shear_modulus = 0.5 * youngs_modulus / ((2 - poisson_ratio) * (1 + poisson_ratio));

    const number distance = std::abs(wall->value(particle.get_location()));

    // Normal vector points from particle to wall.
    normal_vector =
      -wall->gradient(particle.get_location()) / wall->gradient(particle.get_location()).norm();
    normal_overlap = particle.get_property(ObstacleType::Properties::radius) - distance;

    relative_angular_velocity = -particle.get_angular_velocity();

    if constexpr (dim == 3)
      {
        relative_velocity.value =
          particle.linear_velocity() +
          dealii::cross_product_3d(particle.get_property(ObstacleType::Properties::radius) *
                                     particle.get_angular_velocity(),
                                   normal_vector);
      }
    else if constexpr (dim == 2)
      {
        relative_velocity.value =
          particle.linear_velocity() + (particle.get_property(ObstacleType::Properties::radius) *
                                        particle.get_angular_velocity()[0]) *
                                         dealii::cross_product_2d(normal_vector);
      }
    else
      {
        AssertThrow(false, dealii::ExcMessage("Dimension not supported!"));
      }

    relative_velocity.normal_component = (normal_vector * relative_velocity.value) * normal_vector;
    relative_velocity.tangential_component =
      relative_velocity.value - relative_velocity.normal_component;
  }

} // namespace MeltPoolDG

template struct MeltPoolDG::SphericalParticleContactData<double>;

template class MeltPoolDG::
  SphericalParticleContactForce<1, double, MeltPoolDG::SphericalParticle<1, double>>;
template class MeltPoolDG::
  SphericalParticleContactForce<2, double, MeltPoolDG::SphericalParticle<2, double>>;
template class MeltPoolDG::
  SphericalParticleContactForce<3, double, MeltPoolDG::SphericalParticle<3, double>>;
