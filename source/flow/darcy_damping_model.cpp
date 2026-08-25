#include <deal.II/base/exception_macros.h>
#include <deal.II/base/exceptions.h>

#include <meltpooldg/flow/darcy_damping_model.hpp>

namespace MeltPoolDG::Flow
{
  template <typename number>
  DarcyDampingModel<number>::DarcyDampingModel(const DarcyDampingData<number> &data_in)
    : mushy_zone_morphology(data_in.mushy_zone_morphology)
    , avoid_div_zero_constant(data_in.avoid_div_zero_constant)
  {
    AssertThrow(mushy_zone_morphology == 0. or avoid_div_zero_constant > 0.,
                dealii::ExcMessage(
                  "When computing the Darcy damping coefficient, the parameter \"mp solid "
                  "darcy damping avoid div zero constant\" must be greater than zero! Abort.."));
  }

  template <typename number>
  template <typename number2>
  number2
  DarcyDampingModel<number>::compute_darcy_damping_coefficient(const number2 &solid_fraction) const
  {
    // K = -C * fs² / ( (1-fs)³ + b )
    // K := permeability = Darcy damping coefficient
    // C := morphology
    // b := avoid div zero constant
    // fs := solid fraction
    const auto liquid_fraction = 1.0 - solid_fraction;
    return -mushy_zone_morphology * solid_fraction * solid_fraction /
           (liquid_fraction * liquid_fraction * liquid_fraction + avoid_div_zero_constant);
  }

  template class DarcyDampingModel<double>;

  template double
  DarcyDampingModel<double>::compute_darcy_damping_coefficient<double>(const double &) const;

  template dealii::VectorizedArray<double>
  DarcyDampingModel<double>::compute_darcy_damping_coefficient<dealii::VectorizedArray<double>>(
    const dealii::VectorizedArray<double> &) const;
} // namespace MeltPoolDG::Flow
