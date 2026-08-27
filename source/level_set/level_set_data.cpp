#include <deal.II/base/exceptions.h>
#include <deal.II/base/patterns.h>

#include <meltpooldg/level_set/level_set_data.hpp>

namespace MeltPoolDG::LevelSet
{
  template <typename number>
  void
  LevelSetData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("level set");
    {
      fe.add_parameters(prm);

      prm.add_parameter("do localized heaviside",
                        do_localized_heaviside,
                        "Determine if the heaviside representation of the level set should be "
                        "calculated as a localized function, being exactly 0 and 1 outside of "
                        "the interface region.");

      prm.add_parameter(
        "heaviside thickness coefficient",
        heaviside_thickness_coefficient,
        "Parameter C_h that determines the interface region thickness of the smoothed heaviside,"
        " which is also used for regularized delta functions. The resulting interface thickness is obtained from "
        "2 * C_h * eps_phi, where eps_phi is the interface thickness parameter from the tanh level set field."
        "In the default case the thickness is 6 * eps_phi.");

      prm.add_parameter(
        "gradient error evaluation distance cell proportion",
        level_set_DG_specific_data.gradient_error_evaluation_distance_cell_proportion,
        "Factor how many cell diameters away the gradient error should be evaluated");

      nearest_point.add_parameters(prm);
      advec_diff.add_parameters(prm);
      normal_vec.add_parameters(prm);
      curv.add_parameters(prm);
      reinit.add_parameters(prm);
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  LevelSetData<number>::post(const FiniteElementData &base_fe_data,
                             const unsigned int       base_verbosity_level)
  {
    fe.post(base_fe_data);
    advec_diff.post(fe);
    normal_vec.post(base_verbosity_level);
    curv.post(base_verbosity_level);
    reinit.post(fe);
  }

  template <typename number>
  void
  LevelSetData<number>::check_input_parameters(const FiniteElementData &base_fe_data) const
  {
    fe.check_input_parameters(base_fe_data);

    if (base_fe_data.type == FiniteElementType::FE_SimplexP)
      AssertThrow(get_n_subdivisions() == 1,
                  dealii::ExcMessage(
                    "If you use a simplex mesh, n_subdivisions for the level set must be 1."));

    advec_diff.check_input_parameters();
    normal_vec.check_input_parameters(reinit.interface_thickness_parameter.type);
    curv.check_input_parameters(reinit.interface_thickness_parameter.type);
    reinit.check_input_parameters(normal_vec.linear_solver.do_matrix_free);
  }

  template <typename number>
  unsigned int
  LevelSetData<number>::get_n_subdivisions() const
  {
    return fe.get_n_subdivisions();
  }

  template struct LevelSetData<double>;
} // namespace MeltPoolDG::LevelSet
