#include <meltpooldg/level_set/level_set_tools.hpp>
//
#include <deal.II/base/array_view.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/cell_data.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_geometry.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_operation.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>

#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <meltpooldg/utilities/generate_points_along_vector.hpp>
#include <meltpooldg/utilities/triangulation_utils.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <optional>
#include <string>


namespace MeltPoolDG::LevelSet::Tools
{
  template <typename number>
  dealii::LinearAlgebra::distributed::Vector<number>
  merge_two_indicator_fields(const dealii::LinearAlgebra::distributed::Vector<number> &indicator_1,
                             const dealii::LinearAlgebra::distributed::Vector<number> &indicator_2,
                             BooleanType                                               type,
                             const number indicator_value_interior,
                             const number indicator_value_exterior)
  {
    AssertThrow(indicator_1.size() == indicator_2.size(),
                dealii::ExcMessage(
                  "The two level set vectors to be merged must be of equal length."));

    AssertThrow(indicator_value_interior != indicator_value_exterior,
                dealii::ExcMessage(
                  "The indicator value in the interior of the implicit geometry must be "
                  "different than the one in the exterior. Make sure that the function "
                  "arguments indicator_value_interior and indicator_value_exterior are "
                  "set correctly."));

    AssertThrow(indicator_1.linfty_norm() ==
                  std::max(indicator_value_exterior, indicator_value_interior),
                dealii::ExcMessage(
                  "The maximum indicator value does not correspond to the given bounds "
                  "of indicator_value_interior and indicator_value_exterior. Make sure that "
                  "the indicator_value_interior and indicator_value_exterior comply with the "
                  "indicator field."));

    dealii::LinearAlgebra::distributed::Vector<number> merge_indicator(indicator_1);

    bool is_interior_larger_than_exterior = indicator_value_interior > indicator_value_exterior;

    for (const auto &i : indicator_1.locally_owned_elements())
      switch (type)
        {
          case BooleanType::Union:
            merge_indicator[i] = is_interior_larger_than_exterior ?
                                   std::max(indicator_1[i], indicator_2[i]) :
                                   std::min(indicator_1[i], indicator_2[i]);
            break;
          case BooleanType::Intersection:
            merge_indicator[i] = is_interior_larger_than_exterior ?
                                   std::min(indicator_1[i], indicator_2[i]) :
                                   std::max(indicator_1[i], indicator_2[i]);
            break;
          case BooleanType::Subtraction:
            merge_indicator[i] =
              indicator_1[i] == indicator_2[i] && indicator_1[i] == indicator_value_interior ?
                indicator_value_exterior :
                indicator_1[i];
            break;
          default:
            AssertThrow(false, dealii::ExcNotImplemented());
        }

    merge_indicator.compress(dealii::VectorOperation::insert);

    return merge_indicator;
  }

  template <int dim, typename number>
  void
  collect_interface_cells_and_intersection_points(
    std::vector<std::pair<std::pair<unsigned int, unsigned int>, std::vector<dealii::Point<dim>>>>
                                                             &surface_cells_and_unit_points,
    std::vector<dealii::Point<dim>>                          &surface_points,
    const dealii::DoFHandler<dim>                            &dof_handler,
    const dealii::Mapping<dim>                               &mapping,
    const dealii::LinearAlgebra::distributed::Vector<number> &level_set_vector,
    const number                                              contour_value,
    const unsigned int                                        n_subdivisions,
    const number                                              tolerance)
  {
    dealii::GridTools::MarchingCubeAlgorithm<dim,
                                             dealii::LinearAlgebra::distributed::Vector<number>>
      mc(mapping, dof_handler.get_fe(), n_subdivisions, tolerance);

    const bool is_ghosted = level_set_vector.has_ghost_elements();

    if (not is_ghosted)
      level_set_vector.update_ghost_values();

    for (const auto &cell : dof_handler.active_cell_iterators_on_level(
           dof_handler.get_triangulation().n_global_levels() - 1))
      {
        if (not cell->is_locally_owned())
          continue;
        // determine if cell is cut by the interface and if yes, determine the quadrature
        // point location (at the reference cell) and weight
        // determine points and cells of aux surface triangulation
        std::vector<dealii::Point<dim>>                       surface_vertices;
        std::vector<dealii::CellData<dim == 1 ? 1 : dim - 1>> surface_cells;

        // run marching cube algorithm
        if (dim > 1)
          mc.process_cell(cell, level_set_vector, contour_value, surface_vertices, surface_cells);
        else
          mc.process_cell(cell, level_set_vector, contour_value, surface_vertices);

        if (surface_vertices.empty())
          continue; // cell is not cut by interface -> no quadrature points have the be
                    // determined

        std::vector<dealii::Point<dim>> real_points;
        std::vector<dealii::Point<dim>> unit_points;

        if constexpr (dim == 1)
          {
            AssertThrow(surface_vertices.size() == 1,
                        dealii::ExcMessage("The MCA in 1D found too many points."));

            real_points.emplace_back(surface_vertices[0]);
            unit_points.emplace_back(mapping.transform_real_to_unit_cell(cell, real_points[0]));
          }
        else
          {
            // Construct auxiliary triangulation
            dealii::Triangulation<dim == 1 ? 1 : dim - 1, dim> surface_triangulation;
            surface_triangulation.create_triangulation(surface_vertices, surface_cells, {});

            // FE setup for reference mapping
            dealii::FE_SimplexP<dim == 1 ? 1 : dim - 1, dim> fe_simplex(
              dof_handler.get_fe().tensor_degree());
            dealii::MappingFE<dim == 1 ? 1 : dim - 1, dim> simplex_mapping(fe_simplex);

            dealii::FE_Nothing<dim == 1 ? 1 : dim - 1, dim> fe;
            dealii::FEValues<dim == 1 ? 1 : dim - 1, dim>   fe_eval(
              simplex_mapping,
              fe,
              dealii::Quadrature < dim == 1 ? 1 : dim - 1 > (fe_simplex.get_unit_support_points()),
              dealii::update_quadrature_points);

            // loop over all cells ...
            for (const auto &sub_cell : surface_triangulation.active_cell_iterators())
              {
                fe_eval.reinit(sub_cell);

                // ... and collect quadrature points and weights
                for (const auto &q : fe_eval.quadrature_point_indices())
                  {
                    const auto &real_point = fe_eval.quadrature_point(q);
                    real_points.emplace_back(real_point);
                    unit_points.emplace_back(mapping.transform_real_to_unit_cell(cell, real_point));
                  }
              }
          }
        surface_cells_and_unit_points.emplace_back(std::make_pair(cell->level(), cell->index()),
                                                   unit_points);
        for (const auto &p : real_points)
          surface_points.emplace_back(p);
      }

    if (not is_ghosted)
      level_set_vector.zero_out_ghost_values();
  }

  template <int dim, typename number>
  void
  evaluate_at_interface(const dealii::DoFHandler<dim>                            &dof_handler,
                        const dealii::Mapping<dim>                               &mapping,
                        const dealii::LinearAlgebra::distributed::Vector<number> &level_set_vector,
                        const std::function<void(
                          const typename dealii::DoFHandler<dim>::active_cell_iterator & /*cell*/,
                          const std::vector<dealii::Point<dim>> & /*quadrature_points_real*/,
                          const std::vector<dealii::Point<dim>> & /*quadrature_points_reference*/,
                          const std::vector<number> & /*JxW*/)> &evaluate_at_interface_points,
                        const number                             contour_value,
                        const unsigned int                       n_subdivisions,
                        const number                             tolerance,
                        const bool                               use_mca)
  {
    // data structures for marching-cube algorithm
    const dealii::QGaussSimplex<dim == 1 ? 1 : dim - 1> surface_quad(dof_handler.get_fe().degree +
                                                                     1);

    dealii::GridTools::MarchingCubeAlgorithm<dim,
                                             dealii::LinearAlgebra::distributed::Vector<number>>
      mc(mapping,
         dof_handler.get_fe(), // todo
         n_subdivisions,
         tolerance);

    const bool is_ghosted = level_set_vector.has_ghost_elements();

    if (not is_ghosted)
      level_set_vector.update_ghost_values();

    // data structures for NonMatching::FEValues
    const unsigned int nm_n_q_points_1D =
      std::max<unsigned int>(dof_handler.get_fe().degree, n_subdivisions + 1);
    const dealii::hp::QCollection<dim> nm_quad(dealii::QGauss<dim>{nm_n_q_points_1D});
    const dealii::hp::QCollection<1>   nm_surface_quad_1D(dealii::QGauss<1>{nm_n_q_points_1D});
    const dealii::hp::MappingCollection<dim> nm_mapping_collection(mapping);
    const dealii::hp::FECollection<dim>      nm_fe_collection(dof_handler.get_fe());

    dealii::LinearAlgebra::distributed::Vector<number> nm_locally_relevant_level_set_vector;

    std::unique_ptr<dealii::NonMatching::MeshClassifier<dim>> nm_mesh_classifier;
    std::unique_ptr<dealii::NonMatching::FEValues<dim>>       nm_non_matching_fe_values;

    // only for dim == 1
    std::map<number, number> total_nodal_weights; // TODO: also necessary for other dimensions?

    if (use_mca == false) // this is an expensive step; execute only if needed
      {
        nm_mesh_classifier = std::make_unique<dealii::NonMatching::MeshClassifier<dim>>(
          dof_handler, nm_locally_relevant_level_set_vector);

        nm_locally_relevant_level_set_vector.reinit(dof_handler.locally_owned_dofs(),
                                                    dealii::DoFTools::extract_locally_relevant_dofs(
                                                      dof_handler),
                                                    level_set_vector.get_mpi_communicator());
        nm_locally_relevant_level_set_vector.copy_locally_owned_data_from(level_set_vector);
        nm_locally_relevant_level_set_vector.update_ghost_values();
        nm_mesh_classifier->reclassify();

        dealii::NonMatching::RegionUpdateFlags nm_region_update_flags;
        nm_region_update_flags.surface =
          dealii::update_quadrature_points | dealii::update_JxW_values;

        nm_non_matching_fe_values = std::make_unique<dealii::NonMatching::FEValues<dim>>(
          nm_mapping_collection,
          nm_fe_collection,
          nm_quad,
          nm_surface_quad_1D,
          nm_region_update_flags,
          *nm_mesh_classifier,
          dof_handler,
          nm_locally_relevant_level_set_vector);
      }

    for (const auto &cell : dof_handler.active_cell_iterators_on_level(
           dof_handler.get_triangulation().n_global_levels() - 1))
      {
        if (not cell->is_locally_owned())
          continue;
        // determine if cell is cut by the interface and if yes, determine the quadrature
        // point location (at the reference cell) and weight
        const auto fu_mca = [&]() -> std::tuple<std::vector<dealii::Point<dim>>,
                                                std::vector<dealii::Point<dim>>,
                                                std::vector<number>> {
          // determine points and cells of aux surface triangulation
          std::vector<dealii::Point<dim>>                       surface_vertices;
          std::vector<dealii::CellData<dim == 1 ? 1 : dim - 1>> surface_cells;

          // run marching cube algorithm
          if (dim > 1)
            mc.process_cell(cell, level_set_vector, contour_value, surface_vertices, surface_cells);
          else
            mc.process_cell(cell, level_set_vector, contour_value, surface_vertices);

          if (surface_vertices.size() == 0)
            return {}; // cell is not cut by interface -> no quadrature points have the be
                       // determined

          std::vector<dealii::Point<dim>> points_real;
          std::vector<dealii::Point<dim>> points;
          std::vector<number>             weights;

          if constexpr (dim == 1)
            {
              AssertThrow(surface_vertices.size() == 1,
                          dealii::ExcMessage("The MCA in 1D found too many points."));

              points_real.emplace_back(surface_vertices[0]);
              points.emplace_back(mapping.transform_real_to_unit_cell(cell, points_real[0]));

              const auto p = points_real[0][0];

              // check if point is vertex
              if (points_real[0].distance(cell->vertex(0)) < 1e-16 ||
                  points_real[0].distance(cell->vertex(cell->n_vertices() - 1)) < 1e-16)
                {
                  // consistency check, if MCA returns 2 cells if a corner node is found
                  if (total_nodal_weights.find(p) == total_nodal_weights.end())
                    total_nodal_weights[p] = 0.5;
                  else
                    total_nodal_weights[p] += 0.5;

                  weights.emplace_back(0.5);
                }
              else
                {
                  if (total_nodal_weights.find(p) == total_nodal_weights.end())
                    total_nodal_weights[p] = 1.0;
                  else
                    AssertThrow(false,
                                dealii::ExcMessage("Inconsistency found with MCA. Abort..."));

                  weights.emplace_back(1.);
                }
            }
          else
            {
              // create aux triangulation of subcells
              dealii::Triangulation<dim == 1 ? 1 : dim - 1, dim> surface_triangulation;
              surface_triangulation.create_triangulation(surface_vertices, surface_cells, {});

              // for 3D MCA
              dealii::FE_SimplexP<dim == 1 ? 1 : dim - 1, dim> fe_simplex(
                dof_handler.get_fe().tensor_degree());
              dealii::MappingFE<dim == 1 ? 1 : dim - 1, dim> simplex_mapping(fe_simplex);

              dealii::FE_Nothing<dim == 1 ? 1 : dim - 1, dim> fe;
              dealii::FEValues<dim == 1 ? 1 : dim - 1, dim>   fe_eval(
                simplex_mapping,
                fe,
                surface_quad,
                dealii::update_quadrature_points | dealii::update_JxW_values);

              // loop over all cells ...
              for (const auto &sub_cell : surface_triangulation.active_cell_iterators())
                {
                  Assert(sub_cell->reference_cell().is_simplex(),
                         dealii::ExcMessage("MCA subcells are expected to be simplices."));
                  fe_eval.reinit(sub_cell);

                  // ... and collect quadrature points and weights
                  for (const auto &q : fe_eval.quadrature_point_indices())
                    {
                      points_real.emplace_back(fe_eval.quadrature_point(q));
                      points.emplace_back(
                        mapping.transform_real_to_unit_cell(cell, fe_eval.quadrature_point(q)));

                      weights.emplace_back(fe_eval.JxW(q));
                    }
                }
            }
          return {points_real, points, weights};
        };

        const auto fu_non_matching = [&]() -> std::tuple<std::vector<dealii::Point<dim>>,
                                                         std::vector<dealii::Point<dim>>,
                                                         std::vector<number>> {
          nm_non_matching_fe_values->reinit(cell);

          const std::optional<dealii::NonMatching::FEImmersedSurfaceValues<dim>>
            &surface_fe_values = nm_non_matching_fe_values->get_surface_fe_values();

          if (surface_fe_values.has_value() == false)
            return {};

          std::vector<dealii::Point<dim>> points;

          for (const auto &q : surface_fe_values->get_quadrature_points())
            points.emplace_back(mapping.transform_real_to_unit_cell(cell, q));

          return {surface_fe_values->get_quadrature_points(),
                  points,
                  surface_fe_values->get_JxW_values()};
        };

        const auto [points_real, points, weights] = use_mca ? fu_mca() : fu_non_matching();

        if (points_real.size() == 0)
          continue; // cell is not cut but the interface -> nothing to do

        evaluate_at_interface_points(cell, points_real, points, weights);
      }

    // check if total nodal weights are equal to 1
    if (dim == 1)
      for (const auto &[k, w] : total_nodal_weights)
        AssertThrow(std::abs(w - 1) < 1e-10,
                    dealii::ExcMessage(
                      "The MCA delivered only one cell corresponding to the corner node " +
                      std::to_string(k) + ". The total nodal weight is " + std::to_string(w) +
                      ". Abort..."));

    if (not is_ghosted)
      level_set_vector.zero_out_ghost_values();
  }

  template <int dim, typename number>
  std::vector<std::tuple<const typename dealii::Triangulation<dim, dim>::cell_iterator /*cell*/,
                         std::vector<dealii::Point<dim>> /*unit_points*/,
                         std::vector<number> /*weights*/
                         >>
  generate_surface_mesh_info(const dealii::DoFHandler<dim>                            &dof_handler,
                             const dealii::Mapping<dim>                               &mapping,
                             const dealii::LinearAlgebra::distributed::Vector<number> &level_set,
                             const number       contour_value,
                             const unsigned int n_subdivisions,
                             const number       tolerance,
                             const bool         use_mca)
  {
    std::vector<std::tuple<const typename dealii::Triangulation<dim, dim>::cell_iterator /*cell*/,
                           std::vector<dealii::Point<dim>> /*quad_points*/,
                           std::vector<number> /*weights*/
                           >>
      surface_mesh_info;

    evaluate_at_interface<dim, number>(
      dof_handler,
      mapping,
      level_set,
      [&](const auto &cell, const auto &, const auto &unit_points, const auto &weights) {
        if (unit_points.size() > 0)
          surface_mesh_info.emplace_back(cell, unit_points, weights);
      },
      contour_value,
      n_subdivisions,
      tolerance,
      use_mca);

    return surface_mesh_info;
  }

  template <int dim, typename number>
  void
  generate_points_along_normal(
    std::vector<dealii::Point<dim>> &global_points_normal_to_interface,
    std::vector<unsigned int>       &global_points_normal_to_interface_pointer,
    const dealii::DoFHandler<dim>   &dof_handler_ls,
    const dealii::FESystem<dim>     &fe_normal,
    const dealii::Mapping<dim>      &mapping,
    const dealii::LinearAlgebra::distributed::Vector<number>      &level_set_vector,
    const dealii::LinearAlgebra::distributed::BlockVector<number> &normal_vector,
    const number                                                   max_distance_per_side,
    const unsigned int                                             n_inc_per_side,
    const bool                                                     bidirectional,
    const number                                                   contour_value,
    const unsigned int                                             n_subdivisions_MCA)
  {
    const bool level_set_is_ghosted = level_set_vector.has_ghost_elements();
    if (not level_set_is_ghosted)
      level_set_vector.update_ghost_values();

    const bool normal_vector_is_ghosted = normal_vector.has_ghost_elements();
    if (not normal_vector_is_ghosted)
      normal_vector.update_ghost_values();

    dealii::FEPointEvaluation<dim, dim> phi_normal(mapping, fe_normal, dealii::update_values);

    std::vector<number>                          buffer;
    std::vector<number>                          buffer_dim;
    std::vector<dealii::types::global_dof_index> local_dof_indices;

    global_points_normal_to_interface.clear();
    global_points_normal_to_interface_pointer.clear();

    global_points_normal_to_interface_pointer = {0};

    const auto bounding_box =
      dealii::GridTools::compute_bounding_box(dof_handler_ls.get_triangulation());
    const auto boundary_points = bounding_box.get_boundary_points();


    const auto collect_points_along_normal = [&](const auto                  &cell,
                                                 const auto                  &real_points,
                                                 const auto                  &unit_points,
                                                 [[maybe_unused]] const auto &weights) {
      local_dof_indices.resize(cell->get_fe().n_dofs_per_cell());
      buffer.resize(cell->get_fe().n_dofs_per_cell());
      cell->get_dof_indices(local_dof_indices);

      phi_normal.reinit(cell, unit_points);
      /*
       * gather_evaluate normal for the points at the interface
       */
      {
        buffer_dim.resize(fe_normal.n_dofs_per_cell());
        for (int i = 0; i < dim; ++i)
          {
            cell->get_dof_values(normal_vector.block(i), buffer.begin(), buffer.end());

            for (unsigned int c = 0; c < cell->get_fe().n_dofs_per_cell(); ++c)
              buffer_dim[fe_normal.component_to_system_index(i, c)] = buffer[c];
          }

        // normalize
        for (unsigned int c = 0; c < cell->get_fe().n_dofs_per_cell(); ++c)
          {
            number norm = 0.0;
            for (int i = 0; i < dim; ++i)
              norm += dealii::Utilities::fixed_power<2>(
                buffer_dim[fe_normal.component_to_system_index(i, c)]);

            norm = std::sqrt(norm);

            if (norm > 1e-16)
              {
                for (int i = 0; i < dim; ++i)
                  buffer_dim[fe_normal.component_to_system_index(i, c)] /= norm;
              }
          }

        phi_normal.evaluate(dealii::make_array_view(buffer_dim), dealii::EvaluationFlags::values);
      }

      /*
       * loop over generated points from MCA
       */
      for (unsigned int p_index = 0; p_index < real_points.size(); ++p_index)
        {
          /*
           * generate points (x) along normal
           *
           *          x
           *          |
           *          x
           *          |
           *    ---------------  ls = contour_value
           *          |
           *          x
           *          |
           *          x
           */
          std::vector<dealii::Point<dim>> points_normal_to_interface;

          const auto unit_normal = phi_normal.get_value(p_index);

          Utility::generate_points_along_vector<dim>(points_normal_to_interface,
                                                     real_points[p_index],
                                                     unit_normal,
                                                     max_distance_per_side,
                                                     n_inc_per_side,
                                                     bidirectional);

          // check if point is outside domain and if so then project it back to
          // the domain
          //
          // @todo: compute intersection between normal and bounding box and use
          // this point
          for (unsigned int d = 0; d < dim; ++d)
            {
              for (unsigned int counter = 0; counter < points_normal_to_interface.size(); counter++)
                {
                  if (points_normal_to_interface[counter][d] < boundary_points.first[d])
                    {
                      points_normal_to_interface.erase(points_normal_to_interface.begin() +
                                                       counter);
                      continue;
                    }
                  else if (points_normal_to_interface[counter][d] > boundary_points.second[d])
                    points_normal_to_interface.erase(points_normal_to_interface.begin() + counter);
                }
            }

          global_points_normal_to_interface.insert(global_points_normal_to_interface.end(),
                                                   points_normal_to_interface.begin(),
                                                   points_normal_to_interface.end());
          global_points_normal_to_interface_pointer.emplace_back(
            global_points_normal_to_interface.size());
        }
    };

    evaluate_at_interface<dim, number>(dof_handler_ls,
                                       mapping,
                                       level_set_vector,
                                       collect_points_along_normal,
                                       contour_value, /*contour value*/
                                       n_subdivisions_MCA /*n_subdivisions*/);

    if (not level_set_is_ghosted)
      level_set_vector.zero_out_ghost_values();
    if (not normal_vector_is_ghosted)
      normal_vector.zero_out_ghost_values();
  }

  template <int dim, typename number>
  void
  set_material_id_from_level_set(
    const ScratchData<dim, dim, number>                      &scratch_data,
    const unsigned int                                        ls_dof_idx,
    const dealii::LinearAlgebra::distributed::Vector<number> &level_set_heaviside,
    const number                                              lower_threshold)
  {
    const bool has_ghost_elements = level_set_heaviside.has_ghost_elements();

    if (not has_ghost_elements)
      level_set_heaviside.update_ghost_values();

    dealii::Vector<number> hs_local(scratch_data.get_n_dofs_per_cell(ls_dof_idx));
    for (const auto &cell : scratch_data.get_dof_handler(ls_dof_idx).active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            cell->get_dof_values(level_set_heaviside, hs_local);
            const number min_ls = *std::min_element(hs_local.begin(), hs_local.end());
            cell->set_material_id(min_ls >= lower_threshold ? 1 : 0);
          }
      }

    if (not has_ghost_elements)
      level_set_heaviside.zero_out_ghost_values();

    // communicate local data to ghost cells
    if (get_triangulation_type(scratch_data.get_triangulation()) != TriangulationType::serial)
      {
        using active_cell_iterator = typename dealii::Triangulation<dim>::active_cell_iterator;
        auto pack                  = [](const active_cell_iterator &cell) -> unsigned int {
          return cell->material_id();
        };

        auto unpack = [](const active_cell_iterator &cell, const unsigned int material_id) -> void {
          cell->set_material_id(material_id);
        };
        if (const auto tria_shared = dynamic_cast<dealii::parallel::shared::Triangulation<dim> *>(
              const_cast<dealii::Triangulation<dim> *>(&scratch_data.get_triangulation())))
          {
            AssertThrow(
              tria_shared->with_artificial_cells(),
              dealii::ExcMessage(
                "For a shared triangulation, make sure that you allow to create artificial cells."
                "Otherwise every cell is a ghost cell, which prohibits the use of this functionality."));
          }

        dealii::GridTools::exchange_cell_data_to_ghosts<unsigned int, dealii::Triangulation<dim>>(
          scratch_data.get_triangulation(), pack, unpack);
      }
  }



  template dealii::LinearAlgebra::distributed::Vector<double>
  merge_two_indicator_fields(const dealii::LinearAlgebra::distributed::Vector<double> &,
                             const dealii::LinearAlgebra::distributed::Vector<double> &,
                             BooleanType,
                             const double,
                             const double);

  template void
  collect_interface_cells_and_intersection_points(
    std::vector<std::pair<std::pair<unsigned int, unsigned int>, std::vector<dealii::Point<1>>>> &,
    std::vector<dealii::Point<1>> &,
    const dealii::DoFHandler<1> &,
    const dealii::Mapping<1> &,
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    const double,
    const unsigned int,
    const double);
  template void
  collect_interface_cells_and_intersection_points(
    std::vector<std::pair<std::pair<unsigned int, unsigned int>, std::vector<dealii::Point<2>>>> &,
    std::vector<dealii::Point<2>> &,
    const dealii::DoFHandler<2> &,
    const dealii::Mapping<2> &,
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    const double,
    const unsigned int,
    const double);
  template void
  collect_interface_cells_and_intersection_points(
    std::vector<std::pair<std::pair<unsigned int, unsigned int>, std::vector<dealii::Point<3>>>> &,
    std::vector<dealii::Point<3>> &,
    const dealii::DoFHandler<3> &,
    const dealii::Mapping<3> &,
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    const double,
    const unsigned int,
    const double);

  template void
  evaluate_at_interface(
    const dealii::DoFHandler<1> &,
    const dealii::Mapping<1> &,
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    const std::function<void(const typename dealii::DoFHandler<1>::active_cell_iterator &,
                             const std::vector<dealii::Point<1>> &,
                             const std::vector<dealii::Point<1>> &,
                             const std::vector<double> &)> &,
    const double,
    const unsigned int,
    const double,
    const bool);
  template void
  evaluate_at_interface(
    const dealii::DoFHandler<2> &,
    const dealii::Mapping<2> &,
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    const std::function<void(const typename dealii::DoFHandler<2>::active_cell_iterator &,
                             const std::vector<dealii::Point<2>> &,
                             const std::vector<dealii::Point<2>> &,
                             const std::vector<double> &)> &,
    const double,
    const unsigned int,
    const double,
    const bool);
  template void
  evaluate_at_interface(
    const dealii::DoFHandler<3> &,
    const dealii::Mapping<3> &,
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    const std::function<void(const typename dealii::DoFHandler<3>::active_cell_iterator &,
                             const std::vector<dealii::Point<3>> &,
                             const std::vector<dealii::Point<3>> &,
                             const std::vector<double> &)> &,
    const double,
    const unsigned int,
    const double,
    const bool);

  template std::vector<std::tuple<const typename dealii::Triangulation<1, 1>::cell_iterator,
                                  std::vector<dealii::Point<1>>,
                                  std::vector<double>>>
  generate_surface_mesh_info(const dealii::DoFHandler<1> &,
                             const dealii::Mapping<1> &,
                             const dealii::LinearAlgebra::distributed::Vector<double> &,
                             const double,
                             const unsigned int,
                             const double,
                             const bool);
  template std::vector<std::tuple<const typename dealii::Triangulation<2, 2>::cell_iterator,
                                  std::vector<dealii::Point<2>>,
                                  std::vector<double>>>
  generate_surface_mesh_info(const dealii::DoFHandler<2> &,
                             const dealii::Mapping<2> &,
                             const dealii::LinearAlgebra::distributed::Vector<double> &,
                             const double,
                             const unsigned int,
                             const double,
                             const bool);
  template std::vector<std::tuple<const typename dealii::Triangulation<3, 3>::cell_iterator,
                                  std::vector<dealii::Point<3>>,
                                  std::vector<double>>>
  generate_surface_mesh_info(const dealii::DoFHandler<3> &,
                             const dealii::Mapping<3> &,
                             const dealii::LinearAlgebra::distributed::Vector<double> &,
                             const double,
                             const unsigned int,
                             const double,
                             const bool);

  template void
  generate_points_along_normal(std::vector<dealii::Point<2>> &,
                               std::vector<unsigned int> &,
                               const dealii::DoFHandler<2> &,
                               const dealii::FESystem<2> &,
                               const dealii::Mapping<2> &,
                               const dealii::LinearAlgebra::distributed::Vector<double> &,
                               const dealii::LinearAlgebra::distributed::BlockVector<double> &,
                               const double,
                               const unsigned int,
                               const bool,
                               const double,
                               const unsigned int);
  template void
  generate_points_along_normal(std::vector<dealii::Point<3>> &,
                               std::vector<unsigned int> &,
                               const dealii::DoFHandler<3> &,
                               const dealii::FESystem<3> &,
                               const dealii::Mapping<3> &,
                               const dealii::LinearAlgebra::distributed::Vector<double> &,
                               const dealii::LinearAlgebra::distributed::BlockVector<double> &,
                               const double,
                               const unsigned int,
                               const bool,
                               const double,
                               const unsigned int);

  template void
  set_material_id_from_level_set(const ScratchData<1, 1, double> &,
                                 const unsigned int,
                                 const dealii::LinearAlgebra::distributed::Vector<double> &,
                                 const double);
  template void
  set_material_id_from_level_set(const ScratchData<2, 2, double> &,
                                 const unsigned int,
                                 const dealii::LinearAlgebra::distributed::Vector<double> &,
                                 const double);
  template void
  set_material_id_from_level_set(const ScratchData<3, 3, double> &,
                                 const unsigned int,
                                 const dealii::LinearAlgebra::distributed::Vector<double> &,
                                 const double);
} // namespace MeltPoolDG::LevelSet::Tools
