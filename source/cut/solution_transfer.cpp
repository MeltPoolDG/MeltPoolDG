#include <meltpooldg/cut/solution_transfer.hpp>
//

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_operation.h>

#include <deal.II/numerics/solution_transfer.h>

#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>


namespace MeltPoolDG::CutUtil
{
  template <int dim, typename number>
  SolutionTransferOperator<dim, number>::SolutionTransferOperator(
    const GhostPenaltyData<number> &ghost_penalty_in,
    const bool                      is_two_phase,
    const int                       verbosity)
    : fe_degree(0)
    , n_components_per_phase(0)
    , is_dg(false)
    , ghost_penalty(ghost_penalty_in)
    , is_two_phase(is_two_phase)
    , verbosity(verbosity)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {}

  template <int dim, typename number>
  void
  SolutionTransferOperator<dim, number>::reinit(
    dealii::DoFHandler<dim>                               &cut_dof_handler,
    dealii::Triangulation<dim>                            &tria,
    const std::vector<const VectorType *>                 &cut_solutions,
    const dealii::NonMatching::MeshClassifier<dim>        &mesh_classifier_old,
    const dealii::NonMatching::MeshClassifier<dim>        &mesh_classifier,
    const std::function<void(VectorType &)>               &reinit_cut_vector,
    const std::function<void()>                           &setup_dof_system,
    const AttachDoFHandlerAndVectorsType<dim, VectorType> &attach_vectors)
  {
    fe_degree = cut_dof_handler.get_fe_collection().max_degree();

    n_components_per_phase = cut_dof_handler.get_fe_collection().n_components();
    if (is_two_phase)
      {
        AssertThrow(
          n_components_per_phase % 2 == 0,
          dealii::ExcMessage(
            "For a two phase problem, both phases must contain the same number of solution components."));
        n_components_per_phase /= 2;
      }

    // Check if the finite element is of type FE_DGQ or FE_Q.
    // Assert if different FE types are defined in the hp::FECollection object.
    {
      unsigned int dg_counter = 0;
      unsigned int cg_counter = 0;
      if (n_components_per_phase == 1 and not is_two_phase)
        for ([[maybe_unused]] const auto &fe : cut_dof_handler.get_fe_collection())
          {
            if (typeid(fe) == typeid(dealii::FE_Nothing<dim>))
              continue;
            if (typeid(fe) == typeid(dealii::FE_DGQ<dim>))
              ++dg_counter;
            else if (typeid(fe) == typeid(dealii::FE_Q<dim>))
              ++cg_counter;
            else
              AssertThrow(false, dealii::ExcMessage("You did not provide a support FE type."));
          }
      else // multi-component or two phase
        for (unsigned int i = 0; i < cut_dof_handler.get_fe_collection().size(); ++i)
          {
            const auto &fe_system =
              dynamic_cast<const dealii::FESystem<dim> &>(cut_dof_handler.get_fe_collection()[i]);
            for (unsigned int j = 0; j < fe_system.n_components(); ++j)
              {
                [[maybe_unused]] const auto &fe = fe_system.get_sub_fe(j, 1);

                if (typeid(fe) == typeid(dealii::FE_Nothing<dim>))
                  continue;
                if (typeid(fe) == typeid(dealii::FE_DGQ<dim>))
                  ++dg_counter;
                else if (typeid(fe) == typeid(dealii::FE_Q<dim>))
                  ++cg_counter;
                else
                  AssertThrow(false, dealii::ExcMessage("You did not provide a support FE type."));
              }
          }

      if (dg_counter != 0 and cg_counter == 0)
        is_dg = true;
      else if (dg_counter == 0 and cg_counter != 0)
        is_dg = false;
      else
        AssertThrow(false,
                    dealii::ExcMessage(
                      "Two different FE types in the hp::FECollection object are not supported."));
    }

    // in case of a single phase problem, check that no cells are 'outside'
    if (not is_two_phase)
      for (const auto &cell : cut_dof_handler.active_cell_iterators())
        {
          if (not cell->is_locally_owned())
            continue;

          const auto cell_location = mesh_classifier_old.location_to_level_set(cell);

          AssertThrow(
            cell_location != dealii::NonMatching::LocationToLevelSet::inside or
              cell->get_fe().n_dofs_per_cell() == 0,
            dealii::ExcMessage(
              "The active domain should have the FE index 'outside' for a single domain problem."));
        }

    Assert(cut_solutions.size() > 0,
           dealii::ExcMessage("You must attach at least one cut solution vector!"));
    new_solutions.resize(cut_solutions.size());

    for (const auto &v : cut_solutions)
      if (not v->has_ghost_elements())
        v->update_ghost_values();

    // 1) update FE-index and distribute DoFs according to new state with the moved interface
    transfer_solution_constant_dofs(cut_dof_handler,
                                    tria,
                                    cut_solutions,
                                    mesh_classifier_old,
                                    mesh_classifier,
                                    reinit_cut_vector,
                                    setup_dof_system,
                                    attach_vectors);

    // 2) set up and solve system for ghost-penalty extrapolation to determine the values of the
    //    remaining undetermined DoFs
    extrapolate_solution_new_dofs(cut_dof_handler,
                                  mesh_classifier,
                                  mesh_classifier_old,
                                  reinit_cut_vector);
  }

  template <int dim, typename number>
  void
  SolutionTransferOperator<dim, number>::reinit(
    dealii::DoFHandler<dim>                               &cut_dof_handler,
    dealii::Triangulation<dim>                            &tria,
    const VectorType                                      &cut_solution,
    const dealii::NonMatching::MeshClassifier<dim>        &mesh_classifier_old,
    const dealii::NonMatching::MeshClassifier<dim>        &mesh_classifier,
    const std::function<void(VectorType &)>               &reinit_cut_vector,
    const std::function<void()>                           &setup_dof_system,
    const AttachDoFHandlerAndVectorsType<dim, VectorType> &attach_vectors)
  {
    reinit(cut_dof_handler,
           tria,
           {&cut_solution},
           mesh_classifier_old,
           mesh_classifier,
           reinit_cut_vector,
           setup_dof_system,
           attach_vectors);
  }

  template <int dim, typename number>
  void
  SolutionTransferOperator<dim, number>::transfer_solution_constant_dofs(
    dealii::DoFHandler<dim>                               &cut_dof_handler,
    dealii::Triangulation<dim>                            &tria,
    const std::vector<const VectorType *>                 &cut_solutions,
    const dealii::NonMatching::MeshClassifier<dim>        &mesh_classifier_old,
    const dealii::NonMatching::MeshClassifier<dim>        &mesh_classifier,
    const std::function<void(VectorType &)>               &reinit_cut_vector,
    const std::function<void()>                           &setup_dof_system,
    const AttachDoFHandlerAndVectorsType<dim, VectorType> &attach_vectors)
  {
    // update the future FE-index according to the new interface position
    set_fe_index<dim>(cut_dof_handler, mesh_classifier, true /* set_future */);

    // the following is very similar to the code in refine_grid()

    DoFHandlerAndVectorDataType<dim, VectorType> data;

    if (attach_vectors)
      {
        attach_vectors(data);
        data.shrink_to_fit();
      }
    else
      {
        data.reserve(1);
        data.emplace_back(&cut_dof_handler, [&cut_solutions](std::vector<VectorType *> &vectors) {
          vectors.reserve(cut_solutions.size());
          for (const auto &v : cut_solutions)
            vectors.push_back(const_cast<VectorType *>(v));
        });
      }

    const unsigned int n_dof_handlers = data.size();

    Assert(n_dof_handlers > 0, dealii::ExcNotImplemented());

    // Initialize the solution transfer from the old grid to the new grid
    std::vector<std::unique_ptr<dealii::SolutionTransfer<dim, VectorType>>> solution_transfers(
      n_dof_handlers);

    std::vector<std::vector<VectorType *>>       new_grid_solutions(n_dof_handlers);
    std::vector<std::vector<const VectorType *>> old_grid_solutions(n_dof_handlers);
    std::vector<std::vector<bool>>               update_ghost_values(n_dof_handlers);

    for (unsigned int j = 0; j < n_dof_handlers; ++j)
      {
        data[j].second(new_grid_solutions[j]);

        old_grid_solutions[j].resize(new_grid_solutions[j].size());
        update_ghost_values[j].resize(new_grid_solutions[j].size(), true);
        for (unsigned int i = 0; i < new_grid_solutions[j].size(); ++i)
          {
            old_grid_solutions[j][i] = new_grid_solutions[j][i];
            if (not new_grid_solutions[j][i]->has_ghost_elements())
              {
                update_ghost_values[j][i] = false;
                new_grid_solutions[j][i]->update_ghost_values();
              }
          }

        solution_transfers[j] = std::make_unique<dealii::SolutionTransfer<dim, VectorType>>(
          *data[j].first, data[j].first == &cut_dof_handler /* average_values for cut solution */);
        solution_transfers[j]->prepare_for_coarsening_and_refinement(old_grid_solutions[j]);
      }

    // Note: In contrast to CutUtil::refine_grid(), here, we do not need to transfer the level set
    // solution first. But as a requirement, the lambda function setup_dof_system() cannot include
    // classifying the mesh, and cannot include generating the quadrature of intersected cells,
    // because the level set is not yet known for the new DoF system. The mesh must be classified
    // before calling CutUtil::SolutionTransferOperator::reinit(), which leads to this point, and
    // the intersected quadrature must be generated afterward. For an example, see
    // Heat::HeatCutOperation::adapt_to_new_interface_position().

    // needed to trigger hp refinement --> call call-back function created by solution transfer
    tria.execute_coarsening_and_refinement();

    // reinitialize the matrix-free object (cut_dof_handler has changed)
    setup_dof_system();

    // reinit new cut solution vectors
    for (auto &v : new_solutions)
      {
        reinit_cut_vector(v);
        v = 0.;
      }

    // interpolate the given solution to the new discretization
    for (unsigned int j = 0; j < n_dof_handlers; ++j)
      {
        if (data[j].first != &cut_dof_handler)
          {
            for (const auto &v : new_grid_solutions[j])
              v->zero_out_ghost_values();

            solution_transfers[j]->interpolate(new_grid_solutions[j]);

            for (unsigned int i = 0; i < new_grid_solutions[j].size(); ++i)
              if (update_ghost_values[j][i])
                new_grid_solutions[j][i]->update_ghost_values();
          }
        else
          solution_transfers[j]->interpolate(new_solutions);
      }

    for (const auto &v : new_solutions)
      v.update_ghost_values();

    // revert averaging process in solution transfer for cg-case
    // (TODO: fix this issue in dealii)
    if (not is_dg)
      {
        VectorType dof_counter;
        reinit_cut_vector(dof_counter);
        dof_counter = 0.;

        VectorType dof_counter_fe_nothing;
        reinit_cut_vector(dof_counter_fe_nothing);
        dof_counter_fe_nothing = 0.;

        std::vector<dealii::types::global_dof_index> dof_indices;
        for (const auto &cell : cut_dof_handler.active_cell_iterators())
          {
            if (not cell->is_locally_owned() and not cell->is_ghost())
              continue;

            dof_indices.resize(cell->get_fe().n_dofs_per_cell());
            cell->get_dof_indices(dof_indices);

            for (const auto i : dof_indices)
              if (dof_counter.in_local_range(i))
                dof_counter[i] += 1.0;

            const auto cell_location = mesh_classifier.location_to_level_set(cell);

            if (cell_location != dealii::NonMatching::LocationToLevelSet::intersected)
              continue;

            const auto cell_location_old = mesh_classifier_old.location_to_level_set(cell);

            const auto &fe = cell->get_fe();

            std::vector<dealii::types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());
            cell->get_dof_indices(local_dof_indices);

            if (cell_location_old == dealii::NonMatching::LocationToLevelSet::inside)
              for (unsigned int i = 0; i < n_components_per_phase; ++i)
                for (unsigned int q = 0; q < fe.get_sub_fe(i, 1).n_dofs_per_cell(); ++q)
                  {
                    const auto dof_index =
                      local_dof_indices[fe.component_to_system_index(i /*component*/, q)];
                    if (dof_counter_fe_nothing.in_local_range(dof_index))
                      dof_counter_fe_nothing[dof_index] += 1;
                  }
            else if ((cell_location_old == dealii::NonMatching::LocationToLevelSet::outside) and
                     (is_two_phase == true))
              for (unsigned int i = n_components_per_phase; i < 2 * n_components_per_phase; ++i)
                for (unsigned int q = 0; q < fe.get_sub_fe(i, 1).n_dofs_per_cell(); ++q)
                  {
                    const auto dof_index =
                      local_dof_indices[fe.component_to_system_index(i /*component*/, q)];
                    if (dof_counter_fe_nothing.in_local_range(dof_index))
                      dof_counter_fe_nothing[dof_index] += 1;
                  }
          }

        for (auto &new_solution : new_solutions)
          {
            for (const auto i : new_solution.locally_owned_elements())
              if (dof_counter[i] - dof_counter_fe_nothing[i] > 0)
                new_solution[i] *= dof_counter[i] / (dof_counter[i] - dof_counter_fe_nothing[i]);

            new_solution.update_ghost_values();
          }
      }
  }

  template <int dim, typename number>
  dealii::LinearAlgebra::distributed::Vector<number>
  SolutionTransferOperator<dim, number>::mark_dofs_for_gp_extrapolation(
    const dealii::DoFHandler<dim>                  &cut_dof_handler,
    const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier,
    const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier_old) const
  {
    VectorType flags_dofs_gp_extrapolation;
    flags_dofs_gp_extrapolation.reinit(cut_dof_handler.locally_owned_dofs(),
                                       dealii::DoFTools::extract_locally_relevant_dofs(
                                         cut_dof_handler),
                                       cut_dof_handler.get_mpi_communicator());
    flags_dofs_gp_extrapolation = 0.;

    // a) Loop over intersected cells that previously did not belong to the
    //    subdomain and collect the DoFs
    for (const auto &cell : cut_dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() and not cell->is_ghost())
          continue;

        const auto cell_location = mesh_classifier.location_to_level_set(cell);

        AssertThrow(
          not((cell_location == dealii::NonMatching::LocationToLevelSet::inside and
               mesh_classifier_old.location_to_level_set(cell) ==
                 dealii::NonMatching::LocationToLevelSet::outside) or
              (cell_location == dealii::NonMatching::LocationToLevelSet::outside and
               mesh_classifier_old.location_to_level_set(cell) ==
                 dealii::NonMatching::LocationToLevelSet::inside)),
          dealii::ExcMessage(
            "Invalid solution transfer. Extrapolation for interface movement over more than one cell is not implemented."));

        if (cell_location != dealii::NonMatching::LocationToLevelSet::intersected)
          continue;

        AssertThrow(static_cast<unsigned int>(cell->level()) ==
                      cut_dof_handler.get_triangulation().n_global_levels() - 1,
                    dealii::ExcMessage("Invalid solution transfer for cut cells. All intersected "
                                       "cells must be on the finest mesh level."));

        const auto cell_location_old = mesh_classifier_old.location_to_level_set(cell);

        const auto &fe = cell->get_fe();

        std::vector<dealii::types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());
        cell->get_dof_indices(local_dof_indices);

        if (cell_location_old == dealii::NonMatching::LocationToLevelSet::inside)
          for (unsigned int i = 0; i < n_components_per_phase; ++i)
            for (unsigned int q = 0; q < fe.get_sub_fe(i, 1).n_dofs_per_cell(); ++q)
              {
                const auto dof_index =
                  local_dof_indices[fe.component_to_system_index(i /*component*/, q)];
                if (flags_dofs_gp_extrapolation.in_local_range(dof_index))
                  flags_dofs_gp_extrapolation[dof_index] = 1;
              }
        else if ((cell_location_old == dealii::NonMatching::LocationToLevelSet::outside) and
                 (is_two_phase == true))
          for (unsigned int i = n_components_per_phase; i < 2 * n_components_per_phase; ++i)
            for (unsigned int q = 0; q < fe.get_sub_fe(i, 1).n_dofs_per_cell(); ++q)
              {
                const auto dof_index =
                  local_dof_indices[fe.component_to_system_index(i /*component*/, q)];
                if (flags_dofs_gp_extrapolation.in_local_range(dof_index))
                  flags_dofs_gp_extrapolation[dof_index] = 1;
              }
      }

    flags_dofs_gp_extrapolation.update_ghost_values();

    // b) For continuous elements, loop over cells belonging to the subdomain and remove DoFs
    //    from extrapolation. Additionally, if both the old and new cell location are intersected,
    //    remove DoFs from extrapolation.
    if (not is_dg)
      {
        for (const auto &cell : cut_dof_handler.active_cell_iterators())
          {
            if (not cell->is_locally_owned() and not cell->is_ghost())
              continue;

            const auto cell_location = mesh_classifier.location_to_level_set(cell);

            const auto &fe = cell->get_fe();

            std::vector<dealii::types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());
            cell->get_dof_indices(local_dof_indices);

            if (cell_location == dealii::NonMatching::LocationToLevelSet::outside)
              for (unsigned int i = 0; i < n_components_per_phase; ++i)
                for (unsigned int q = 0; q < fe.get_sub_fe(i, 1).n_dofs_per_cell(); ++q)
                  {
                    const auto dof_index =
                      local_dof_indices[fe.component_to_system_index(i /*component*/, q)];
                    if (flags_dofs_gp_extrapolation.in_local_range(dof_index))
                      flags_dofs_gp_extrapolation[dof_index] = 0;
                  }
            else if ((cell_location == dealii::NonMatching::LocationToLevelSet::inside) and
                     (is_two_phase == true))
              for (unsigned int i = n_components_per_phase; i < 2 * n_components_per_phase; ++i)
                for (unsigned int q = 0; q < fe.get_sub_fe(i, 1).n_dofs_per_cell(); ++q)
                  {
                    const auto dof_index =
                      local_dof_indices[fe.component_to_system_index(i /*component*/, q)];
                    if (flags_dofs_gp_extrapolation.in_local_range(dof_index))
                      flags_dofs_gp_extrapolation[dof_index] = 0;
                  }
            else if (cell_location == dealii::NonMatching::LocationToLevelSet::intersected)
              {
                const auto cell_location_old = mesh_classifier_old.location_to_level_set(cell);
                if (cell_location_old == dealii::NonMatching::LocationToLevelSet::intersected)
                  {
                    for (unsigned int i = 0; i < n_components_per_phase; ++i)
                      for (unsigned int q = 0; q < fe.get_sub_fe(i, 1).n_dofs_per_cell(); ++q)
                        {
                          const auto dof_index =
                            local_dof_indices[fe.component_to_system_index(i /*component*/, q)];
                          if (flags_dofs_gp_extrapolation.in_local_range(dof_index))
                            flags_dofs_gp_extrapolation[dof_index] = 0;
                        }

                    if (is_two_phase == true)
                      {
                        for (unsigned int i = n_components_per_phase;
                             i < 2 * n_components_per_phase;
                             ++i)
                          for (unsigned int q = 0; q < fe.get_sub_fe(i, 1).n_dofs_per_cell(); ++q)
                            {
                              const auto dof_index =
                                local_dof_indices[fe.component_to_system_index(i /*component*/, q)];
                              if (flags_dofs_gp_extrapolation.in_local_range(dof_index))
                                flags_dofs_gp_extrapolation[dof_index] = 0;
                            }
                      }
                  }
              }
          }
      }

    return flags_dofs_gp_extrapolation;
  }

  template <int dim, typename number>
  dealii::AffineConstraints<number>
  SolutionTransferOperator<dim, number>::create_constraints_gp_extrapolation(
    const dealii::DoFHandler<dim> &cut_dof_handler,
    const VectorType              &flags_dofs_gp_extrapolation) const
  {
    flags_dofs_gp_extrapolation.update_ghost_values();

    const dealii::Utilities::MPI::Partitioner partitioner_dof(
      cut_dof_handler.locally_owned_dofs(),
      dealii::DoFTools::extract_locally_relevant_dofs(cut_dof_handler),
      cut_dof_handler.get_mpi_communicator());

    // Check if the partitioner of new_solution is globally compatible with partitioner_dof.
    // If not, transfer new_solution to default vector structure.
    // The full set of ghost-values is required for setting the constraints.
    VectorType new_solution_including_all_ghosts;
    const bool is_globally_compatible =
      partitioner_dof.is_globally_compatible(*new_solutions[0].get_partitioner());
    if (not is_globally_compatible)
      {
        new_solution_including_all_ghosts.reinit(cut_dof_handler.locally_owned_dofs(),
                                                 dealii::DoFTools::extract_locally_relevant_dofs(
                                                   cut_dof_handler),
                                                 cut_dof_handler.get_mpi_communicator());
        new_solution_including_all_ghosts = new_solutions[0];
        new_solution_including_all_ghosts.update_ghost_values();
      }

    dealii::AffineConstraints<number> constraints_gp;
    constraints_gp.reinit(cut_dof_handler.locally_owned_dofs(),
                          dealii::DoFTools::extract_locally_relevant_dofs(cut_dof_handler));

    std::vector<bool> dof_processed(partitioner_dof.locally_owned_size() +
                                      partitioner_dof.n_ghost_indices(),
                                    false);

    for (const auto &cell : cut_dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() and not cell->is_ghost())
          continue;

        std::vector<dealii::types::global_dof_index> dof_indices(cell->get_fe().n_dofs_per_cell());
        cell->get_dof_indices(dof_indices);

        dealii::Vector<number> extrapolate_dofs(cell->get_fe().n_dofs_per_cell());
        cell->get_dof_values(flags_dofs_gp_extrapolation, extrapolate_dofs);

        for (unsigned int i = 0; i < extrapolate_dofs.size(); ++i)
          {
            const auto local_dof_idx = partitioner_dof.global_to_local(dof_indices[i]);
            AssertIndexRange(local_dof_idx, dof_processed.size());

            if (dof_processed[local_dof_idx] == false)
              {
                // constrain entries when they are not extrapolated
                if (extrapolate_dofs[i] == 0)
                  {
                    if (not is_globally_compatible)
                      constraints_gp.add_constraint(
                        dof_indices[i], {}, new_solution_including_all_ghosts[dof_indices[i]]);
                    else
                      constraints_gp.add_constraint(dof_indices[i],
                                                    {},
                                                    new_solutions[0][dof_indices[i]]);
                  }
                dof_processed[local_dof_idx] = true;
              }
          }
      }

    Assert(constraints_gp.is_consistent_in_parallel(
             dealii::Utilities::MPI::all_gather(cut_dof_handler.get_mpi_communicator(),
                                                cut_dof_handler.locally_owned_dofs()),
             dealii::DoFTools::extract_locally_active_dofs(cut_dof_handler),
             cut_dof_handler.get_mpi_communicator()),
           dealii::ExcInternalError());

    constraints_gp.close();

    return constraints_gp;
  }

  template <int dim, typename number>
  void
  SolutionTransferOperator<dim, number>::extrapolate_solution_new_dofs(
    const dealii::DoFHandler<dim>                  &cut_dof_handler,
    const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier,
    const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier_old,
    const std::function<void(VectorType &)>        &reinit_cut_vector)
  {
    // identify and mark DoFs, which require ghost-penalty extrapolation
    const VectorType flags_dofs_gp_extrapolation =
      mark_dofs_for_gp_extrapolation(cut_dof_handler, mesh_classifier, mesh_classifier_old);

    // check if DoFs have to be extrapolated, otherwise no gp-extrapolation is required
    const number sum_gp_extrap = flags_dofs_gp_extrapolation.l1_norm();

    if (sum_gp_extrap == 0.)
      return;

    // set inhomogeneous Dirichlet constraints for DoFs which do no not require ghost-penalty
    // extrapolation
    const dealii::AffineConstraints<number> constraints_gp =
      create_constraints_gp_extrapolation(cut_dof_handler, flags_dofs_gp_extrapolation);

    // set up sparsity pattern
    dealii::TrilinosWrappers::SparsityPattern dsp;
    dsp.reinit(cut_dof_handler.locally_owned_dofs(),
               cut_dof_handler.locally_owned_dofs(),
               dealii::DoFTools::extract_locally_relevant_dofs(cut_dof_handler),
               cut_dof_handler.get_mpi_communicator(),
               0);

    dealii::DoFTools::make_flux_sparsity_pattern(cut_dof_handler, dsp, constraints_gp, false);

    dsp.compress();

    if (verbosity >= 1)
      {
        std::ostringstream str;
        str << "Number of new DoFs: " << std::setw(15) << sum_gp_extrap;
        Journal::print_line(pcout, str.str(), "cut solution transfer");
      }
    if (verbosity >= 3)
      {
        std::ostringstream str;
        str << "Number of nonzero elements: " << std::setw(15) << dsp.n_nonzero_elements();
        Journal::print_line(pcout, str.str(), "cut solution transfer");
      }

    dealii::TrilinosWrappers::SparseMatrix sparse_matrix;
    sparse_matrix.reinit(dsp);

    VectorType rhs;
    reinit_cut_vector(rhs);
    rhs = 0.;

    dealii::hp::QCollection<dim - 1> face_quadrature;
    face_quadrature.push_back(dealii::QGauss<dim - 1>(fe_degree + 1));

    constexpr unsigned int invalid = dealii::numbers::invalid_unsigned_int;

    // fill matrix
    for (const auto &cell : cut_dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned())
          continue;

        const auto   cell_location    = mesh_classifier.location_to_level_set(cell);
        const number cell_side_length = cell->minimum_vertex_distance();

        if (cell_location != dealii::NonMatching::LocationToLevelSet::intersected)
          continue;

        // definition of lambda-function for ghost-penalty evaluation
        const auto eval_ghost_penalty =
          [&](dealii::FEValuesExtractors::Scalar     &u_extractor,
              dealii::NonMatching::LocationToLevelSet inactive_location) {
            dealii::UpdateFlags update_flags =
              dealii::update_gradients | dealii::update_JxW_values | dealii::update_normal_vectors;
            if (is_dg)
              update_flags = dealii::update_values | update_flags;
            if (fe_degree == 2)
              update_flags = update_flags | dealii::update_hessians;

            dealii::FEInterfaceValues<dim> fe_interface_values(cut_dof_handler.get_fe_collection(),
                                                               face_quadrature,
                                                               update_flags);

            for (const unsigned int f : cell->face_indices())
              {
                // check if current face is a ghost-penalty face
                if (not face_has_ghost_penalty(mesh_classifier, cell, f, inactive_location))
                  continue;

                fe_interface_values.reinit(
                  cell, f, invalid, cell->neighbor(f), cell->neighbor_of_neighbor(f), invalid);

                const unsigned int n_interface_dofs =
                  fe_interface_values.n_current_interface_dofs();

                // Determine prefactor, for ghost-penalty face term evaluation.
                // Prefactor 0.5 is used for faces, which are visited twice.
                number prefactor = 1.;
                if (is_new_intersected_face(
                      mesh_classifier, mesh_classifier_old, cell, f, inactive_location))
                  prefactor = 0.5;

                // initialize local ghost-penalty constraint matrix and local rhs
                dealii::FullMatrix<number> local_ghost_penalty_matrix(n_interface_dofs,
                                                                      n_interface_dofs);
                dealii::Vector<number>     local_rhs(n_interface_dofs);

                // compute entries of local ghost-penalty constraint matrix
                for (const auto i : fe_interface_values.dof_indices())
                  for (const auto j : fe_interface_values.dof_indices())
                    for (unsigned int q = 0; q < fe_interface_values.n_quadrature_points; ++q)
                      {
                        const dealii::Tensor<1, dim, number> normal =
                          fe_interface_values.normal_vector(q);

                        // contributions from 1. normal derivative jump
                        local_ghost_penalty_matrix(i, j) +=
                          prefactor * normal *
                          fe_interface_values[u_extractor].jump_in_gradients(i, q) * normal *
                          fe_interface_values[u_extractor].jump_in_gradients(j, q) *
                          cell_side_length * ghost_penalty.gamma_M_degree_1 *
                          fe_interface_values.JxW(q);

                        if (is_dg)
                          {
                            // contributions from 0. normal derivative jump
                            local_ghost_penalty_matrix(i, j) +=
                              prefactor * fe_interface_values[u_extractor].jump_in_values(i, q) *
                              fe_interface_values[u_extractor].jump_in_values(j, q) /
                              cell_side_length * ghost_penalty.gamma_M_degree_0 *
                              fe_interface_values.JxW(q);
                          }

                        if (fe_degree == 2)
                          {
                            // contributions from 2. normal derivative jump
                            local_ghost_penalty_matrix(i, j) +=
                              prefactor *
                              (normal * fe_interface_values[u_extractor].jump_in_hessians(i, q) *
                               normal /*double contraction*/) *
                              (normal * fe_interface_values[u_extractor].jump_in_hessians(j, q) *
                               normal /*double contraction*/) *
                              dealii::Utilities::fixed_power<3>(cell_side_length) *
                              ghost_penalty.gamma_M_degree_2 * fe_interface_values.JxW(q);
                          }
                      }

                // distribute local ghost-penalty constraint matrix and local_rhs to
                // global system
                const std::vector<dealii::types::global_dof_index> local_interface_dof_indices =
                  fe_interface_values.get_interface_dof_indices();

                constraints_gp.distribute_local_to_global(local_ghost_penalty_matrix,
                                                          local_rhs,
                                                          local_interface_dof_indices,
                                                          sparse_matrix,
                                                          rhs);
              }
          };

        const auto cell_location_old = mesh_classifier_old.location_to_level_set(cell);

        if (cell_location_old == dealii::NonMatching::LocationToLevelSet::inside)
          for (unsigned int i = 0; i < n_components_per_phase; ++i)
            {
              dealii::FEValuesExtractors::Scalar u(i);
              eval_ghost_penalty(u, cell_location_old);
            }
        else if (cell_location_old == dealii::NonMatching::LocationToLevelSet::outside and
                 is_two_phase == true)
          for (unsigned int i = n_components_per_phase; i < 2 * n_components_per_phase; ++i)
            {
              dealii::FEValuesExtractors::Scalar u(i);
              eval_ghost_penalty(u, cell_location_old);
            }
      }

    sparse_matrix.compress(dealii::VectorOperation::add);
    rhs.compress(dealii::VectorOperation::add);

    // solve system
    dealii::ReductionControl solver_control(1000, 1e-10, 1e-10);

    dealii::SolverCG<VectorType> solver(solver_control);
    for (auto &new_solution : new_solutions)
      solver.solve(sparse_matrix, new_solution, rhs, dealii::PreconditionIdentity());

    if (verbosity >= 2)
      {
        std::ostringstream str;
        str << "Ghost-penalty extrapolation system solved in " << solver_control.last_step()
            << " iterations.";
        Journal::print_line(pcout, str.str(), "cut_solution_transfer");
      }

    // apply constraints
    for (auto &new_solution : new_solutions)
      constraints_gp.distribute(new_solution);
  }

  template class SolutionTransferOperator<1, double>;
  template class SolutionTransferOperator<2, double>;
  template class SolutionTransferOperator<3, double>;
} // namespace MeltPoolDG::CutUtil
