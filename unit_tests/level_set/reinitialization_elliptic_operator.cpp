#include <gtest/gtest.h>

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/level_set/reinitialization_data.hpp>
#include <meltpooldg/level_set/reinitialization_elliptic_operator_fixed_point.hpp>

#include <mpi.h>

namespace MeltPoolDG::LevelSet
{
  namespace
  {
    struct MPIInitializer
    {
      MPIInitializer()
      {
        int    argc = 0;
        char **argv = nullptr;
        MPI_Init(&argc, &argv);
      }

      ~MPIInitializer()
      {
        MPI_Finalize();
      }
    };

    const MPIInitializer mpi_initializer;


    template <int dim>
    struct EllipticOperatorFixture
    {
      using number          = double;
      using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
      using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;
      using MappingInfoType = CutUtil::MappingInfoType<dim, number>;

      dealii::Triangulation<dim>                                triangulation;
      dealii::FE_Q<dim>                                         fe{2};
      dealii::DoFHandler<dim>                                   dof_handler{triangulation};
      dealii::AffineConstraints<number>                         constraints;
      dealii::MappingQ1<dim>                                    mapping;
      MappingInfoType                                           mapping_info_surface;
      dealii::QGauss<dim>                                       quadrature{fe.degree + 1};
      MeltPoolDG::ScratchData<dim, dim, number>                 scratch_data;
      ReinitializationData<number>                              reinit_data;
      BlockVectorType                                           normal_vector;
      VectorType                                                level_set;
      std::shared_ptr<dealii::NonMatching::MeshClassifier<dim>> mesh_classifier;

      EllipticOperatorFixture()
        : mapping_info_surface(mapping,
                               dealii::update_values | dealii::update_gradients |
                                 dealii::update_JxW_values | dealii::update_normal_vectors)
        , scratch_data(MPI_COMM_WORLD, 0, true)
      {}

      void
      setup_rectangular_grid(const std::vector<unsigned int> &subdivisions,
                             const dealii::Point<dim>        &lower_left,
                             const dealii::Point<dim>        &upper_right)
      {
        dealii::GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                          subdivisions,
                                                          lower_left,
                                                          upper_right);
        dof_handler.distribute_dofs(fe);

        constraints.close();

        scratch_data.reinit(
          mapping, {&dof_handler}, {&constraints}, {quadrature}, false, false, false);

        scratch_data.initialize_dof_vector(normal_vector, 0);
        scratch_data.initialize_dof_vector(level_set, 0);
        level_set = 1.0;

        mesh_classifier =
          std::make_shared<dealii::NonMatching::MeshClassifier<dim>>(dof_handler, level_set);
      }

      ReinitializationEllipticOperator<dim, number>
      make_operator()
      {
        return ReinitializationEllipticOperator<dim, number>(
          scratch_data, reinit_data, 0, 0, mapping_info_surface, 0, mesh_classifier);
      }
    };

    template <int dim>
    void
    run_constructor_smoke_test()
    {
      EllipticOperatorFixture<dim> fixture;
      const dealii::Point<dim>     upper_right = []() {
        dealii::Point<dim> p;
        for (unsigned int d = 0; d < dim; ++d)
          p[d] = 1.0;
        return p;
      }();

      fixture.setup_rectangular_grid(std::vector<unsigned int>(dim, 2),
                                     dealii::Point<dim>(),
                                     upper_right);

      ASSERT_NO_THROW({
        auto op = fixture.make_operator();
        (void)op;
      });
    }
  } // namespace

  TEST(ReinitializationEllipticOperatorTest, ConstructorSmokeTest2D)
  {
    run_constructor_smoke_test<2>();
  }
} // namespace MeltPoolDG::LevelSet
