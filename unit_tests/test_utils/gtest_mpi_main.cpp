#include <gtest/gtest.h>

#include <deal.II/base/mpi.h>


int
main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  // All ranks receive --gtest_output, but only rank 0 may write the XML
  // report; otherwise all ranks would write to the same file.
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != 0)
    {
      auto &listeners = ::testing::UnitTest::GetInstance()->listeners();
      delete listeners.Release(listeners.default_xml_generator());
    }

  return RUN_ALL_TESTS();
}
