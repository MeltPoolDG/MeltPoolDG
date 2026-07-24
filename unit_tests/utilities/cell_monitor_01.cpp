#include <gtest/gtest.h>

#include <deal.II/base/exceptions.h>

#include <meltpooldg/utilities/cell_monitor.hpp>

#include <string>

namespace
{
  using Monitor = MeltPoolDG::CellMonitor<double>;

  class CellMonitorTest : public ::testing::Test
  {
  protected:
    void
    SetUp() override
    {
      Monitor::clear();
    }

    void
    TearDown() override
    {
      Monitor::clear();
    }
  };
} // namespace


TEST_F(CellMonitorTest, InitializesStatisticsFromFirstCall)
{
  const std::string label = "mesh";

  Monitor::add_info(label, 10, 0.10, 0.50);

  const auto statistics = Monitor::get_statistics(label);

  EXPECT_EQ(statistics.n_calls, 1u);
  EXPECT_DOUBLE_EQ(statistics.n_cells_averaged, 10.0);

  EXPECT_EQ(statistics.cells_min, 10u);
  EXPECT_EQ(statistics.cells_max, 10u);

  EXPECT_DOUBLE_EQ(statistics.cell_size_min, 0.10);
  EXPECT_DOUBLE_EQ(statistics.cell_size_max, 0.50);
}


TEST_F(CellMonitorTest, UpdatesStatisticsOverMultipleCalls)
{
  const std::string label = "mesh";

  Monitor::add_info(label, 10, 0.10, 0.50);
  Monitor::add_info(label, 20, 0.05, 0.60);
  Monitor::add_info(label, 30, 0.08, 0.40);

  const auto statistics = Monitor::get_statistics(label);

  EXPECT_EQ(statistics.n_calls, 3u);
  EXPECT_DOUBLE_EQ(statistics.n_cells_averaged, 20.0);

  EXPECT_EQ(statistics.cells_min, 10u);
  EXPECT_EQ(statistics.cells_max, 30u);

  EXPECT_DOUBLE_EQ(statistics.cell_size_min, 0.05);
  EXPECT_DOUBLE_EQ(statistics.cell_size_max, 0.60);
}


TEST_F(CellMonitorTest, ComputesNonIntegerAverage)
{
  const std::string label = "mesh";

  Monitor::add_info(label, 10, 0.10, 0.50);
  Monitor::add_info(label, 11, 0.10, 0.50);

  const auto statistics = Monitor::get_statistics(label);

  EXPECT_EQ(statistics.n_calls, 2u);
  EXPECT_DOUBLE_EQ(statistics.n_cells_averaged, 10.5);
}


TEST_F(CellMonitorTest, TracksMinimumAndMaximumValues)
{
  const std::string label = "mesh";

  Monitor::add_info(label, 25, 0.20, 0.40);
  Monitor::add_info(label, 10, 0.05, 0.35);
  Monitor::add_info(label, 40, 0.10, 0.60);

  const auto statistics = Monitor::get_statistics(label);

  EXPECT_EQ(statistics.cells_min, 10u);
  EXPECT_EQ(statistics.cells_max, 40u);

  EXPECT_DOUBLE_EQ(statistics.cell_size_min, 0.05);
  EXPECT_DOUBLE_EQ(statistics.cell_size_max, 0.60);
}


TEST_F(CellMonitorTest, KeepsLabelsIndependent)
{
  Monitor::add_info("coarse", 10, 0.10, 0.50);
  Monitor::add_info("coarse", 20, 0.08, 0.60);

  Monitor::add_info("fine", 100, 0.01, 0.05);
  Monitor::add_info("fine", 200, 0.005, 0.08);

  const auto coarse_statistics = Monitor::get_statistics("coarse");
  const auto fine_statistics   = Monitor::get_statistics("fine");

  EXPECT_EQ(coarse_statistics.n_calls, 2u);
  EXPECT_DOUBLE_EQ(coarse_statistics.n_cells_averaged, 15.0);
  EXPECT_EQ(coarse_statistics.cells_min, 10u);
  EXPECT_EQ(coarse_statistics.cells_max, 20u);
  EXPECT_DOUBLE_EQ(coarse_statistics.cell_size_min, 0.08);
  EXPECT_DOUBLE_EQ(coarse_statistics.cell_size_max, 0.60);

  EXPECT_EQ(fine_statistics.n_calls, 2u);
  EXPECT_DOUBLE_EQ(fine_statistics.n_cells_averaged, 150.0);
  EXPECT_EQ(fine_statistics.cells_min, 100u);
  EXPECT_EQ(fine_statistics.cells_max, 200u);
  EXPECT_DOUBLE_EQ(fine_statistics.cell_size_min, 0.005);
  EXPECT_DOUBLE_EQ(fine_statistics.cell_size_max, 0.08);
}


TEST_F(CellMonitorTest, ThrowsForUnknownLabel)
{
  EXPECT_THROW(Monitor::get_statistics("unknown"), dealii::ExceptionBase);
}


TEST_F(CellMonitorTest, ClearRemovesStoredStatistics)
{
  Monitor::add_info("mesh", 10, 0.10, 0.50);

  EXPECT_NO_THROW(Monitor::get_statistics("mesh"));

  Monitor::clear();

  EXPECT_THROW(Monitor::get_statistics("mesh"), dealii::ExceptionBase);
}

TEST_F(CellMonitorTest, ComputesAverageForDifferentLargeCellCounts)
{
  constexpr unsigned int first  = 3'000'000'000u;
  constexpr unsigned int second = 4'000'000'000u;

  Monitor::add_info("mesh", first, 0.10, 0.50);
  Monitor::add_info("mesh", second, 0.05, 0.60);

  const auto statistics = Monitor::get_statistics("mesh");

  EXPECT_EQ(statistics.n_calls, 2u);
  EXPECT_DOUBLE_EQ(statistics.n_cells_averaged, 3'500'000'000.0);

  EXPECT_EQ(statistics.cells_min, first);
  EXPECT_EQ(statistics.cells_max, second);
}
