/**
 * @file test_generators.hpp
 * @brief Functions that generate the inputs shared by several unit tests, e.g. subdivided
 * hyper-rectangular triangulations or reproducible pseudo-arbitrary values.
 */

#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <random>

namespace MeltPoolDG::TestUtils
{
  /**
   * Return a pseudo-arbitrary value drawn uniformly from [@p lower_bound, @p upper_bound).
   *
   * All calls with the same @p number type share one random number generator, which is seeded with
   * @p seed on the first call only. Hence, the values are reproducible for a fixed sequence of
   * calls, but depend on all previous calls in the same program.
   */
  template <typename number>
  number
  arbitrary_value(const number       lower_bound = -73.,
                  const number       upper_bound = 73.,
                  const unsigned int seed        = 73)
  {
    static std::mt19937                    gen(seed);
    std::uniform_real_distribution<number> dist(lower_bound, upper_bound);
    return dist(gen);
  }

  /**
   * Set @p value to a pseudo-arbitrary value in [@p lower_bound, @p upper_bound), see the overload
   * above.
   */
  template <typename number>
  void
  arbitrary_value(number &value, const number lower_bound = -73., const number upper_bound = 73.)
  {
    value = arbitrary_value(lower_bound, upper_bound);
  }

  /**
   * Set each lane of @p value to its own pseudo-arbitrary value in [@p lower_bound,
   * @p upper_bound).
   */
  template <typename number>
  void
  arbitrary_value(dealii::VectorizedArray<number> &value,
                  const number                     lower_bound = -73.,
                  const number                     upper_bound = 73.,
                  const unsigned int               seed        = 73)
  {
    for (unsigned int lane = 0; lane < dealii::VectorizedArray<number>::size(); ++lane)
      value[lane] = arbitrary_value(lower_bound, upper_bound, seed);
  }

  /**
   * Set each component of @p value to its own pseudo-arbitrary value in [@p lower_bound,
   * @p upper_bound). Each lane in the vectorized array gets its own value.
   */
  template <int n_components, typename number>
  void
  arbitrary_value(dealii::Tensor<1, n_components, dealii::VectorizedArray<number>> &value,
                  const number       lower_bound = -73.,
                  const number       upper_bound = 73.,
                  const unsigned int seed        = 73)
  {
    for (unsigned int c = 0; c < n_components; ++c)
      arbitrary_value(value[c], lower_bound, upper_bound, seed);
  }

  /**
   * Create a subdivided hyper rectangle whose extent and number of cells differ per direction, such
   * that the cells are anisotropic. In direction `d`, the domain spans [-0.5 * (d + 1), 1 + d] and
   * is divided into 3 + d cells.
   *
   * @param triangulation The triangulation to be created.
   */
  template <int dim>
  void
  create_test_subdivided_hyper_rectangle(dealii::Triangulation<dim> &triangulation)
  {
    dealii::Point<dim>        lower_left;
    dealii::Point<dim>        upper_right;
    std::vector<unsigned int> repetitions(dim);
    for (unsigned int d = 0; d < dim; ++d)
      {
        lower_left[d]  = -0.5 * (d + 1);
        upper_right[d] = 1. + d;
        repetitions[d] = 3 + d;
      }
    dealii::GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                      repetitions,
                                                      lower_left,
                                                      upper_right);
  }
} // namespace MeltPoolDG::TestUtils
