#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <cstddef>
#include <type_traits>

namespace dealii
{
  template <typename number, std::size_t N>
  Tensor<1, 1, VectorizedArray<number, N>>
  operator+=(const Tensor<1, 1, VectorizedArray<number, N>> &vec,
             const VectorizedArray<number, N>               &scalar)
  {
    auto temp = vec;
    temp[0] += scalar;

    return temp;
  }

  template <typename number, std::size_t N>
  Tensor<1, 1, VectorizedArray<number, N>>
  operator+=(const VectorizedArray<number, N>               &scalar,
             const Tensor<1, 1, VectorizedArray<number, N>> &vec)
  {
    auto temp = vec;
    temp[0] += scalar;

    return temp;
  }

  template <typename number, std::size_t N>
  Tensor<1, 1, VectorizedArray<number, N>>
  operator-=(const Tensor<1, 1, VectorizedArray<number, N>> &vec,
             const VectorizedArray<number, N>               &scalar)
  {
    auto temp = vec;
    temp[0] -= scalar;

    return temp;
  }

  template <typename number, std::size_t N>
  Tensor<1, 1, VectorizedArray<number, N>>
  operator-=(const VectorizedArray<number, N>               &scalar,
             const Tensor<1, 1, VectorizedArray<number, N>> &vec)
  {
    auto temp = -vec;
    temp[0] += scalar;

    return temp;
  }
} // namespace dealii

namespace MeltPoolDG
{
  template <typename number, std::size_t N>
  dealii::VectorizedArray<number, N>
  scalar_product(const dealii::VectorizedArray<number, N>                       &scalar,
                 const dealii::Tensor<1, 1, dealii::VectorizedArray<number, N>> &vec)
  {
    return vec[0] * scalar;
  }

  template <typename number, std::size_t N>
  dealii::VectorizedArray<number, N>
  scalar_product(const dealii::Tensor<1, 1, dealii::VectorizedArray<number, N>> &vec,
                 const dealii::VectorizedArray<number, N>                       &scalar)
  {
    return vec[0] * scalar;
  }

  /**
   * Helper functions for matrix-vector and matrix-matrix computations when both matrix and vector
   * are implemented as Tensor.
   */
  template <int n_rows, int n_columns, typename number>
  dealii::Tensor<1, n_rows, number>
  matrix_vector_product(
    const dealii::Tensor<1, n_rows, dealii::Tensor<1, n_columns, number>> &matrix,
    const dealii::Tensor<1, n_columns, number>                            &vector)
  {
    dealii::Tensor<1, n_rows, number> result;
    for (unsigned int i = 0; i < n_rows; ++i)
      for (unsigned int j = 0; j < n_columns; ++j)
        result[i] += matrix[i][j] * vector[j];
    return result;
  }

  template <int a, int b, int c, typename number>
  dealii::Tensor<1, a, dealii::Tensor<1, c, number>>
  matrix_matrix_product(const dealii::Tensor<1, a, dealii::Tensor<1, b, number>> &matrix1,
                        const dealii::Tensor<1, b, dealii::Tensor<1, c, number>> &matrix2)
  {
    dealii::Tensor<1, a, dealii::Tensor<1, c, number>> result;
    for (unsigned int i = 0; i < a; ++i)
      for (unsigned int j = 0; j < c; ++j)
        for (unsigned int k = 0; k < b; ++k)
          result[i][j] += matrix1[i][k] * matrix2[k][j];
    return result;
  }

  /**
   * Contracts the given second order tensor with a vector, which results in a vector. Note that
   * the second order tensor is provided as two nested first order tensors in this function.
   *
   * @param tensor Second order tensor of type 'Tensor<1, dim_1, Tensor<1, dim_2, number>>'
   * @param vector Vector of type 'Tensor<1, dim_2, number>'
   *
   * @return Result of the contraction, i.e. result_i = tensor_ij * vector_j. The result has vector
   * type 'Tensor<1, dim_1, number>'.
   *
   * @ note The dimensions of the provided tensor and the vector have to match, i.e. the second
   * basis of the tensor and the vector must have the same dimension @p dim_2.
   */
  template <int dim_1, int dim_2, typename number>
  dealii::Tensor<1, dim_1, number>
  contract_tensor_with_vector(
    const dealii::Tensor<1, dim_1, dealii::Tensor<1, dim_2, number>> &tensor,
    const dealii::Tensor<1, dim_2, number>                           &vector)
  {
    dealii::Tensor<1, dim_1, number> result;

    for (unsigned int i = 0; i < dim_1; ++i)
      result[i] = tensor[i] * vector;

    return result;
  }

  /**
   * Contracts the average of two given second order tensors with a vector, which results in a
   * vector. Note that the second order tensors are provided as two nested first order tensors in
   * this function.
   *
   * @param tensor_1 First second order tensor of type 'Tensor<1, dim_1, Tensor<1, dim_2, number>>'
   * @param tensor_2 Second second order tensor of type 'Tensor<1, dim_1, Tensor<1, dim_2, number>>'
   * @param vector Vector of type 'Tensor<1, dim_2, number>'
   *
   * @return Result of the contraction, i.e. result_i = 0.5 * (tensor_1_ij + tensor_2_ij) * vector_j.
   * The result has vector type 'Tensor<1, dim_1, VectorizedArray<number>>'.
   *
   * @note The dimensions of the provided tensors and the vector have to match, i.e. the second
   * basis of the tensors and the vector must have the same dimension @p dim_2.
   */
  template <int dim_1, int dim_2, typename number>
  dealii::Tensor<1, dim_1, number>
  contract_average_tensor_with_vector(
    const dealii::Tensor<1, dim_1, dealii::Tensor<1, dim_2, number>> &tensor_1,
    const dealii::Tensor<1, dim_1, dealii::Tensor<1, dim_2, number>> &tensor_2,
    const dealii::Tensor<1, dim_2, number>                           &vector)
  {
    dealii::Tensor<1, dim_1, number> result;

    for (unsigned int i = 0; i < dim_1; ++i)
      result[i] = (tensor_1[i] + tensor_2[i]) * vector;

    return 0.5 * result;
  }

  template <int dim, typename number>
  dealii::Tensor<1, dim, dealii::VectorizedArray<number>>
  normalize(const dealii::VectorizedArray<number> &in, const number zero = 1e-16)
  {
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> vec;

    for (unsigned int v = 0; v < dealii::VectorizedArray<number>::size(); ++v)
      vec[0][v] = in[v] >= zero ? 1.0 : -1.0;

    return vec;
  }

  template <int dim, typename number>
  dealii::Tensor<1, dim, dealii::VectorizedArray<number>>
  normalize(const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &in,
            const number                                                   zero = 1e-16)
  {
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> vec;

    const auto n_norm = in.norm();

    for (unsigned int v = 0; v < dealii::VectorizedArray<number>::size(); ++v)
      if (n_norm[v] > zero)
        for (unsigned int d = 0; d < dim; ++d)
          vec[d][v] = in[d][v] / n_norm[v];
      else
        for (unsigned int d = 0; d < dim; ++d)
          vec[d][v] = in[d][v] / zero;

    return vec;
  }

  /**
   * Compute the dyadic product of two rank-1 tensors passing a pointer to the start of the values
   * of the two tensors.
   */
  template <int T1_dim, int T2_dim, typename number>
  dealii::Tensor<1, T1_dim, dealii::Tensor<1, T2_dim, number>>
  dyadic_product(const number *a_start, const number *b_start)
  {
    dealii::Tensor<1, T1_dim, dealii::Tensor<1, T2_dim, number>> c;
    for (unsigned int i = 0; i < T1_dim; ++i)
      {
        for (unsigned int j = 0; j < T2_dim; ++j)
          {
            c[i][j] = *(a_start + i) * *(b_start + j);
          }
      }
    return c;
  }

  /**
   * Compute the dyadic product of two rank-1 tensors
   */
  template <int T1_dim, int T2_dim, typename number>
  dealii::Tensor<1, T1_dim, dealii::Tensor<1, T2_dim, number>>
  dyadic_product(const dealii::Tensor<1, T1_dim, number> &a,
                 const dealii::Tensor<1, T2_dim, number> &b)
  {
    return dyadic_product<T1_dim, T2_dim, number>(&a[0], &b[0]);
  }

  /**
   * Return the transpose of a dealii::Tensor<dealii::Tensor>
   */
  template <int T1_dim, int T2_dim, typename number>
  dealii::Tensor<1, T2_dim, dealii::Tensor<1, T1_dim, number>>
  transpose(const dealii::Tensor<1, T1_dim, dealii::Tensor<1, T2_dim, number>> &in)
  {
    dealii::Tensor<1, T2_dim, dealii::Tensor<1, T1_dim, number>> out;
    for (unsigned int i = 0; i < T1_dim; ++i)
      for (unsigned int j = 0; j < T2_dim; ++j)
        out[j][i] = in[i][j];
    return out;
  }

  /**
   * Return the trace of a dealii::Tensor<dealii::Tensor>
   */
  template <int dim, typename number>
  number
  trace(const dealii::Tensor<1, dim, dealii::Tensor<1, dim, number>> &in)
  {
    number tr(0.0);
    for (unsigned int i = 0; i < dim; ++i)
      tr += in[i][i];
    return tr;
  }

  template <int dim, typename number>
  number
  trace(const dealii::Tensor<2, dim, number> &in)
  {
    number tr(0.0);
    for (unsigned int i = 0; i < dim; ++i)
      tr += in[i][i];
    return tr;
  }

  /**
   * Return identity matrix
   */
  template <int dim, typename number>
  dealii::Tensor<1, dim, dealii::Tensor<1, dim, number>>
  identity()
  {
    dealii::Tensor<1, dim, dealii::Tensor<1, dim, number>> out;
    for (unsigned int i = 0; i < dim; ++i)
      out[i][i] = number(1.0);
    return out;
  }

  /**
   * Compute the jump between two tensors.
   */
  template <int dim1, int dim2, typename number>
  dealii::Tensor<1, dim1, dealii::Tensor<1, dim2, number>>
  jump(const dealii::Tensor<1, dim1, number> &tensor_m,
       const dealii::Tensor<1, dim1, number> &tensor_p,
       const dealii::Tensor<1, dim2, number> &normal)
  {
    dealii::Tensor<1, dim1, dealii::Tensor<1, dim2, number>> jump;
    for (unsigned int i = 0; i < dim1; ++i)
      for (unsigned int j = 0; j < dim2; ++j)
        jump[i][j] = (tensor_m[i] - tensor_p[i]) * normal[j];
    return jump;
  }

  /*
   * Trait to detect whether a type is a specialization of `dealii::Tensor`. The trait consists of
   * a primary template defaulting to `false` and a specialized version for `dealii::Tensor` types
   * inheriting from `std::true_type`.
   */
  template <typename>
  struct is_dealii_tensor : std::false_type
  {};

  template <int rank, int components, typename number>
  struct is_dealii_tensor<dealii::Tensor<rank, components, number>> : std::true_type
  {};


} // namespace MeltPoolDG
