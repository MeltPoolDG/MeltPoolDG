#pragma once
#include <deal.II/base/exceptions.h>

#include <string>

#define AssertThrowZeroTimeIncrement(dt) AssertThrow(dt > 0, MeltPoolDG::ExcZeroTimeIncrement())

namespace MeltPoolDG
{
  DeclException1(ExcBCAlreadyAssigned,
                 std::string,
                 << "You try to attach a " << arg1 << " boundary condition "
                 << "for a boundary_id for which a boundary condition is already "
                 << "specified. Check your input related to boundary conditions!");

  DeclException2(ExcFieldNotAttached,
                 std::string,
                 std::string,
                 << "It seems that you have not called SimulationCaseBase::" << arg1
                 << "() for the operator \"" << arg2 << "\". You can do that, e.g., "
                 << "in your simulation by overriding SimulationCaseBase::set_field_conditions().");
  DeclExceptionMsg(ExcZeroTimeIncrement,
                   "It seems that the time increment is zero. Make sure that "
                   "the time increment is larger than zero.");
  DeclExceptionMsg(ExcNewtonDidNotConverge, "The Newton-Raphson solver did not converge.");
  DeclExceptionMsg(ExcFixedPointDidNotConverge, "The fixed point solver did not converge.");
  DeclExceptionMsg(ExcHeatTransferNoConvergence, "The heat transfer solver did not converge.");
  DeclException2(ExcInvalidCSVInputColumns,
                 unsigned int,
                 unsigned int,
                 << "Expected " << arg1 << " columns in each line of the CSV file, but got " << arg2
                 << ". Please check your input file.");
  DeclException1(
    ExcFailedToConvertStringToNumber,
    std::string,
    << "Failed to convert the string \"" << arg1
    << "\" to a number. Please ensure that the string is a valid representation of a number and "
    << "does not contain any extraneous characters.");
} // namespace MeltPoolDG
