# 💻 How to Work With MeltPoolDG

## Glossary

**Application:** Handles the solution of a potentially time-dependent and coupled problem between individual operations for an application-specific simulation case.

**Data:** A structured collection of input data required for a specific class.

**Operation:** Executes the solution of a single time step for a specific (sub)problem, typically including both nonlinear and linear solvers for a given operator.

**Operator:** Assembles a matrix, a vector or performs the matrix-vector product depending on the formulation (matrix-free or not).

**Parameter:** A defined set of individual data collections specific to a simulation case.

**Simulation Case:** Encompasses all user-defined and case-specific building blocks of a simulation, including triangulation, parameters, initial conditions, boundary conditions etc.

## Coding conventions

Our coding conventions are primarily based on [deal.II](https://www.dealii.org/current/doxygen/deal.II/CodingConventions.html). For unclear cases, we typically refer to the [Google C++ style guide](https://google.github.io/styleguide/cppguide.html). 

However, some key conventions we follow may differ from these sources.

### General guidelines
- The minimum required template parameters should be `number` and `dim` (if applicable)
- Use file extensions as follows:
    -  `.cpp` for source files
    -  `.hpp` for header files
    -  `.cc` for unit tests
- The minmal namespace scope is `MeltPoolDG`.
- Use `pragma once` as the header include guard.
- Avoid `using namespace dealii` in header files; always use fully qualified names (`std::string`, `std::vector`) in headers.
- Avoid public variables; instead, use getter functions to access private data.
- Prefer the alternative logical operators `not`, `and` and etc. instead of `!`, `&&`.
- Ensure **one blank line** between function definitions and declarations.
- `CamelCase` for class names or namespace names and `snake_case` for variables and functions
- Always include **one blank line at the end of each file** for compatibility and readability.
- avoid public variables and use getter functions instead
- Follow this class structure in header files:

### Class Structure Example
```cpp
// 1) Header guard
#pragma once

// 2) Included headers
#include <string>
#include <vector>

// 3) Namespace
namespace MeltPoolDG {

// 4) Forward declaration
class AnotherClass;

class MyClass {
// 5) Public section
public:
    // 6) Constructors & Destructor
    MyClass();
    explicit MyClass(const std::string& name);
    ~MyClass();

    // 7) Public Methods
    void doSomething() const;
    std::string getName() const;

// 8) Private or Protected section
private:
    // 9) Using directives
    using VectorType = std::vector<double>;

    // 10) Private members and functions
    std::string name;
    std::vector<int> data;

    void helperFunction();
};

} // namespace MeltPoolDG
```

### Documentation and comments
- Use `//` for inline comments within the code.
- Follow the Doxygen format for documentation of functions, classes, variables in header files.

## How to add new parameters

When adding new parameters to a data structure using `ParameterHandler::add_parameters()`, follow these conventions. A best-practice example can be found [here](https://github.com/MeltPoolDG/MeltPoolDG-dev/blob/master/source/level_set/advection_diffusion_data.cpp).
- Use meaningful variable names that are self-explanatory, avoiding abbreviations.
- Carefully decide whether to provide useful default values or require users to explicitly set a value by initializing it to an invalid state.
- Use `BetterEnum` over `std::string`. This allows for faster comparisons and reduces the risk of errors by automatically triggering an assertion for invalid types.
- Use [dealii::Patterns](https://dealii.org/current/doxygen/deal.II/namespacePatterns.html) to validate the parameter.
- If the pattern-based validation is insufficient, add an assertion during the `check_input_parameters()` operation to prevent invalid parameters.
- Provide a helpful description of the meaning of the parameter. If applicable, also add SI units in the description.



