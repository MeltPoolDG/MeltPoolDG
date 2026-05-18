# mp-reinit: Parameter description

## Contents

- [`base`](#base)
- [`time stepping`](#time-stepping)
- [`adaptive meshing`](#adaptive-meshing)
- [`reinitialization`](#reinitialization)
- [`normal vector`](#normal-vector)
- [`curvature`](#curvature)
- [`output`](#output)
- [`profiling`](#profiling)
- [`application specific parameters`](#application-specific-parameters)

---

<a id="base"></a>
## `🔷 base`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `case name` | `string` | `not_initialized` | Sets the base name for the application that will be fed to the problem type. |
| `dimension` | `integer` | `2` | Defines the dimension of the problem |
| `number` | `string` | `double` | Floating point number format. Currently, only 'double' is explicitely instantiated.<br><br>Allowed values:<br>- `double` |
| `global refinements` | `integer` | `1` | Defines the number of initial global refinements |
| `do print parameters` | `boolean` | `True` | Set this parameter to true to list parameters in output |
| `verbosity level` | `integer` | `1` | Sets the verbosity level of the console output: 0: silent: for non-robust tests and benchmark runs; 1: minimal: for robust tests; 2: detailed; 3: full |
| [`fe`](#base-fe) | `object` |  | [See table](#base-fe) |

<a id="base-fe"></a>
### `base: fe`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `FE_Q` | Finite Element.FE_Q: hexahedral continuous finite element with polynomial degree p; FE_SimplexP: tetrahedral continuous finite element with polynomial degree p; FE_Q_iso_Q1: hexahedral continuous finite element with p subdivisions containing linear elements; FE_DGQ: hexahedral discontinuous finite element with polynomial degree p<br><br>Allowed values:<br>- `not_initialized`<br>- `FE_Q`<br>- `FE_SimplexP`<br>- `FE_Q_iso_Q1`<br>- `FE_DGQ` |
| `degree` | `integer` | `1` | Defines the degree p of the finite element type. If "type" is "FE_Q_iso_Q1" this parameter defines the number of subdivisions. |


---

<a id="time-stepping"></a>
## `🔷 time stepping`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `start time` | `number` | `0.0` | Defines the start time for the solution of the levelset problem |
| `end time` | `number` | `1.0` | Sets the end time for the solution of the levelset problem |
| `time step size` | `number` | `0.01` | Sets the step size for time stepping. For non-uniform time stepping, this parameter determines the size of the first time step. |
| `max n steps` | `integer` | `10000000` | Sets the maximum number of melt_pool steps |
| `time step size function` | `string` | `0.0*t` | Set an analytical function to determine the time step size. For the prediction of the new time increment, the old time is used. |


---

<a id="adaptive-meshing"></a>
## `🔷 adaptive meshing`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `do amr` | `boolean` | `False` | Set this parameter to true to activate adaptive meshing |
| `do not modify boundary cells` | `boolean` | `False` | Set this parameter to true to not refine/coarsen along boundaries. |
| `upper perc to refine` | `number` | `0.0` | Defines the (upper) percentage of elements that should be refined |
| `lower perc to coarsen` | `number` | `0.0` | Defines the (lower) percentage of elements that should be coarsened |
| `max grid refinement level` | `integer` | `12` | Defines the number of maximum refinement steps one grid cell will be undergone. |
| `min grid refinement level` | `integer` | `-1` | Defines the number of minimum refinement steps one grid cell will be undergone. |
| `n initial refinement cycles` | `integer` | `0` | Defines the number of initial refinements. |
| `every n step` | `integer` | `1` | Defines at every nth step the amr should be performed. |
| `min cells marked to refine` | `integer` | `1` | Minimum number of cells that must be marked for refinement/coarsening before the mesh is updated. |
| `min indicator threshold to refine cell` | `number` | `0.0` | Minimum indicator value required for a cell to be considered for refinement. |
| `solution transfer average values` | `boolean` | `False` | Set this parameter to true to average the contribututions to the same DoF coming from different cells during solution transfer. |


---

<a id="reinitialization"></a>
## `🔷 reinitialization`

| Parameter | Type | Default | Description |
|---|---|---|---|
| [`fe`](#reinitialization-fe) | `object` |  | [See table](#reinitialization-fe) |
| `enable` | `boolean` | `True` | Set to true to activate reinitialization. |
| `n initial steps` | `integer` | `-1` | Defines the number of initial reinitialization steps of the level set function. In the default case, the number is set equal to the number of max n steps. |
| `pseudo time step size` | `number` | `-1.0` | Sets the reinitialization time step size. By default its computed from the cell size. |
| `pseudo time step factor` | `number` | `1.0` | Factor on the reinitialization time step size that is computed from the cell size. |
| `max n steps` | `integer` | `5` | Sets the maximum number of reinitialization steps |
| `tolerance` | `number` | `2.22507e-308` | Set the tolerance for reinitialization. If the maximum change of the level set field, i.e.  orΔФ or∞, exceeds the tolerance, reinitialization steps will be performed. |
| `tangential diffusion factor` | `number` | `0.0` | Factor that multiplies the normal diffusion factor (diffusion length) to obtain the diffusion factor in the tangential direction. |
| `type` | `string` | `olsson2007` | Sets the type of reinitialization model that should be used.<br><br>Allowed values:<br>- `olsson2007`<br>- `elliptic` |
| `implementation` | `string` | `meltpooldg` | Choose the corresponding implementation of the reinitialization operation.<br><br>Allowed values:<br>- `meltpooldg`<br>- `adaflo` |
| [`Discontinous Galerkin`](#reinitialization-discontinous-galerkin) | `object` |  | [See table](#reinitialization-discontinous-galerkin) |
| [`elliptic`](#reinitialization-elliptic) | `object` |  | [See table](#reinitialization-elliptic) |
| [`interface thickness parameter`](#reinitialization-interface-thickness-parameter) | `object` |  | [See table](#reinitialization-interface-thickness-parameter) |
| [`predictor`](#reinitialization-predictor) | `object` |  | [See table](#reinitialization-predictor) |
| [`linear solver`](#reinitialization-linear-solver) | `object` |  | [See table](#reinitialization-linear-solver) |

<a id="reinitialization-fe"></a>
### `reinitialization: fe`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `not_initialized` | Finite Element.FE_Q: hexahedral continuous finite element with polynomial degree p; FE_SimplexP: tetrahedral continuous finite element with polynomial degree p; FE_Q_iso_Q1: hexahedral continuous finite element with p subdivisions containing linear elements; FE_DGQ: hexahedral discontinuous finite element with polynomial degree p<br><br>Allowed values:<br>- `not_initialized`<br>- `FE_Q`<br>- `FE_SimplexP`<br>- `FE_Q_iso_Q1`<br>- `FE_DGQ` |
| `degree` | `integer` | `-1` | Defines the degree p of the finite element type. If "type" is "FE_Q_iso_Q1" this parameter defines the number of subdivisions. |

<a id="reinitialization-discontinous-galerkin"></a>
### `reinitialization: Discontinous Galerkin`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `factor diffusivity` | `number` | `0.25` | Set the factor for diffusivity |
| `IP diffusion` | `number` | `100.0` | Set the internal penalty for diffusivity |
| `use const gradient in RI` | `boolean` | `False` | Set if the Godunov gradient should be updated every reinitilization step |
| `do CFL based time stepping` | `boolean` | `False` | Sets a flag if the time stepping should be based on the CFL condition |
| `time integration scheme` | `string` | `LSRK_stage_5_order_4` | Determines the general time integration scheme for the pseudo time integration of the reinilization equation.<br><br>Allowed values:<br>- `not_initialized`<br>- `LSRK_stage_1_order_1`<br>- `LSRK_stage_3_order_3`<br>- `LSRK_stage_5_order_4`<br>- `LSRK_stage_7_order_4`<br>- `LSRK_stage_9_order_5`<br>- `implicit_euler`<br>- `explicit_euler`<br>- `crank_nicolson`<br>- `bdf_1`<br>- `bdf_2`<br>- `bdf_3`<br>- `bdf_4`<br>- `bdf_5`<br>- `bdf_6`<br>- `imex` |
| `IMEX integration scheme` | `string` | `not_initialized` | If a IMEX integration scheme is specifiead, the integration in pseudo time of the reinilization is done with an Implict-Explicit (IMEX) scheme.this means that the diffusion part is treated with the IMEX integration scheme and the hamiltonian is treated with the general time integration scheme.When choosing an implicit scheme with A-stability larger time steps can be chosen only limited by the stability of the hamiltonian part.This is done, since the diffusion part is the most restrictive part for explicit time integration scheme.If a scheme is set, the time step calculation based on a CFL number assumes an A-stable scheme and only calculates the time step based on the hamiltonian.<br><br>Allowed values:<br>- `not_initialized`<br>- `LSRK_stage_1_order_1`<br>- `LSRK_stage_3_order_3`<br>- `LSRK_stage_5_order_4`<br>- `LSRK_stage_7_order_4`<br>- `LSRK_stage_9_order_5`<br>- `implicit_euler`<br>- `explicit_euler`<br>- `crank_nicolson`<br>- `bdf_1`<br>- `bdf_2`<br>- `bdf_3`<br>- `bdf_4`<br>- `bdf_5`<br>- `bdf_6`<br>- `imex` |
| `CFL` | `number` | `1.0` | Set a CFL number for the pseudo time stepping in reinitilization. |
| `avoid zero division smoothed signum` | `number` | `1e-16` | Sets a constant to avoid zero division in the computation of the smoothed signum. |
| `signum smoothness paramater` | `number` | `2.0` | Sets the smoothness parameter for the smoothed signum. |
| `use directed diffusion stabilization` | `boolean` | `False` | Sets a flag if directed diffusion stabilization should be used for reinitilization. |
| `hyperbolic weighting function_type` | `string` | `smoothed_signum` | Sets the type of weighting function for the hyperbolic part of the reinit equation.<br><br>Allowed values:<br>- `smoothed_signum`<br>- `initial_levelset` |
| `use spatially constant diffusion` | `boolean` | `True` | Sets a flag if a spatially constant diffusion should be used for reinitilization. |
| `use interface movement penalization` | `boolean` | `False` | Sets a flag if a penalization of the interface movement should be used. |
| `gradient error time derivative threshold` | `number` | `1e-16` | Sets the threshold in the time derivative when a reinit procedure reaches a stationary point |

<a id="reinitialization-elliptic"></a>
### `reinitialization: elliptic`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `penalty parameter` | `number` | `0.0` | Penalty parameter for the enforcement of the initial position of the zero level-set iso-surface during the elliptic reinitialization. |

<a id="reinitialization-interface-thickness-parameter"></a>
### `reinitialization: interface thickness parameter`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `proportional_to_cell_size` | Choose the value type of the interface thickness parameter.<br><br>Allowed values:<br>- `proportional_to_cell_size`<br>- `absolute_value`<br>- `number_of_cells_across_interface` |
| `val` | `number` | `0.5` | Defines the value of the chosen interface thickness parameter type |

<a id="reinitialization-predictor"></a>
### `reinitialization: predictor`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `none` | Choose a predictor type: none: use old value as initial guess; zero: se zeros as initial guess; linear_extrapolation: calculate the predictor by a linear combination from the two old solution vectors; least_squares_projection: least squares projection (WIP)<br><br>Allowed values:<br>- `none`<br>- `zero`<br>- `linear_extrapolation`<br>- `least_squares_projection` |
| `n old solutions` | `integer` | `2` | Choose the number of old solution vectors considered.This parameter is only relevant for least squares projection.For all other predictors, this parameter will be set appropriately. |

<a id="reinitialization-linear-solver"></a>
### `reinitialization: linear solver`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `solver type` | `string` | `CG` | Set this parameter for choosing an iterative linear solver type.<br><br>Allowed values:<br>- `CG`<br>- `GMRES` |
| `preconditioner type` | `string` | `Diagonal` | Set this parameter for choosing a preconditioner type.<br><br>Allowed values:<br>- `Identity`<br>- `AMG`<br>- `ILU`<br>- `Diagonal` |
| `max iterations` | `integer` | `10000` | Set the maximum number of iterations for solving the linear system of equations. |
| `rel tolerance` | `number` | `1e-12` | Set the relative tolerance for a successful solution of the linear system of equations. |
| `abs tolerance` | `number` | `1e-20` | Set the absolute tolerance for a successful solution of the linear system of equations. |
| `do matrix free` | `boolean` | `True` | Set this parameter if a matrix free solution procedure should be performed. |
| `monitor type` | `string` | `none` | Set the monitor type of the linear solver.<br><br>Allowed values:<br>- `none`<br>- `reduced`<br>- `all` |


---

<a id="normal-vector"></a>
## `🔷 normal vector`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filter parameter` | `number` | `2.0` | normal vector computation: damping = (cell size)²  * filter parameter |
| `implementation` | `string` | `meltpooldg` | Choose the corresponding implementation of the normal vector operation.<br><br>Allowed values:<br>- `meltpooldg`<br>- `adaflo` |
| `verbosity level` | `integer` | `-1` | Sets the maximum verbosity level of the console output. The maximum level with respect to the  base value is decisive. |
| `compute normalized vector` | `boolean` | `False` | If set to true, the normal vector resulting from the filtering equation will be a unit vector. |
| [`narrow band`](#normal-vector-narrow-band) | `object` |  | [See table](#normal-vector-narrow-band) |
| [`predictor`](#normal-vector-predictor) | `object` |  | [See table](#normal-vector-predictor) |
| [`linear solver`](#normal-vector-linear-solver) | `object` |  | [See table](#normal-vector-linear-solver) |
| [`Discontinous Galerkin`](#normal-vector-discontinous-galerkin) | `object` |  | [See table](#normal-vector-discontinous-galerkin) |

<a id="normal-vector-narrow-band"></a>
### `normal vector: narrow band`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable` | `boolean` | `False` | Set this parameter to true to compute the normal vector only in the interfacial region. |
| `level set threshold` | `number` | `1.0` | If narrow band is enabled to true this parameter determines the level set treshold for the narrow band. |

<a id="normal-vector-predictor"></a>
### `normal vector: predictor`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `none` | Choose a predictor type: none: use old value as initial guess; zero: se zeros as initial guess; linear_extrapolation: calculate the predictor by a linear combination from the two old solution vectors; least_squares_projection: least squares projection (WIP)<br><br>Allowed values:<br>- `none`<br>- `zero`<br>- `linear_extrapolation`<br>- `least_squares_projection` |
| `n old solutions` | `integer` | `2` | Choose the number of old solution vectors considered.This parameter is only relevant for least squares projection.For all other predictors, this parameter will be set appropriately. |

<a id="normal-vector-linear-solver"></a>
### `normal vector: linear solver`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `solver type` | `string` | `CG` | Set this parameter for choosing an iterative linear solver type.<br><br>Allowed values:<br>- `CG`<br>- `GMRES` |
| `preconditioner type` | `string` | `Diagonal` | Set this parameter for choosing a preconditioner type.<br><br>Allowed values:<br>- `Identity`<br>- `AMG`<br>- `ILU`<br>- `Diagonal` |
| `max iterations` | `integer` | `10000` | Set the maximum number of iterations for solving the linear system of equations. |
| `rel tolerance` | `number` | `1e-12` | Set the relative tolerance for a successful solution of the linear system of equations. |
| `abs tolerance` | `number` | `1e-20` | Set the absolute tolerance for a successful solution of the linear system of equations. |
| `do matrix free` | `boolean` | `True` | Set this parameter if a matrix free solution procedure should be performed. |
| `monitor type` | `string` | `none` | Set the monitor type of the linear solver.<br><br>Allowed values:<br>- `none`<br>- `reduced`<br>- `all` |

<a id="normal-vector-discontinous-galerkin"></a>
### `normal vector: Discontinous Galerkin`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `penalty factor` | `number` | `100.0` | Set the jump penalty factor of the diffusion term |


---

<a id="curvature"></a>
## `🔷 curvature`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable` | `boolean` | `True` | Set this parameter to true if curvature should be computed. This is required in case of surface tension forces. |
| `do curvature correction` | `boolean` | `False` | Set this parameter to true if the curvature value at the discrete interface i.e. where the level set is 0, should be extended to the interface region. |
| `filter parameter` | `number` | `2.0` | curvature computation: damping = (cell size)² * filter parameter |
| `implementation` | `string` | `meltpooldg` | Choose the corresponding implementation of the curvature operation.<br><br>Allowed values:<br>- `meltpooldg`<br>- `adaflo` |
| `verbosity level` | `integer` | `-1` | Sets the maximum verbosity level of the console output. The maximum level with respect to the base value is decisive. |
| [`narrow band`](#curvature-narrow-band) | `object` |  | [See table](#curvature-narrow-band) |
| [`Discontinous Galerkin`](#curvature-discontinous-galerkin) | `object` |  | [See table](#curvature-discontinous-galerkin) |
| [`predictor`](#curvature-predictor) | `object` |  | [See table](#curvature-predictor) |
| [`linear solver`](#curvature-linear-solver) | `object` |  | [See table](#curvature-linear-solver) |

<a id="curvature-narrow-band"></a>
### `curvature: narrow band`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable` | `boolean` | `False` | Set this parameter to true to compute the normal vector only in the interfacial region. |
| `level set threshold` | `number` | `1.0` | If narrow band is enabled to true this parameter determines the level set treshold for the narrow band. |

<a id="curvature-discontinous-galerkin"></a>
### `curvature: Discontinous Galerkin`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `penalty factor` | `number` | `100.0` | Set the jump penalty factor of the diffusion term |

<a id="curvature-predictor"></a>
### `curvature: predictor`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `none` | Choose a predictor type: none: use old value as initial guess; zero: se zeros as initial guess; linear_extrapolation: calculate the predictor by a linear combination from the two old solution vectors; least_squares_projection: least squares projection (WIP)<br><br>Allowed values:<br>- `none`<br>- `zero`<br>- `linear_extrapolation`<br>- `least_squares_projection` |
| `n old solutions` | `integer` | `2` | Choose the number of old solution vectors considered.This parameter is only relevant for least squares projection.For all other predictors, this parameter will be set appropriately. |

<a id="curvature-linear-solver"></a>
### `curvature: linear solver`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `solver type` | `string` | `CG` | Set this parameter for choosing an iterative linear solver type.<br><br>Allowed values:<br>- `CG`<br>- `GMRES` |
| `preconditioner type` | `string` | `Diagonal` | Set this parameter for choosing a preconditioner type.<br><br>Allowed values:<br>- `Identity`<br>- `AMG`<br>- `ILU`<br>- `Diagonal` |
| `max iterations` | `integer` | `10000` | Set the maximum number of iterations for solving the linear system of equations. |
| `rel tolerance` | `number` | `1e-12` | Set the relative tolerance for a successful solution of the linear system of equations. |
| `abs tolerance` | `number` | `1e-20` | Set the absolute tolerance for a successful solution of the linear system of equations. |
| `do matrix free` | `boolean` | `True` | Set this parameter if a matrix free solution procedure should be performed. |
| `monitor type` | `string` | `none` | Set the monitor type of the linear solver.<br><br>Allowed values:<br>- `none`<br>- `reduced`<br>- `all` |


---

<a id="output"></a>
## `🔷 output`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `directory` | `string` | `./` | Sets the base directory for all output. |
| `write frequency` | `integer` | `1` | Every n timestep that should be written |
| `write time step size` | `number` | `1.79769e+308` | Write output output every given time step. If this parameter is set, the output write frequency is deactivated. |
| `output variables` | `array` | `['all']` | Specify variables that you request to output. |
| `do user defined postprocessing` | `boolean` | `False` | Set this parameter to true to enable user defined postprocessing. |
| [`paraview`](#output-paraview) | `object` |  | [See table](#output-paraview) |
| [`particles`](#output-particles) | `object` |  | [See table](#output-particles) |

<a id="output-paraview"></a>
### `output: paraview`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable` | `boolean` | `False` | Set this parameter to true to activate paraview output. |
| `filename` | `string` | `solution` | Sets the base name for paraview output files. |
| `n digits timestep` | `integer` | `4` | Number of digits for the frame number of the vtu-file. |
| `print boundary id` | `boolean` | `False` | Set this parameter to true to output a vtu-file with the boundary id. |
| `output subdomains` | `boolean` | `False` | Set this parameter to true to output the subdomain ranks. |
| `output material id` | `boolean` | `False` | Set to true to output the material id. |
| `write higher order cells` | `boolean` | `True` | Set this parameter to false to write bi- or trilinear data only. Set this parameter to true to write higher order cell data. Note: higher order cell data can only be written for hexahedron meshes and 2 or 3 dimensions. |
| `n groups` | `integer` | `1` | Number of parallel written vtu-files. |
| `n patches` | `integer` | `0` | Control number of patches to enable high-order. |

<a id="output-particles"></a>
### `output: particles`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable` | `boolean` | `False` | Set this parameter to true to activate particle paraview output. |
| `filename` | `string` | `particle` | Sets the base name for particle output files. |


---

<a id="profiling"></a>
## `🔷 profiling`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable` | `boolean` | `False` | Set this parameter to true if profiling should be enabled. It will be automaticallyenabled for verbosity level >=1. |
| `write time step size` | `number` | `10.0` | Write profiling output every given time step size. If this parameter is set, the specified parameter for write frequency is overwritten. |
| `time type` | `string` | `real` | Choose the type of time measure to write profiling information.<br><br>Allowed values:<br>- `real`<br>- `simulation` |


---

<a id="application-specific-parameters"></a>
## `🔷 application specific parameters`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `do update normal vector` | `boolean` | `False` | Set this parameter to true to update the normal vector each pseudo-time step. |


---
