# mp-advec-diff: Parameter description

## Contents

- [`base`](#base)
- [`time stepping`](#time-stepping)
- [`adaptive meshing`](#adaptive-meshing)
- [`advection diffusion`](#advection-diffusion)
- [`output`](#output)
- [`profiling`](#profiling)

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

<a id="advection-diffusion"></a>
## `🔷 advection diffusion`

| Parameter | Type | Default | Description |
|---|---|---|---|
| [`fe`](#advection-diffusion-fe) | `object` |  | [See table](#advection-diffusion-fe) |
| [`convection stabilization`](#advection-diffusion-convection-stabilization) | `object` |  | [See table](#advection-diffusion-convection-stabilization) |
| `diffusivity` | `number` | `0.0` | Defines the diffusivity for the advection diffusion equation |
| `implementation` | `string` | `meltpooldg` | Choose the corresponding implementation of the advection diffusion operation.<br><br>Allowed values:<br>- `meltpooldg`<br>- `adaflo` |
| `enable time dependent bc` | `boolean` | `False` | Set this parameter to true to enable time-dependent bc. |
| [`predictor`](#advection-diffusion-predictor) | `object` |  | [See table](#advection-diffusion-predictor) |
| [`linear solver`](#advection-diffusion-linear-solver) | `object` |  | [See table](#advection-diffusion-linear-solver) |
| [`time integration`](#advection-diffusion-time-integration) | `object` |  | [See table](#advection-diffusion-time-integration) |

<a id="advection-diffusion-fe"></a>
### `advection diffusion: fe`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `not_initialized` | Finite Element.FE_Q: hexahedral continuous finite element with polynomial degree p; FE_SimplexP: tetrahedral continuous finite element with polynomial degree p; FE_Q_iso_Q1: hexahedral continuous finite element with p subdivisions containing linear elements; FE_DGQ: hexahedral discontinuous finite element with polynomial degree p<br><br>Allowed values:<br>- `not_initialized`<br>- `FE_Q`<br>- `FE_SimplexP`<br>- `FE_Q_iso_Q1`<br>- `FE_DGQ` |
| `degree` | `integer` | `-1` | Defines the degree p of the finite element type. If "type" is "FE_Q_iso_Q1" this parameter defines the number of subdivisions. |

<a id="advection-diffusion-convection-stabilization"></a>
### `advection diffusion: convection stabilization`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `none` | Defines the type for convection stabilization.<br><br>Allowed values:<br>- `none`<br>- `SUPG` |
| `coefficient` | `number` | `-1.0` | Defines the stabilization coefficient for convection. (default velocity-dependent). |

<a id="advection-diffusion-predictor"></a>
### `advection diffusion: predictor`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `none` | Choose a predictor type: none: use old value as initial guess; zero: se zeros as initial guess; linear_extrapolation: calculate the predictor by a linear combination from the two old solution vectors; least_squares_projection: least squares projection (WIP)<br><br>Allowed values:<br>- `none`<br>- `zero`<br>- `linear_extrapolation`<br>- `least_squares_projection` |
| `n old solutions` | `integer` | `2` | Choose the number of old solution vectors considered.This parameter is only relevant for least squares projection.For all other predictors, this parameter will be set appropriately. |

<a id="advection-diffusion-linear-solver"></a>
### `advection diffusion: linear solver`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `solver type` | `string` | `GMRES` | Set this parameter for choosing an iterative linear solver type.<br><br>Allowed values:<br>- `CG`<br>- `GMRES` |
| `preconditioner type` | `string` | `Diagonal` | Set this parameter for choosing a preconditioner type.<br><br>Allowed values:<br>- `Identity`<br>- `AMG`<br>- `ILU`<br>- `Diagonal` |
| `max iterations` | `integer` | `10000` | Set the maximum number of iterations for solving the linear system of equations. |
| `rel tolerance` | `number` | `1e-12` | Set the relative tolerance for a successful solution of the linear system of equations. |
| `abs tolerance` | `number` | `1e-20` | Set the absolute tolerance for a successful solution of the linear system of equations. |
| `do matrix free` | `boolean` | `True` | Set this parameter if a matrix free solution procedure should be performed. |
| `monitor type` | `string` | `none` | Set the monitor type of the linear solver.<br><br>Allowed values:<br>- `none`<br>- `reduced`<br>- `all` |

<a id="advection-diffusion-time-integration"></a>
### `advection diffusion: time integration`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `crank_nicolson` | Name of the time integration scheme.<br><br>Allowed values:<br>- `not_initialized`<br>- `LSRK_stage_1_order_1`<br>- `LSRK_stage_3_order_3`<br>- `LSRK_stage_5_order_4`<br>- `LSRK_stage_7_order_4`<br>- `LSRK_stage_9_order_5`<br>- `implicit_euler`<br>- `explicit_euler`<br>- `crank_nicolson`<br>- `bdf_1`<br>- `bdf_2`<br>- `bdf_3`<br>- `bdf_4`<br>- `bdf_5`<br>- `bdf_6`<br>- `imex` |
| `preconditioner update frequency` | `integer` | `100` | Frequency at which the preconditioner gets updated. |
| [`nlsolve`](#advection-diffusion-time-integration-nlsolve) | `object` |  | [See table](#advection-diffusion-time-integration-nlsolve) |
| [`linear solver`](#advection-diffusion-time-integration-linear-solver) | `object` |  | [See table](#advection-diffusion-time-integration-linear-solver) |

<a id="advection-diffusion-time-integration-nlsolve"></a>
#### `advection diffusion: time integration: nlsolve`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max nonlinear iterations` | `integer` | `10` | Set the number of maximum nonlinear iterations with standard tolerances. |
| `field correction tolerance` | `number` | `1e-10` | Set the tolerance for the maximum allowed correction of the unknown field. |
| `residual tolerance` | `number` | `1e-09` | Set the tolerance for the maximum allowed residual of the nonlinear system. |
| `max nonlinear iterations alt` | `integer` | `0` | Set the number of maximum nonlinear iterations with alternative tolerances. |
| `field correction tolerance alt` | `number` | `1e-09` | Set the alternative tolerance for the maximum allowed correction of the unknown field. |
| `residual tolerance alt` | `number` | `1e-08` | Set the alternative tolerance for the maximum allowed residual of the nonlinear system. |
| `verbosity level` | `integer` | `-1` | Set to one for detailed solver output. |

<a id="advection-diffusion-time-integration-linear-solver"></a>
#### `advection diffusion: time integration: linear solver`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `solver type` | `string` | `GMRES` | Set this parameter for choosing an iterative linear solver type.<br><br>Allowed values:<br>- `CG`<br>- `GMRES` |
| `preconditioner type` | `string` | `Identity` | Set this parameter for choosing a preconditioner type.<br><br>Allowed values:<br>- `Identity`<br>- `AMG`<br>- `ILU`<br>- `Diagonal` |
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
