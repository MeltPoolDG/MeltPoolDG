# mp-level-set: Parameter description

## Contents

- [`base`](#base)
- [`time stepping`](#time-stepping)
- [`adaptive meshing`](#adaptive-meshing)
- [`level set`](#level-set)
- [`output`](#output)
- [`profiling`](#profiling)
- [`evaporation`](#evaporation)
- [`material`](#material)
- [`amr`](#amr)

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

<a id="level-set"></a>
## `🔷 level set`

| Parameter | Type | Default | Description |
|---|---|---|---|
| [`fe`](#level-set-fe) | `object` |  | [See table](#level-set-fe) |
| `do localized heaviside` | `boolean` | `True` | Determine if the heaviside representation of the level set should be calculated as a localized function, being exactly 0 and 1 outside of the interface region. |
| `gradient error evaluation distance cell proportion` | `number` | `3.0` | Factor how many cell diameters away the gradient error should be evaluated |
| [`nearest point`](#level-set-nearest-point) | `object` |  | [See table](#level-set-nearest-point) |
| [`advection diffusion`](#level-set-advection-diffusion) | `object` |  | [See table](#level-set-advection-diffusion) |
| [`normal vector`](#level-set-normal-vector) | `object` |  | [See table](#level-set-normal-vector) |
| [`curvature`](#level-set-curvature) | `object` |  | [See table](#level-set-curvature) |
| [`reinitialization`](#level-set-reinitialization) | `object` |  | [See table](#level-set-reinitialization) |

<a id="level-set-fe"></a>
### `level set: fe`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `not_initialized` | Finite Element.FE_Q: hexahedral continuous finite element with polynomial degree p; FE_SimplexP: tetrahedral continuous finite element with polynomial degree p; FE_Q_iso_Q1: hexahedral continuous finite element with p subdivisions containing linear elements; FE_DGQ: hexahedral discontinuous finite element with polynomial degree p<br><br>Allowed values:<br>- `not_initialized`<br>- `FE_Q`<br>- `FE_SimplexP`<br>- `FE_Q_iso_Q1`<br>- `FE_DGQ` |
| `degree` | `integer` | `-1` | Defines the degree p of the finite element type. If "type" is "FE_Q_iso_Q1" this parameter defines the number of subdivisions. |

<a id="level-set-nearest-point"></a>
### `level set: nearest point`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max iter` | `integer` | `20` | Maximum number of corrections of the point projection towards the interface. |
| `rel tol` | `number` | `1e-06` | Relative tolerance to be achieved within the projection. |
| `narrow band threshold` | `number` | `-1.0` | Maximum value of the level set for defining narrow band where CPP is performed. |
| `type` | `string` | `closest_point_normal` | Choose the type for calculating the nearest point to the interface.<br><br>Allowed values:<br>- `closest_point_normal`<br>- `closest_point_normal_collinear`<br>- `closest_point_normal_collinear_coquerelle`<br>- `nearest_point`<br>- `nearest_point_fast` |
| `verbosity level` | `integer` | `0` | Set the verbosity level. |
| [`marching cube`](#level-set-nearest-point-marching-cube) | `object` |  | [See table](#level-set-nearest-point-marching-cube) |

<a id="level-set-nearest-point-marching-cube"></a>
#### `level set: nearest point: marching cube`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n subdivisions` | `integer` | `3` | Specify the number of subdivisions to create a quadrature rule with n_subdivisions+1 equally-positioned quadrature points. |
| `tol` | `number` | `1e-10` | Absolute tolerance specifying the minimum distance between a vertex and the cut point so that a line is considered cut. |

<a id="level-set-advection-diffusion"></a>
### `level set: advection diffusion`

| Parameter | Type | Default | Description |
|---|---|---|---|
| [`fe`](#level-set-advection-diffusion-fe) | `object` |  | [See table](#level-set-advection-diffusion-fe) |
| [`convection stabilization`](#level-set-advection-diffusion-convection-stabilization) | `object` |  | [See table](#level-set-advection-diffusion-convection-stabilization) |
| `diffusivity` | `number` | `0.0` | Defines the diffusivity for the advection diffusion equation |
| `implementation` | `string` | `meltpooldg` | Choose the corresponding implementation of the advection diffusion operation.<br><br>Allowed values:<br>- `meltpooldg`<br>- `adaflo` |
| `enable time dependent bc` | `boolean` | `False` | Set this parameter to true to enable time-dependent bc. |
| [`predictor`](#level-set-advection-diffusion-predictor) | `object` |  | [See table](#level-set-advection-diffusion-predictor) |
| [`linear solver`](#level-set-advection-diffusion-linear-solver) | `object` |  | [See table](#level-set-advection-diffusion-linear-solver) |
| [`time integration`](#level-set-advection-diffusion-time-integration) | `object` |  | [See table](#level-set-advection-diffusion-time-integration) |

<a id="level-set-advection-diffusion-fe"></a>
#### `level set: advection diffusion: fe`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `not_initialized` | Finite Element.FE_Q: hexahedral continuous finite element with polynomial degree p; FE_SimplexP: tetrahedral continuous finite element with polynomial degree p; FE_Q_iso_Q1: hexahedral continuous finite element with p subdivisions containing linear elements; FE_DGQ: hexahedral discontinuous finite element with polynomial degree p<br><br>Allowed values:<br>- `not_initialized`<br>- `FE_Q`<br>- `FE_SimplexP`<br>- `FE_Q_iso_Q1`<br>- `FE_DGQ` |
| `degree` | `integer` | `-1` | Defines the degree p of the finite element type. If "type" is "FE_Q_iso_Q1" this parameter defines the number of subdivisions. |

<a id="level-set-advection-diffusion-convection-stabilization"></a>
#### `level set: advection diffusion: convection stabilization`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `none` | Defines the type for convection stabilization.<br><br>Allowed values:<br>- `none`<br>- `SUPG` |
| `coefficient` | `number` | `-1.0` | Defines the stabilization coefficient for convection. (default velocity-dependent). |

<a id="level-set-advection-diffusion-predictor"></a>
#### `level set: advection diffusion: predictor`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `none` | Choose a predictor type: none: use old value as initial guess; zero: se zeros as initial guess; linear_extrapolation: calculate the predictor by a linear combination from the two old solution vectors; least_squares_projection: least squares projection (WIP)<br><br>Allowed values:<br>- `none`<br>- `zero`<br>- `linear_extrapolation`<br>- `least_squares_projection` |
| `n old solutions` | `integer` | `2` | Choose the number of old solution vectors considered.This parameter is only relevant for least squares projection.For all other predictors, this parameter will be set appropriately. |

<a id="level-set-advection-diffusion-linear-solver"></a>
#### `level set: advection diffusion: linear solver`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `solver type` | `string` | `GMRES` | Set this parameter for choosing an iterative linear solver type.<br><br>Allowed values:<br>- `CG`<br>- `GMRES` |
| `preconditioner type` | `string` | `Diagonal` | Set this parameter for choosing a preconditioner type.<br><br>Allowed values:<br>- `Identity`<br>- `AMG`<br>- `ILU`<br>- `Diagonal` |
| `max iterations` | `integer` | `10000` | Set the maximum number of iterations for solving the linear system of equations. |
| `rel tolerance` | `number` | `1e-12` | Set the relative tolerance for a successful solution of the linear system of equations. |
| `abs tolerance` | `number` | `1e-20` | Set the absolute tolerance for a successful solution of the linear system of equations. |
| `do matrix free` | `boolean` | `True` | Set this parameter if a matrix free solution procedure should be performed. |
| `monitor type` | `string` | `none` | Set the monitor type of the linear solver.<br><br>Allowed values:<br>- `none`<br>- `reduced`<br>- `all` |

<a id="level-set-advection-diffusion-time-integration"></a>
#### `level set: advection diffusion: time integration`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `crank_nicolson` | Name of the time integration scheme.<br><br>Allowed values:<br>- `not_initialized`<br>- `LSRK_stage_1_order_1`<br>- `LSRK_stage_3_order_3`<br>- `LSRK_stage_5_order_4`<br>- `LSRK_stage_7_order_4`<br>- `LSRK_stage_9_order_5`<br>- `implicit_euler`<br>- `explicit_euler`<br>- `crank_nicolson`<br>- `bdf_1`<br>- `bdf_2`<br>- `bdf_3`<br>- `bdf_4`<br>- `bdf_5`<br>- `bdf_6`<br>- `imex` |
| `preconditioner update frequency` | `integer` | `100` | Frequency at which the preconditioner gets updated. |
| [`nlsolve`](#level-set-advection-diffusion-time-integration-nlsolve) | `object` |  | [See table](#level-set-advection-diffusion-time-integration-nlsolve) |
| [`linear solver`](#level-set-advection-diffusion-time-integration-linear-solver) | `object` |  | [See table](#level-set-advection-diffusion-time-integration-linear-solver) |

<a id="level-set-advection-diffusion-time-integration-nlsolve"></a>
##### `level set: advection diffusion: time integration: nlsolve`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max nonlinear iterations` | `integer` | `10` | Set the number of maximum nonlinear iterations with standard tolerances. |
| `field correction tolerance` | `number` | `1e-10` | Set the tolerance for the maximum allowed correction of the unknown field. |
| `residual tolerance` | `number` | `1e-09` | Set the tolerance for the maximum allowed residual of the nonlinear system. |
| `max nonlinear iterations alt` | `integer` | `0` | Set the number of maximum nonlinear iterations with alternative tolerances. |
| `field correction tolerance alt` | `number` | `1e-09` | Set the alternative tolerance for the maximum allowed correction of the unknown field. |
| `residual tolerance alt` | `number` | `1e-08` | Set the alternative tolerance for the maximum allowed residual of the nonlinear system. |
| `verbosity level` | `integer` | `-1` | Set to one for detailed solver output. |

<a id="level-set-advection-diffusion-time-integration-linear-solver"></a>
##### `level set: advection diffusion: time integration: linear solver`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `solver type` | `string` | `GMRES` | Set this parameter for choosing an iterative linear solver type.<br><br>Allowed values:<br>- `CG`<br>- `GMRES` |
| `preconditioner type` | `string` | `Identity` | Set this parameter for choosing a preconditioner type.<br><br>Allowed values:<br>- `Identity`<br>- `AMG`<br>- `ILU`<br>- `Diagonal` |
| `max iterations` | `integer` | `10000` | Set the maximum number of iterations for solving the linear system of equations. |
| `rel tolerance` | `number` | `1e-12` | Set the relative tolerance for a successful solution of the linear system of equations. |
| `abs tolerance` | `number` | `1e-20` | Set the absolute tolerance for a successful solution of the linear system of equations. |
| `do matrix free` | `boolean` | `True` | Set this parameter if a matrix free solution procedure should be performed. |
| `monitor type` | `string` | `none` | Set the monitor type of the linear solver.<br><br>Allowed values:<br>- `none`<br>- `reduced`<br>- `all` |

<a id="level-set-normal-vector"></a>
### `level set: normal vector`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filter parameter` | `number` | `2.0` | normal vector computation: damping = (cell size)²  * filter parameter |
| `implementation` | `string` | `meltpooldg` | Choose the corresponding implementation of the normal vector operation.<br><br>Allowed values:<br>- `meltpooldg`<br>- `adaflo` |
| `verbosity level` | `integer` | `-1` | Sets the maximum verbosity level of the console output. The maximum level with respect to the  base value is decisive. |
| `compute normalized vector` | `boolean` | `False` | If set to true, the normal vector resulting from the filtering equation will be a unit vector. |
| [`narrow band`](#level-set-normal-vector-narrow-band) | `object` |  | [See table](#level-set-normal-vector-narrow-band) |
| [`predictor`](#level-set-normal-vector-predictor) | `object` |  | [See table](#level-set-normal-vector-predictor) |
| [`linear solver`](#level-set-normal-vector-linear-solver) | `object` |  | [See table](#level-set-normal-vector-linear-solver) |
| [`Discontinous Galerkin`](#level-set-normal-vector-discontinous-galerkin) | `object` |  | [See table](#level-set-normal-vector-discontinous-galerkin) |

<a id="level-set-normal-vector-narrow-band"></a>
#### `level set: normal vector: narrow band`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable` | `boolean` | `False` | Set this parameter to true to compute the normal vector only in the interfacial region. |
| `level set threshold` | `number` | `1.0` | If narrow band is enabled to true this parameter determines the level set treshold for the narrow band. |

<a id="level-set-normal-vector-predictor"></a>
#### `level set: normal vector: predictor`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `none` | Choose a predictor type: none: use old value as initial guess; zero: se zeros as initial guess; linear_extrapolation: calculate the predictor by a linear combination from the two old solution vectors; least_squares_projection: least squares projection (WIP)<br><br>Allowed values:<br>- `none`<br>- `zero`<br>- `linear_extrapolation`<br>- `least_squares_projection` |
| `n old solutions` | `integer` | `2` | Choose the number of old solution vectors considered.This parameter is only relevant for least squares projection.For all other predictors, this parameter will be set appropriately. |

<a id="level-set-normal-vector-linear-solver"></a>
#### `level set: normal vector: linear solver`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `solver type` | `string` | `CG` | Set this parameter for choosing an iterative linear solver type.<br><br>Allowed values:<br>- `CG`<br>- `GMRES` |
| `preconditioner type` | `string` | `Diagonal` | Set this parameter for choosing a preconditioner type.<br><br>Allowed values:<br>- `Identity`<br>- `AMG`<br>- `ILU`<br>- `Diagonal` |
| `max iterations` | `integer` | `10000` | Set the maximum number of iterations for solving the linear system of equations. |
| `rel tolerance` | `number` | `1e-12` | Set the relative tolerance for a successful solution of the linear system of equations. |
| `abs tolerance` | `number` | `1e-20` | Set the absolute tolerance for a successful solution of the linear system of equations. |
| `do matrix free` | `boolean` | `True` | Set this parameter if a matrix free solution procedure should be performed. |
| `monitor type` | `string` | `none` | Set the monitor type of the linear solver.<br><br>Allowed values:<br>- `none`<br>- `reduced`<br>- `all` |

<a id="level-set-normal-vector-discontinous-galerkin"></a>
#### `level set: normal vector: Discontinous Galerkin`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `penalty factor` | `number` | `100.0` | Set the jump penalty factor of the diffusion term |

<a id="level-set-curvature"></a>
### `level set: curvature`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable` | `boolean` | `True` | Set this parameter to true if curvature should be computed. This is required in case of surface tension forces. |
| `do curvature correction` | `boolean` | `False` | Set this parameter to true if the curvature value at the discrete interface i.e. where the level set is 0, should be extended to the interface region. |
| `filter parameter` | `number` | `2.0` | curvature computation: damping = (cell size)² * filter parameter |
| `implementation` | `string` | `meltpooldg` | Choose the corresponding implementation of the curvature operation.<br><br>Allowed values:<br>- `meltpooldg`<br>- `adaflo` |
| `verbosity level` | `integer` | `-1` | Sets the maximum verbosity level of the console output. The maximum level with respect to the base value is decisive. |
| [`narrow band`](#level-set-curvature-narrow-band) | `object` |  | [See table](#level-set-curvature-narrow-band) |
| [`Discontinous Galerkin`](#level-set-curvature-discontinous-galerkin) | `object` |  | [See table](#level-set-curvature-discontinous-galerkin) |
| [`predictor`](#level-set-curvature-predictor) | `object` |  | [See table](#level-set-curvature-predictor) |
| [`linear solver`](#level-set-curvature-linear-solver) | `object` |  | [See table](#level-set-curvature-linear-solver) |

<a id="level-set-curvature-narrow-band"></a>
#### `level set: curvature: narrow band`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable` | `boolean` | `False` | Set this parameter to true to compute the normal vector only in the interfacial region. |
| `level set threshold` | `number` | `1.0` | If narrow band is enabled to true this parameter determines the level set treshold for the narrow band. |

<a id="level-set-curvature-discontinous-galerkin"></a>
#### `level set: curvature: Discontinous Galerkin`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `penalty factor` | `number` | `100.0` | Set the jump penalty factor of the diffusion term |

<a id="level-set-curvature-predictor"></a>
#### `level set: curvature: predictor`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `none` | Choose a predictor type: none: use old value as initial guess; zero: se zeros as initial guess; linear_extrapolation: calculate the predictor by a linear combination from the two old solution vectors; least_squares_projection: least squares projection (WIP)<br><br>Allowed values:<br>- `none`<br>- `zero`<br>- `linear_extrapolation`<br>- `least_squares_projection` |
| `n old solutions` | `integer` | `2` | Choose the number of old solution vectors considered.This parameter is only relevant for least squares projection.For all other predictors, this parameter will be set appropriately. |

<a id="level-set-curvature-linear-solver"></a>
#### `level set: curvature: linear solver`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `solver type` | `string` | `CG` | Set this parameter for choosing an iterative linear solver type.<br><br>Allowed values:<br>- `CG`<br>- `GMRES` |
| `preconditioner type` | `string` | `Diagonal` | Set this parameter for choosing a preconditioner type.<br><br>Allowed values:<br>- `Identity`<br>- `AMG`<br>- `ILU`<br>- `Diagonal` |
| `max iterations` | `integer` | `10000` | Set the maximum number of iterations for solving the linear system of equations. |
| `rel tolerance` | `number` | `1e-12` | Set the relative tolerance for a successful solution of the linear system of equations. |
| `abs tolerance` | `number` | `1e-20` | Set the absolute tolerance for a successful solution of the linear system of equations. |
| `do matrix free` | `boolean` | `True` | Set this parameter if a matrix free solution procedure should be performed. |
| `monitor type` | `string` | `none` | Set the monitor type of the linear solver.<br><br>Allowed values:<br>- `none`<br>- `reduced`<br>- `all` |

<a id="level-set-reinitialization"></a>
### `level set: reinitialization`

| Parameter | Type | Default | Description |
|---|---|---|---|
| [`fe`](#level-set-reinitialization-fe) | `object` |  | [See table](#level-set-reinitialization-fe) |
| `enable` | `boolean` | `True` | Set to true to activate reinitialization. |
| `n initial steps` | `integer` | `-1` | Defines the number of initial reinitialization steps of the level set function. In the default case, the number is set equal to the number of max n steps. |
| `pseudo time step size` | `number` | `-1.0` | Sets the reinitialization time step size. By default its computed from the cell size. |
| `pseudo time step factor` | `number` | `1.0` | Factor on the reinitialization time step size that is computed from the cell size. |
| `max n steps` | `integer` | `5` | Sets the maximum number of reinitialization steps |
| `tolerance` | `number` | `2.22507e-308` | Set the tolerance for reinitialization. If the maximum change of the level set field, i.e.  orΔФ or∞, exceeds the tolerance, reinitialization steps will be performed. |
| `tangential diffusion factor` | `number` | `0.0` | Factor that multiplies the normal diffusion factor (diffusion length) to obtain the diffusion factor in the tangential direction. |
| `type` | `string` | `olsson2007` | Sets the type of reinitialization model that should be used.<br><br>Allowed values:<br>- `olsson2007`<br>- `elliptic` |
| `implementation` | `string` | `meltpooldg` | Choose the corresponding implementation of the reinitialization operation.<br><br>Allowed values:<br>- `meltpooldg`<br>- `adaflo` |
| [`Discontinous Galerkin`](#level-set-reinitialization-discontinous-galerkin) | `object` |  | [See table](#level-set-reinitialization-discontinous-galerkin) |
| [`elliptic`](#level-set-reinitialization-elliptic) | `object` |  | [See table](#level-set-reinitialization-elliptic) |
| [`interface thickness parameter`](#level-set-reinitialization-interface-thickness-parameter) | `object` |  | [See table](#level-set-reinitialization-interface-thickness-parameter) |
| [`predictor`](#level-set-reinitialization-predictor) | `object` |  | [See table](#level-set-reinitialization-predictor) |
| [`linear solver`](#level-set-reinitialization-linear-solver) | `object` |  | [See table](#level-set-reinitialization-linear-solver) |

<a id="level-set-reinitialization-fe"></a>
#### `level set: reinitialization: fe`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `not_initialized` | Finite Element.FE_Q: hexahedral continuous finite element with polynomial degree p; FE_SimplexP: tetrahedral continuous finite element with polynomial degree p; FE_Q_iso_Q1: hexahedral continuous finite element with p subdivisions containing linear elements; FE_DGQ: hexahedral discontinuous finite element with polynomial degree p<br><br>Allowed values:<br>- `not_initialized`<br>- `FE_Q`<br>- `FE_SimplexP`<br>- `FE_Q_iso_Q1`<br>- `FE_DGQ` |
| `degree` | `integer` | `-1` | Defines the degree p of the finite element type. If "type" is "FE_Q_iso_Q1" this parameter defines the number of subdivisions. |

<a id="level-set-reinitialization-discontinous-galerkin"></a>
#### `level set: reinitialization: Discontinous Galerkin`

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

<a id="level-set-reinitialization-elliptic"></a>
#### `level set: reinitialization: elliptic`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `penalty parameter` | `number` | `0.0` | Penalty parameter for the enforcement of the initial position of the zero level-set iso-surface during the elliptic reinitialization. |

<a id="level-set-reinitialization-interface-thickness-parameter"></a>
#### `level set: reinitialization: interface thickness parameter`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `proportional_to_cell_size` | Choose the value type of the interface thickness parameter.<br><br>Allowed values:<br>- `proportional_to_cell_size`<br>- `absolute_value`<br>- `number_of_cells_across_interface` |
| `val` | `number` | `0.5` | Defines the value of the chosen interface thickness parameter type |

<a id="level-set-reinitialization-predictor"></a>
#### `level set: reinitialization: predictor`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `none` | Choose a predictor type: none: use old value as initial guess; zero: se zeros as initial guess; linear_extrapolation: calculate the predictor by a linear combination from the two old solution vectors; least_squares_projection: least squares projection (WIP)<br><br>Allowed values:<br>- `none`<br>- `zero`<br>- `linear_extrapolation`<br>- `least_squares_projection` |
| `n old solutions` | `integer` | `2` | Choose the number of old solution vectors considered.This parameter is only relevant for least squares projection.For all other predictors, this parameter will be set appropriately. |

<a id="level-set-reinitialization-linear-solver"></a>
#### `level set: reinitialization: linear solver`

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

<a id="evaporation"></a>
## `🔷 evaporation`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `evaporative mass flux model` | `string` | `analytical` | Choose the formulation how the evaporative mass flux mDot (kg/(m2s)) will be calculated.<br><br>Allowed values:<br>- `analytical`<br>- `recoil_pressure`<br>- `saturated_vapor_pressure`<br>- `hardt_wondra`<br>- `pressure_aware` |
| `interface temperature evaluation type` | `string` | `local_value` | Choose the formulation how the (local) evaporative mass flux will be converted to a DoF vector.will be calculated. When the CutFEM heat transfer operator is used, this input parameter is ignored and the temperature is evaluated at the sharp interface which is equivalent to "sharp".<br><br>Allowed values:<br>- `local_value`<br>- `interface_value` |
| [`analytical`](#evaporation-analytical) | `object` |  | [See table](#evaporation-analytical) |
| [`hardt wondra`](#evaporation-hardt-wondra) | `object` |  | [See table](#evaporation-hardt-wondra) |
| [`pressure aware`](#evaporation-pressure-aware) | `object` |  | [See table](#evaporation-pressure-aware) |
| [`evaporative dilation rate`](#evaporation-evaporative-dilation-rate) | `object` |  | [See table](#evaporation-evaporative-dilation-rate) |
| [`evaporative cooling`](#evaporation-evaporative-cooling) | `object` |  | [See table](#evaporation-evaporative-cooling) |
| [`recoil pressure`](#evaporation-recoil-pressure) | `object` |  | [See table](#evaporation-recoil-pressure) |
| `formulation source term level set` | `string` | `interface_velocity_local` | Select the type how the evaporative mass flux should be considered in the level set equation.<br><br>Allowed values:<br>- `interface_velocity_sharp`<br>- `interface_velocity_sharp_heavy`<br>- `interface_velocity_local`<br>- `rhs` |
| `do level set pressure gradient interpolation` | `boolean` | `False` | Set if the level set gradient for computing the delta function within the evaporative mass flux source terms should be computed based on an interpolation to the pressure space. This is only implemented for evapor_level_set_source_term_type = rhs. |

<a id="evaporation-analytical"></a>
### `evaporation: analytical`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `function` | `string` | `not_initialized` | For evapor evaporation model == analytical, prescribe a spatially constant mass flux due to evaporation (SI unit in kg/m²s), as a function over time t , e.g. min(2.*t,0.01). |

<a id="evaporation-hardt-wondra"></a>
### `evaporation: hardt wondra`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `coefficient` | `number` | `0.0` | Evaporation coefficient for the model by Hardt and Wondra. |

<a id="evaporation-pressure-aware"></a>
### `evaporation: pressure aware`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `Km` | `array` | `[]` | Fitting parameters for the evaporative mass flux function of the pressure-aware model. |
| `ambient_gas_pressure` | `number` | `0.0` | Ambient gas pressure for the pressure-aware model. |

<a id="evaporation-evaporative-dilation-rate"></a>
### `evaporation: evaporative dilation rate`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable` | `boolean` | `False` | Set this parameter to true to consider the evaporative dilation rate in the Navier-Stokes equation. This results in an evaporation-induced jump in the normal velocity component. |
| `model` | `string` | `regularized` | Select how the additional source term due to evaporation in the continuity equation (=evaporative dilation rate) is computed.<br><br>Allowed values:<br>- `regularized`<br>- `sharp` |

<a id="evaporation-evaporative-cooling"></a>
### `evaporation: evaporative cooling`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable` | `boolean` | `False` | Set this parameter to true to consider evaporative cooling in the heat equation |
| `consider enthalpy transport vapor mass flux` | `string` | `default` | Set this parameter to true to account for the enthalpy transported by the vapor mass flux in the heat equation. This is only recommended if the vapor mass flux is not considered in the Navier-Stokes equations.<br><br>Allowed values:<br>- `default`<br>- `true`<br>- `false` |
| `activation temperature` | `number` | `-1e+100` | Activation temperature for the evaporative cooling. It must be smaller than or equal to the boiling temperature. By default, it will be chosen such that the transition from the linear activation ramp is kink-free. |
| `model` | `string` | `regularized` | Select how the additional source term due to evaporation in the heat equation (evaporative cooling) is computed.<br><br>Allowed values:<br>- `none`<br>- `regularized`<br>- `sharp`<br>- `sharp_conforming` |
| [`dirac delta function approximation`](#evaporation-evaporative-cooling-dirac-delta-function-approximation) | `object` |  | [See table](#evaporation-evaporative-cooling-dirac-delta-function-approximation) |

<a id="evaporation-evaporative-cooling-dirac-delta-function-approximation"></a>
#### `evaporation: evaporative cooling: dirac delta function approximation`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `norm_of_indicator_gradient` | Choose how to smear a parameter over the interface.<br><br>Allowed values:<br>- `norm_of_indicator_gradient`<br>- `heaviside_phase_weighted`<br>- `heaviside_times_heaviside_phase_weighted`<br>- `reciprocal_phase_weighted`<br>- `reciprocal_times_heaviside_phase_weighted`<br>- `heavy_phase_only` |
| `auto weights` | `boolean` | `False` | Choose if weights should be computed automatically. |
| `gas phase weight` | `number` | `1.0` | If >>> dirac delta function approximation type <<< is set to any phase weighted optionthis parameter controls the (first) weight of the gas phase (level set = -1). |
| `heavy phase weight` | `number` | `1.0` | If >>> dirac delta function approximation type <<< is set to any phase weighted optionthis parameter controls the (first) weight of the heavy phase (level set = 1). |
| `gas phase weight 2` | `number` | `1.0` | If >>> dirac delta function approximation type <<< is set to >>> heaviside_times_heaviside_phase_weighted <<< this parameter controls the second weight of the gas phase (level set = -1). |
| `heavy phase weight 2` | `number` | `1.0` | If >>> dirac delta function approximation type <<< is set to >>> heaviside_times_heaviside_phase_weighted <<< this parameter controls the second weight of the heavy liquid/solid phase (level set = 1). |

<a id="evaporation-recoil-pressure"></a>
### `evaporation: recoil pressure`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable` | `boolean` | `False` | Set this parameter to true to prescribe the evaporation-induced jump in the pressure field (i.e. recoil pressure), considered as an interfacial force in the momentum balance equation.If 'evaporative dilation rate' is enabled, this pressure jump will be added to the one resulting from the discontinuous normal velocity field. |
| `ambient gas pressure` | `number` | `101300.0` | Ambient gas pressure for the recoil pressure model. |
| `pressure coefficient` | `number` | `0.55` | Pressure coefficient for the recoil pressure model. |
| `temperature constant` | `number` | `-1.0` | Temperature constant for the recoil pressure model. If this parameter is not set, the value is computed by latent_heat_evaporation * molar_mass / universal_gas_constant; |
| `sticking constant` | `number` | `1.0` | Sticking constant. |
| `interface distributed flux type` | `string` | `local_value` | Type that determines how the recoil pressure force is computed in the interfacial zone.<br><br>Allowed values:<br>- `local_value`<br>- `interface_value` |
| `activation temperature` | `number` | `-1e+100` | Activation temperature for the recoil pressure. It must be smaller than or equal to the boiling temperature. As default value, the boiling temperature is chosen. |
| [`dirac delta function approximation`](#evaporation-recoil-pressure-dirac-delta-function-approximation) | `object` |  | [See table](#evaporation-recoil-pressure-dirac-delta-function-approximation) |
| `type` | `string` | `phenomenological` | Choose the model to compute the recoil pressure coefficient: phenomenological or hybrid, in case there is also an evaporation-induced velocity jump.<br><br>Allowed values:<br>- `phenomenological`<br>- `hybrid`<br>- `pressure_aware` |

<a id="evaporation-recoil-pressure-dirac-delta-function-approximation"></a>
#### `evaporation: recoil pressure: dirac delta function approximation`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `norm_of_indicator_gradient` | Choose how to smear a parameter over the interface.<br><br>Allowed values:<br>- `norm_of_indicator_gradient`<br>- `heaviside_phase_weighted`<br>- `heaviside_times_heaviside_phase_weighted`<br>- `reciprocal_phase_weighted`<br>- `reciprocal_times_heaviside_phase_weighted`<br>- `heavy_phase_only` |
| `auto weights` | `boolean` | `False` | Choose if weights should be computed automatically. |
| `gas phase weight` | `number` | `1.0` | If >>> dirac delta function approximation type <<< is set to any phase weighted optionthis parameter controls the (first) weight of the gas phase (level set = -1). |
| `heavy phase weight` | `number` | `1.0` | If >>> dirac delta function approximation type <<< is set to any phase weighted optionthis parameter controls the (first) weight of the heavy phase (level set = 1). |
| `gas phase weight 2` | `number` | `1.0` | If >>> dirac delta function approximation type <<< is set to >>> heaviside_times_heaviside_phase_weighted <<< this parameter controls the second weight of the gas phase (level set = -1). |
| `heavy phase weight 2` | `number` | `1.0` | If >>> dirac delta function approximation type <<< is set to >>> heaviside_times_heaviside_phase_weighted <<< this parameter controls the second weight of the heavy liquid/solid phase (level set = 1). |


---

<a id="material"></a>
## `🔷 material`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `material template` | `string` | `none` | If this parameter is initialized, the material parameters of the specified material will be used as template. Individual properties can be modified. However, be aware to put <material template> in the first place of the <material> section in these cases.<br><br>Allowed values:<br>- `none`<br>- `stainless_steel`<br>- `Ti64`<br>- `Ti64Cunningham` |
| [`gas`](#material-gas) | `object` |  | [See table](#material-gas) |
| [`liquid`](#material-liquid) | `object` |  | [See table](#material-liquid) |
| [`solid`](#material-solid) | `object` |  | [See table](#material-solid) |
| `solidus temperature` | `number` | `0.0` | Solidus temperature (K). |
| `liquidus temperature` | `number` | `0.0` | Liquidus temperature (K). |
| `apparent capacity type` | `string` | `qlq` | Function type for the apparent capacity method to model latent heat during solidification. constant: apparent capacity is constant between the solidus and liquidus temperature; qlq: apparent capacity is given by a quadratic/quadratic function of temperature between the solidus and liquidus temperature (default); poly4_bell: apparent capacity is given by a bell-shaped quartic polynomial function of temperature between the solidus and liquidus temperature.<br><br>Allowed values:<br>- `poly4_bell`<br>- `constant`<br>- `qlq` |
| `latent heat of fusion` | `number` | `0.0` | Latent heat of fusion (J/kg) |
| `boiling temperature` | `number` | `0.0` | Boiling temperature (K). |
| `latent heat of evaporation` | `number` | `0.0` | Latent heat of evaporation (J/kg). |
| `molar mass` | `number` | `0.0` | Molar mass (mol/kg). |
| `specific enthalpy reference temperature` | `number` | `-1e+100` | Reference temperature of the specific enthalpy |
| `two phase fluid properties transition type` | `string` | `smooth` | Choose how to interpolate the properties over the interface. sharp: properties jump at heaviside = 0.5; smooth: properties are smeared between the phases proportional to the heaviside (default); consistent_with_evaporation: same as "smooth", but the density is interpolated proportional by the harmonic mean.<br><br>Allowed values:<br>- `sharp`<br>- `smooth`<br>- `consistent_with_evaporation` |
| `solid liquid properties transition type` | `string` | `mushy_zone` | Choose how to interpolate the properties over between the liquid and the solid phase. mushy_zone: solid and liquid properties are interpolated between the solidus and liquidus temperature (default); sharp: the solid and liquid properties jump at the melting point, which is set via the solidus temperature.<br><br>Allowed values:<br>- `mushy_zone`<br>- `sharp` |

<a id="material-gas"></a>
### `material: gas`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `thermal conductivity` | `number` | `0.0` | thermal conductivity of the gas phase |
| `specific heat capacity` | `number` | `0.0` | specific heat capacity of the gas phase |
| `density` | `number` | `0.0` | density of the gas phase |
| `dynamic viscosity` | `number` | `0.0` | dynamic viscosity of the gas phase |

<a id="material-liquid"></a>
### `material: liquid`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `thermal conductivity` | `number` | `0.0` | thermal conductivity of the liquid phase |
| `specific heat capacity` | `number` | `0.0` | specific heat capacity of the liquid phase |
| `density` | `number` | `0.0` | density of the liquid phase |
| `dynamic viscosity` | `number` | `0.0` | dynamic viscosity of the liquid phase |

<a id="material-solid"></a>
### `material: solid`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `thermal conductivity` | `number` | `0.0` | thermal conductivity of the solid phase |
| `specific heat capacity` | `number` | `0.0` | specific heat capacity of the solid phase |
| `density` | `number` | `0.0` | density of the solid phase |
| `dynamic viscosity` | `number` | `0.0` | dynamic viscosity of the solid phase |


---

<a id="amr"></a>
## `🔷 amr`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `strategy` | `string` | `generic` | Select the AMR strategy.<br><br>Allowed values:<br>- `generic`<br>- `refine_all_interface_cells` |


---
