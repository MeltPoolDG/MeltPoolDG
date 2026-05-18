# mp-compressible-flow: Parameter description

## Contents

- [`base`](#base)
- [`compressible navier stokes`](#compressible-navier-stokes)
- [`output`](#output)
- [`material`](#material)
- [`cut`](#cut)
- [`time stepping`](#time-stepping)
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

<a id="compressible-navier-stokes"></a>
## `🔷 compressible navier stokes`

| Parameter | Type | Default | Description |
|---|---|---|---|
| [`fe`](#compressible-navier-stokes-fe) | `object` |  | [See table](#compressible-navier-stokes-fe) |
| [`time integration`](#compressible-navier-stokes-time-integration) | `object` |  | [See table](#compressible-navier-stokes-time-integration) |
| `linearization jump convective flux` | `string` | `complete_fd` | Calculation method of the linearized jump operator in the convective numerical flux (required for implicit time stepping). The options are "analytic", "complete_fd" and "lambda_fd".<br><br>Allowed values:<br>- `analytic`<br>- `lambda_fd`<br>- `complete_fd` |
| `numerical flux type` | `string` | `lax_friedrichs_modified` | Type of the numerical flux.<br><br>Allowed values:<br>- `lax_friedrichs_modified`<br>- `lax_friedrichs_exact`<br>- `harten_lax_vanleer` |
| `courant number` | `number` | `0.15` | Courant number for convective time-step limit. |
| `viscous courant number` | `number` | `1.0` | Characteristic Courant-like number for viscous time-step limit. |
| `jacobian type` | `string` | `exact` | Type of the jacobian. Choose between 'exact' and 'finite_difference'.<br><br>Allowed values:<br>- `exact`<br>- `finite_difference` |
| `use cfl criteria` | `boolean` | `False` | If set to true, the CFL time step size criteria is used to determine the time step size in each iteration. |
| `gravity constant` | `number` | `0.0` | Gravity constant. |
| `domain representation type` | `string` | `fitted` | Domain representation type. Choose between 'fitted' and 'cut'.<br><br>Allowed values:<br>- `fitted`<br>- `cut` |
| `verbosity level` | `integer` | `-1` | Verbosity level for output. |

<a id="compressible-navier-stokes-fe"></a>
### `compressible navier stokes: fe`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `not_initialized` | Finite Element.FE_Q: hexahedral continuous finite element with polynomial degree p; FE_SimplexP: tetrahedral continuous finite element with polynomial degree p; FE_Q_iso_Q1: hexahedral continuous finite element with p subdivisions containing linear elements; FE_DGQ: hexahedral discontinuous finite element with polynomial degree p<br><br>Allowed values:<br>- `not_initialized`<br>- `FE_Q`<br>- `FE_SimplexP`<br>- `FE_Q_iso_Q1`<br>- `FE_DGQ` |
| `degree` | `integer` | `-1` | Defines the degree p of the finite element type. If "type" is "FE_Q_iso_Q1" this parameter defines the number of subdivisions. |

<a id="compressible-navier-stokes-time-integration"></a>
### `compressible navier stokes: time integration`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `not_initialized` | Name of the time integration scheme.<br><br>Allowed values:<br>- `not_initialized`<br>- `LSRK_stage_1_order_1`<br>- `LSRK_stage_3_order_3`<br>- `LSRK_stage_5_order_4`<br>- `LSRK_stage_7_order_4`<br>- `LSRK_stage_9_order_5`<br>- `implicit_euler`<br>- `explicit_euler`<br>- `crank_nicolson`<br>- `bdf_1`<br>- `bdf_2`<br>- `bdf_3`<br>- `bdf_4`<br>- `bdf_5`<br>- `bdf_6`<br>- `imex` |
| `preconditioner update frequency` | `integer` | `100` | Frequency at which the preconditioner gets updated. |
| [`nlsolve`](#compressible-navier-stokes-time-integration-nlsolve) | `object` |  | [See table](#compressible-navier-stokes-time-integration-nlsolve) |
| [`linear solver`](#compressible-navier-stokes-time-integration-linear-solver) | `object` |  | [See table](#compressible-navier-stokes-time-integration-linear-solver) |

<a id="compressible-navier-stokes-time-integration-nlsolve"></a>
#### `compressible navier stokes: time integration: nlsolve`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max nonlinear iterations` | `integer` | `10` | Set the number of maximum nonlinear iterations with standard tolerances. |
| `field correction tolerance` | `number` | `1e-10` | Set the tolerance for the maximum allowed correction of the unknown field. |
| `residual tolerance` | `number` | `1e-09` | Set the tolerance for the maximum allowed residual of the nonlinear system. |
| `max nonlinear iterations alt` | `integer` | `0` | Set the number of maximum nonlinear iterations with alternative tolerances. |
| `field correction tolerance alt` | `number` | `1e-09` | Set the alternative tolerance for the maximum allowed correction of the unknown field. |
| `residual tolerance alt` | `number` | `1e-08` | Set the alternative tolerance for the maximum allowed residual of the nonlinear system. |
| `verbosity level` | `integer` | `-1` | Set to one for detailed solver output. |

<a id="compressible-navier-stokes-time-integration-linear-solver"></a>
#### `compressible navier stokes: time integration: linear solver`

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
| [`paraview`](#output-paraview) | `object` |  | [See table](#output-paraview) |
| `directory` | `string` | `./` | Sets the base directory for all output. |
| `write frequency` | `integer` | `1` | Every n timestep that should be written |
| `write time step size` | `number` | `1.79769e+308` | Write output output every given time step. If this parameter is set, the output write frequency is deactivated. |
| `output variables` | `array` | `['all']` | Specify variables that you request to output. |
| `do user defined postprocessing` | `boolean` | `False` | Set this parameter to true to enable user defined postprocessing. |
| [`particles`](#output-particles) | `object` |  | [See table](#output-particles) |

<a id="output-paraview"></a>
### `output: paraview`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `compressible output types` | `array` | `['conserved_variables']` | Type of the variables added to the output. |
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

<a id="material"></a>
## `🔷 material`

| Parameter | Type | Default | Description |
|---|---|---|---|
| [`gas`](#material-gas) | `object` |  | [See table](#material-gas) |

<a id="material-gas"></a>
### `material: gas`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `number of species` | `integer` | `1` | Number of different species in the fluid phase. |
| `eos type` | `string` | `ideal_gas` | Type of equation of state. The options are "ideal_gas", "stiffened_gas" and "noble_abel_stiffened_gas".<br><br>Allowed values:<br>- `ideal_gas`<br>- `stiffened_gas`<br>- `noble_abel_stiffened_gas` |
| `reference density` | `number` | `1.0` | Reference density for computing the interior penalty factor. A good first guess is to choose a value in the order of the fluid density. If instabilities occur, the reference density can be decreased, so that the symmetric interior penalization is increased. |
| `reference dynamic viscosity` | `number` | `0.000625` | Reference dynamic viscosity for computing the interior penalty factor. A good first guess is to choose a value in the order of the fluid dynamic viscosity. If instabilities occur, the reference dynamic viscosity can be increased, so that the symmetric interior penalization is increased. |
| [`species 1`](#material-gas-species-1) | `object` |  | [See table](#material-gas-species-1) |
| `name` | `string` | `species` | Name of the species used for output purposes. |
| `specific isobaric heat` | `number` | `1000.0` | Specific isobaric heat. |
| `dynamic viscosity` | `number` | `0.000625` | Dynamic viscosity. |
| `gamma` | `number` | `1.4` | Isentropic exponent, i.e., ratio of specific heat (c_p/c_v). |
| `specific gas constant` | `number` | `287.1` | Specific gas constant. |
| `thermal conductivity` | `number` | `1.79769e+308` | Thermal conductivity. |
| `molar mass` | `number` | `1.79769e+308` | Molar mass. |
| [`equation of state`](#material-gas-equation-of-state) | `object` |  | [See table](#material-gas-equation-of-state) |
| [`species 2`](#material-gas-species-2) | `object` |  | [See table](#material-gas-species-2) |
| [`species interactions`](#material-gas-species-interactions) | `object` |  | [See table](#material-gas-species-interactions) |

<a id="material-gas-species-1"></a>
#### `material: gas: species 1`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `string` | `species` | Name of the species used for output purposes. |
| `specific isobaric heat` | `number` | `1000.0` | Specific isobaric heat. |
| `dynamic viscosity` | `number` | `0.000625` | Dynamic viscosity. |
| `gamma` | `number` | `1.4` | Isentropic exponent, i.e., ratio of specific heat (c_p/c_v). |
| `specific gas constant` | `number` | `287.1` | Specific gas constant. |
| `thermal conductivity` | `number` | `1.79769e+308` | Thermal conductivity. |
| `molar mass` | `number` | `1.79769e+308` | Molar mass. |
| [`equation of state`](#material-gas-species-1-equation-of-state) | `object` |  | [See table](#material-gas-species-1-equation-of-state) |

<a id="material-gas-species-1-equation-of-state"></a>
##### `material: gas: species 1: equation of state`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `p inf` | `number` | `1.79769e+308` | Numerical EOS parameter to model the molecular attraction within condensed matter. The variable is required for the stiffened gas EOS and the Noble-Abel stiffened gas EOS. The minimum value is 0. |
| `b` | `number` | `1.79769e+308` | Numerical EOS parameter to model the covolume of the fluid, i.e., the exclude volume due to the finite size of the molecules. The variable is required for the Noble-Abel stiffened gas EOS. The minimum value is 0. |
| `q` | `N/A` | `2.22507e-308` | Numerical EOS parameter to model the 'heat bound', i.e., the energy due to chemical bounds, hydrogen bondings, latent heat,.... The variable is required for the Noble-Abel stiffened gas EOS. The maximum value is 0. |

<a id="material-gas-equation-of-state"></a>
#### `material: gas: equation of state`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `p inf` | `number` | `1.79769e+308` | Numerical EOS parameter to model the molecular attraction within condensed matter. The variable is required for the stiffened gas EOS and the Noble-Abel stiffened gas EOS. The minimum value is 0. |
| `b` | `number` | `1.79769e+308` | Numerical EOS parameter to model the covolume of the fluid, i.e., the exclude volume due to the finite size of the molecules. The variable is required for the Noble-Abel stiffened gas EOS. The minimum value is 0. |
| `q` | `N/A` | `2.22507e-308` | Numerical EOS parameter to model the 'heat bound', i.e., the energy due to chemical bounds, hydrogen bondings, latent heat,.... The variable is required for the Noble-Abel stiffened gas EOS. The maximum value is 0. |

<a id="material-gas-species-2"></a>
#### `material: gas: species 2`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `string` | `species` | Name of the species used for output purposes. |
| `specific isobaric heat` | `number` | `1000.0` | Specific isobaric heat. |
| `dynamic viscosity` | `number` | `0.000625` | Dynamic viscosity. |
| `gamma` | `number` | `1.4` | Isentropic exponent, i.e., ratio of specific heat (c_p/c_v). |
| `specific gas constant` | `number` | `287.1` | Specific gas constant. |
| `thermal conductivity` | `number` | `1.79769e+308` | Thermal conductivity. |
| `molar mass` | `number` | `1.79769e+308` | Molar mass. |
| [`equation of state`](#material-gas-species-2-equation-of-state) | `object` |  | [See table](#material-gas-species-2-equation-of-state) |

<a id="material-gas-species-2-equation-of-state"></a>
##### `material: gas: species 2: equation of state`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `p inf` | `number` | `1.79769e+308` | Numerical EOS parameter to model the molecular attraction within condensed matter. The variable is required for the stiffened gas EOS and the Noble-Abel stiffened gas EOS. The minimum value is 0. |
| `b` | `number` | `1.79769e+308` | Numerical EOS parameter to model the covolume of the fluid, i.e., the exclude volume due to the finite size of the molecules. The variable is required for the Noble-Abel stiffened gas EOS. The minimum value is 0. |
| `q` | `N/A` | `2.22507e-308` | Numerical EOS parameter to model the 'heat bound', i.e., the energy due to chemical bounds, hydrogen bondings, latent heat,.... The variable is required for the Noble-Abel stiffened gas EOS. The maximum value is 0. |

<a id="material-gas-species-interactions"></a>
#### `material: gas: species interactions`

| Parameter | Type | Default | Description |
|---|---|---|---|
| [`interaction pair 1`](#material-gas-species-interactions-interaction-pair-1) | `object` |  | [See table](#material-gas-species-interactions-interaction-pair-1) |

<a id="material-gas-species-interactions-interaction-pair-1"></a>
##### `material: gas: species interactions: interaction pair 1`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `species 1` | `integer` | `1` | Number of the first species in the interaction pair. |
| `species 2` | `integer` | `2` | Number of the second species in the interaction pair. |
| `diffusion coefficient` | `number` | `0.0` | Diffusion coefficient for the interaction between the two species. |


---

<a id="cut"></a>
## `🔷 cut`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `unfitted flow boundary condition` | `string` | `no_slip_wall` | Flow boundary condition type at the unfitted boundary. Choose between 'no_slip_wall' and 'inflow'. |
| [`stabilization`](#cut-stabilization) | `object` |  | [See table](#cut-stabilization) |

<a id="cut-stabilization"></a>
### `cut: stabilization`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `nitsche parameter` | `number` | `1.0` | Nitsche stabilization parameter. |
| [`ghost-penalty`](#cut-stabilization-ghost-penalty) | `object` |  | [See table](#cut-stabilization-ghost-penalty) |

<a id="cut-stabilization-ghost-penalty"></a>
#### `cut: stabilization: ghost-penalty`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gamma M degree 0` | `number` | `1.0` | Mass matrix ghost-penalty parameter for degree 0. |
| `gamma M degree 1` | `number` | `1.0` | Mass matrix ghost-penalty parameter for degree 1. |
| `gamma M degree 2` | `number` | `1.0` | Mass matrix ghost-penalty parameter for degree 2. |
| `gamma A degree 0` | `number` | `1.0` | Stiffness matrix ghost-penalty parameter for degree 0. |
| `gamma A degree 1` | `number` | `1.0` | Stiffness matrix ghost-penalty parameter for degree 1. |
| `gamma A degree 2` | `number` | `1.0` | Stiffness matrix ghost-penalty parameter for degree 2. |


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

<a id="profiling"></a>
## `🔷 profiling`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable` | `boolean` | `False` | Set this parameter to true if profiling should be enabled. It will be automaticallyenabled for verbosity level >=1. |
| `write time step size` | `number` | `10.0` | Write profiling output every given time step size. If this parameter is set, the specified parameter for write frequency is overwritten. |
| `time type` | `string` | `real` | Choose the type of time measure to write profiling information.<br><br>Allowed values:<br>- `real`<br>- `simulation` |


---
