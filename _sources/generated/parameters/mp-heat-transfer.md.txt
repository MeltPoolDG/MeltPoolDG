# mp-heat-transfer: Parameter description

## Contents

- [`base`](#base)
- [`time stepping`](#time-stepping)
- [`adaptive meshing`](#adaptive-meshing)
- [`heat`](#heat)
- [`material`](#material)
- [`laser`](#laser)
- [`rte`](#rte)
- [`evaporation`](#evaporation)
- [`output`](#output)
- [`profiling`](#profiling)
- [`application specific`](#application-specific)

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

<a id="heat"></a>
## `🔷 heat`

| Parameter | Type | Default | Description |
|---|---|---|---|
| [`fe`](#heat-fe) | `object` |  | [See table](#heat-fe) |
| `operator type` | `string` | `diffuse` | Choose the heat operator implementation. Options: diffuse, cut<br><br>Allowed values:<br>- `diffuse`<br>- `cut` |
| [`cut`](#heat-cut) | `object` |  | [See table](#heat-cut) |
| `enable time dependent bc` | `boolean` | `False` | Set this parameter to true to enable time-dependent bc. |
| [`diffuse`](#heat-diffuse) | `object` |  | [See table](#heat-diffuse) |
| [`radiative boundary condition`](#heat-radiative-boundary-condition) | `object` |  | [See table](#heat-radiative-boundary-condition) |
| [`convective boundary condition`](#heat-convective-boundary-condition) | `object` |  | [See table](#heat-convective-boundary-condition) |
| `verbosity level` | `integer` | `-1` | Sets the maximum verbosity level of the console output. |
| [`nlsolve`](#heat-nlsolve) | `object` |  | [See table](#heat-nlsolve) |
| [`linear solver`](#heat-linear-solver) | `object` |  | [See table](#heat-linear-solver) |
| [`predictor`](#heat-predictor) | `object` |  | [See table](#heat-predictor) |

<a id="heat-fe"></a>
### `heat: fe`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `not_initialized` | Finite Element.FE_Q: hexahedral continuous finite element with polynomial degree p; FE_SimplexP: tetrahedral continuous finite element with polynomial degree p; FE_Q_iso_Q1: hexahedral continuous finite element with p subdivisions containing linear elements; FE_DGQ: hexahedral discontinuous finite element with polynomial degree p<br><br>Allowed values:<br>- `not_initialized`<br>- `FE_Q`<br>- `FE_SimplexP`<br>- `FE_Q_iso_Q1`<br>- `FE_DGQ` |
| `degree` | `integer` | `-1` | Defines the degree p of the finite element type. If "type" is "FE_Q_iso_Q1" this parameter defines the number of subdivisions. |

<a id="heat-cut"></a>
### `heat: cut`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `two phase` | `boolean` | `True` | Set this parameter to "false" to ignore the gas phase. |
| `theta` | `number` | `0.5` | Parameter for one step theta time integration. |
| `do explicit symmetry term` | `boolean` | `True` | Set this parameter to true to consider the explicit symmetry term. Note: this parameter only applies if the setup is two-phase. |
| [`stabilization`](#heat-cut-stabilization) | `object` |  | [See table](#heat-cut-stabilization) |

<a id="heat-cut-stabilization"></a>
#### `heat: cut: stabilization`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `nitsche parameter` | `number` | `1.0` | Nitsche stabilization parameter. |
| [`ghost-penalty`](#heat-cut-stabilization-ghost-penalty) | `object` |  | [See table](#heat-cut-stabilization-ghost-penalty) |

<a id="heat-cut-stabilization-ghost-penalty"></a>
##### `heat: cut: stabilization: ghost-penalty`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gamma M degree 0` | `number` | `1.0` | Mass matrix ghost-penalty parameter for degree 0. |
| `gamma M degree 1` | `number` | `1.0` | Mass matrix ghost-penalty parameter for degree 1. |
| `gamma M degree 2` | `number` | `1.0` | Mass matrix ghost-penalty parameter for degree 2. |
| `gamma A degree 0` | `number` | `1.0` | Stiffness matrix ghost-penalty parameter for degree 0. |
| `gamma A degree 1` | `number` | `1.0` | Stiffness matrix ghost-penalty parameter for degree 1. |
| `gamma A degree 2` | `number` | `1.0` | Stiffness matrix ghost-penalty parameter for degree 2. |

<a id="heat-diffuse"></a>
### `heat: diffuse`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `use volume-specific thermal capacity for phase interpolation` | `boolean` | `False` | Perform phase interpolation via the volumetric thermal capacity (product of density  and capacity) instead of interpolating density and thermal capacity individually. |

<a id="heat-radiative-boundary-condition"></a>
### `heat: radiative boundary condition`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `emissivity` | `number` | `0.0` | Emissivity. |
| `temperature infinity` | `number` | `0.0` | Infinity temperature. |

<a id="heat-convective-boundary-condition"></a>
### `heat: convective boundary condition`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `convection coefficient` | `number` | `0.0` | Convection coefficient. |
| `temperature infinity` | `number` | `0.0` | Infinity temperature. |

<a id="heat-nlsolve"></a>
### `heat: nlsolve`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max nonlinear iterations` | `integer` | `10` | Set the number of maximum nonlinear iterations with standard tolerances. |
| `field correction tolerance` | `number` | `1e-10` | Set the tolerance for the maximum allowed correction of the unknown field. |
| `residual tolerance` | `number` | `1e-09` | Set the tolerance for the maximum allowed residual of the nonlinear system. |
| `max nonlinear iterations alt` | `integer` | `0` | Set the number of maximum nonlinear iterations with alternative tolerances. |
| `field correction tolerance alt` | `number` | `1e-09` | Set the alternative tolerance for the maximum allowed correction of the unknown field. |
| `residual tolerance alt` | `number` | `1e-08` | Set the alternative tolerance for the maximum allowed residual of the nonlinear system. |
| `verbosity level` | `integer` | `-1` | Set to one for detailed solver output. |

<a id="heat-linear-solver"></a>
### `heat: linear solver`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `solver type` | `string` | `GMRES` | Set this parameter for choosing an iterative linear solver type.<br><br>Allowed values:<br>- `CG`<br>- `GMRES` |
| `preconditioner type` | `string` | `Diagonal` | Set this parameter for choosing a preconditioner type.<br><br>Allowed values:<br>- `Identity`<br>- `AMG`<br>- `ILU`<br>- `Diagonal` |
| `max iterations` | `integer` | `10000` | Set the maximum number of iterations for solving the linear system of equations. |
| `rel tolerance` | `number` | `1e-12` | Set the relative tolerance for a successful solution of the linear system of equations. |
| `abs tolerance` | `number` | `1e-20` | Set the absolute tolerance for a successful solution of the linear system of equations. |
| `do matrix free` | `boolean` | `True` | Set this parameter if a matrix free solution procedure should be performed. |
| `monitor type` | `string` | `none` | Set the monitor type of the linear solver.<br><br>Allowed values:<br>- `none`<br>- `reduced`<br>- `all` |

<a id="heat-predictor"></a>
### `heat: predictor`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `linear_extrapolation` | Choose a predictor type: none: use old value as initial guess; zero: se zeros as initial guess; linear_extrapolation: calculate the predictor by a linear combination from the two old solution vectors; least_squares_projection: least squares projection (WIP)<br><br>Allowed values:<br>- `none`<br>- `zero`<br>- `linear_extrapolation`<br>- `least_squares_projection` |
| `n old solutions` | `integer` | `2` | Choose the number of old solution vectors considered.This parameter is only relevant for least squares projection.For all other predictors, this parameter will be set appropriately. |


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

<a id="laser"></a>
## `🔷 laser`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `string` | `not_initialized` | Laser model. analytical_temperature: see Mirkoohi et al. (2019); volumetric: volumetric heat source, the intensity is defined by ""intensity profile""; interface_projection: projection-based regularized continuum surface flux in "direction", the intensity is defined by ""intensity profile""; interface_projection_sharp: projection-based sharp surface flux in "direction", the intensity is defined by ""intensity profile""; interface_projection_sharp_conforming: projection-based sharp surface flux in "direction" on a conforming mesh, the intensity is defined by ""intensity profile""; RTE: continuum surface flux projected using the radiative transport equation in "direction", supporting shadowing of undercuts, the intensity is defined by ""intensity profile"";<br><br>Allowed values:<br>- `not_initialized`<br>- `analytical_temperature`<br>- `volumetric`<br>- `interface_projection_regularized`<br>- `interface_projection_sharp`<br>- `interface_projection_sharp_conforming`<br>- `RTE` |
| `intensity profile` | `string` | `Gauss` | Laser intensity profile. uniform: note that the "power" input is treated as the uniform power density in the whole domain; Gauss: Gaussian laser intensity shape with "radius" that retains the "power"; Gusarov: see Gusarov et al. (2009);<br><br>Allowed values:<br>- `uniform`<br>- `Gauss`<br>- `Gusarov` |
| `power` | `number` | `0.0` | Laser power |
| `power over time` | `string` | `constant` | Temporal distribution of the laser power<br><br>Allowed values:<br>- `constant`<br>- `ramp` |
| `power start time` | `number` | `0.0` | In case of time-dependent laser power: activation time of |
| `power end time` | `number` | `1.79769e+308` | In case of time-dependent laser power: end time of |
| `absorptivity gas` | `number` | `1.0` | Laser energy absorptivity of the gaseous part of the domain. |
| `absorptivity liquid` | `number` | `1.0` | Laser energy absorptivity of the liquid part of the domain. |
| `starting position` | `array` | `[]` | Center coordinates of the laser beam starting position on the interface melt/gas. |
| `scan speed` | `number` | `0.0` | Scan speed of the laser |
| `scan direction` | `array` | `[]` | Direction of laser motion as a vector |
| `beam direction` | `array` | `[]` | Laser beam direction. |
| `beam rotation axis` | `array` | `[]` | Axis around which the initial laser beam direction will be rotated. Relevant only in 3D. |
| `beam rotation angle` | `number` | `0.0` | Rotation angle applied to the laser beam direction (in 3D about 'beam rotation axis' following the right-hand rule; in 2D: as defined by the 2D rotation matrix |
| `radius` | `number` | `0.0` | Laser beam radius. |
| [`gusarov`](#laser-gusarov) | `object` |  | [See table](#laser-gusarov) |
| [`analytical`](#laser-analytical) | `object` |  | [See table](#laser-analytical) |
| [`dirac delta function approximation`](#laser-dirac-delta-function-approximation) | `object` |  | [See table](#laser-dirac-delta-function-approximation) |

<a id="laser-gusarov"></a>
### `laser: gusarov`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `reflectivity` | `number` | `0.0` | Reflectivity of the material. |
| `extinction coefficient` | `number` | `0.0` | Extinction coefficient in [1/m]. |
| `layer thickness` | `number` | `0.0` | Layer thickness |

<a id="laser-analytical"></a>
### `laser: analytical`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ambient temperature` | `number` | `0.0` | Ambient temperature in the inert gas. |
| `max temperature` | `number` | `0.0` | Maximum temperature arising in the melt pool. If this temperature is lower than the boiling temperature, this value is corrected to correspond to the boiling temperature + 500 K. |
| `temperature x to y ratio` | `number` | `1.0` | This factor scales the analytical temperature field to be anisotropic. |

<a id="laser-dirac-delta-function-approximation"></a>
### `laser: dirac delta function approximation`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `norm_of_indicator_gradient` | Choose how to smear a parameter over the interface.<br><br>Allowed values:<br>- `norm_of_indicator_gradient`<br>- `heaviside_phase_weighted`<br>- `heaviside_times_heaviside_phase_weighted`<br>- `reciprocal_phase_weighted`<br>- `reciprocal_times_heaviside_phase_weighted`<br>- `heavy_phase_only` |
| `auto weights` | `boolean` | `False` | Choose if weights should be computed automatically. |
| `gas phase weight` | `number` | `1.0` | If >>> dirac delta function approximation type <<< is set to any phase weighted optionthis parameter controls the (first) weight of the gas phase (level set = -1). |
| `heavy phase weight` | `number` | `1.0` | If >>> dirac delta function approximation type <<< is set to any phase weighted optionthis parameter controls the (first) weight of the heavy phase (level set = 1). |
| `gas phase weight 2` | `number` | `1.0` | If >>> dirac delta function approximation type <<< is set to >>> heaviside_times_heaviside_phase_weighted <<< this parameter controls the second weight of the gas phase (level set = -1). |
| `heavy phase weight 2` | `number` | `1.0` | If >>> dirac delta function approximation type <<< is set to >>> heaviside_times_heaviside_phase_weighted <<< this parameter controls the second weight of the heavy liquid/solid phase (level set = 1). |


---

<a id="rte"></a>
## `🔷 rte`

| Parameter | Type | Default | Description |
|---|---|---|---|
| [`fe`](#rte-fe) | `object` |  | [See table](#rte-fe) |
| `rte verbosity level` | `integer` | `-1` | Sets the maximum verbosity level of the console output. The maximum level with respect to the  base value is decisive. |
| `predictor type` | `string` | `none` | Choose a predictor type.<br><br>Allowed values:<br>- `none`<br>- `pseudo_time_stepping` |
| `absorptivity type` | `string` | `gradient_based` | Chooses the formulation of the absorptivity coefficient<br><br>Allowed values:<br>- `constant`<br>- `gradient_based` |
| `avoid singular matrix absorptivity` | `number` | `1e-16` | Minimum value for absorptivity to ensure a non-singular matrix for RTE. |
| [`linear solver`](#rte-linear-solver) | `object` |  | [See table](#rte-linear-solver) |
| [`pseudo time stepping`](#rte-pseudo-time-stepping) | `object` |  | [See table](#rte-pseudo-time-stepping) |
| [`absorptivity`](#rte-absorptivity) | `object` |  | [See table](#rte-absorptivity) |

<a id="rte-fe"></a>
### `rte: fe`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `not_initialized` | Finite Element.FE_Q: hexahedral continuous finite element with polynomial degree p; FE_SimplexP: tetrahedral continuous finite element with polynomial degree p; FE_Q_iso_Q1: hexahedral continuous finite element with p subdivisions containing linear elements; FE_DGQ: hexahedral discontinuous finite element with polynomial degree p<br><br>Allowed values:<br>- `not_initialized`<br>- `FE_Q`<br>- `FE_SimplexP`<br>- `FE_Q_iso_Q1`<br>- `FE_DGQ` |
| `degree` | `integer` | `-1` | Defines the degree p of the finite element type. If "type" is "FE_Q_iso_Q1" this parameter defines the number of subdivisions. |

<a id="rte-linear-solver"></a>
### `rte: linear solver`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `solver type` | `string` | `GMRES` | Set this parameter for choosing an iterative linear solver type.<br><br>Allowed values:<br>- `CG`<br>- `GMRES` |
| `preconditioner type` | `string` | `ILU` | Set this parameter for choosing a preconditioner type.<br><br>Allowed values:<br>- `Identity`<br>- `AMG`<br>- `ILU`<br>- `Diagonal` |
| `max iterations` | `integer` | `10000` | Set the maximum number of iterations for solving the linear system of equations. |
| `rel tolerance` | `number` | `1e-12` | Set the relative tolerance for a successful solution of the linear system of equations. |
| `abs tolerance` | `number` | `1e-20` | Set the absolute tolerance for a successful solution of the linear system of equations. |
| `do matrix free` | `boolean` | `True` | Set this parameter if a matrix free solution procedure should be performed. |
| `monitor type` | `string` | `none` | Set the monitor type of the linear solver.<br><br>Allowed values:<br>- `none`<br>- `reduced`<br>- `all` |

<a id="rte-pseudo-time-stepping"></a>
### `rte: pseudo time stepping`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `diffusion term scaling` | `number` | `1.0` | Scaling parameter of diffusion term. |
| `advection term scaling` | `number` | `1.0` | Scaling parameter of advection term. |
| `pseudo time scaling` | `number` | `0.01` | Determine the pseudo-time step as the product of this scaling and minimum cell size. |
| `rel tolerance` | `number` | `0.001` | Pseudo-time stepping relative tolerance. |
| [`time stepping`](#rte-pseudo-time-stepping-time-stepping) | `object` |  | [See table](#rte-pseudo-time-stepping-time-stepping) |
| [`linear solver`](#rte-pseudo-time-stepping-linear-solver) | `object` |  | [See table](#rte-pseudo-time-stepping-linear-solver) |

<a id="rte-pseudo-time-stepping-time-stepping"></a>
#### `rte: pseudo time stepping: time stepping`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `start time` | `number` | `0.0` | Defines the start time for the solution of the levelset problem |
| `end time` | `number` | `1.79769e+308` | Sets the end time for the solution of the levelset problem |
| `time step size` | `number` | `0.0` | Sets the step size for time stepping. For non-uniform time stepping, this parameter determines the size of the first time step. |
| `max n steps` | `integer` | `1` | Sets the maximum number of melt_pool steps |
| `time step size function` | `string` | `0.0*t` | Set an analytical function to determine the time step size. For the prediction of the new time increment, the old time is used. |

<a id="rte-pseudo-time-stepping-linear-solver"></a>
#### `rte: pseudo time stepping: linear solver`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `solver type` | `string` | `CG` | Set this parameter for choosing an iterative linear solver type.<br><br>Allowed values:<br>- `CG`<br>- `GMRES` |
| `preconditioner type` | `string` | `ILU` | Set this parameter for choosing a preconditioner type.<br><br>Allowed values:<br>- `Identity`<br>- `AMG`<br>- `ILU`<br>- `Diagonal` |
| `max iterations` | `integer` | `10000` | Set the maximum number of iterations for solving the linear system of equations. |
| `rel tolerance` | `number` | `1e-12` | Set the relative tolerance for a successful solution of the linear system of equations. |
| `abs tolerance` | `number` | `1e-20` | Set the absolute tolerance for a successful solution of the linear system of equations. |
| `do matrix free` | `boolean` | `True` | Set this parameter if a matrix free solution procedure should be performed. |
| `monitor type` | `string` | `none` | Set the monitor type of the linear solver.<br><br>Allowed values:<br>- `none`<br>- `reduced`<br>- `all` |

<a id="rte-absorptivity"></a>
### `rte: absorptivity`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `absorptivity gas` | `number` | `0.1` | Sets the absorptivity of the gas phase. |
| `absorptivity liquid` | `number` | `0.9` | Sets the absorptivity of the liquid phase. |
| `avoid div zero constant` | `number` | `1e-16` | Sets the absorptivity of the gas phase. |


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

<a id="application-specific"></a>
## `🔷 application specific`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `do solidification` | `boolean` | `False` | Set this parameter to true if you want to consider melting/solidification effects. |
| `amr strategy` | `string` | `KellyErrorEstimator` | Select the AMR strategy.<br><br>Allowed values:<br>- `KellyErrorEstimator`<br>- `generic` |


---
