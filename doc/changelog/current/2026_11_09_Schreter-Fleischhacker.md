## [Main] - 2026/11/09

### Fixed

- MINOR Fixed the mask that scales the surface tension force if `flow/surface tension/zero surface tension in solid` is enabled. Since the solid field is the solid fraction weighted by the heaviside representation of the level set, the previous mask `1 - f_s H` evaluated to about `1 - 0.5 f_s` at the interface, such that roughly half of the surface tension force still acted on a fully solid surface. The mask now recovers the liquid fraction `1 - f_s` across the diffuse interface, so surface tension vanishes on solid surfaces.
