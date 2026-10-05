## [Main] - 2026/10/02

### Added

- MAJOR Added fixed point elliptic level-set reinitialisation procedure for a CG- or DG-FEM-based discrete level-set field by solving an elliptic problem.
It is based on the following publication: Adams, T., Giani, S., & Coombs, W. M. (2019). A high-order elliptic PDE based level set reinitialisation method using a discontinuous Galerkin discretisation. Journal of Computational Physics, 379, 373-391.

- MAJOR For CG-FEM, added an analytical Newton-Raphson formulation (derived from the weak form given in the literature).

- MAJOR Integrated elliptic reinitialisation into the `mp-reinit` application.

- MAJOR Integrated elliptic reinitialisation into `mp-level-set` application, adjusted the CG and DG level set operation files accordingly.
