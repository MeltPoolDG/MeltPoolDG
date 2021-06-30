# MeltPoolDG

This project is work in progress. We will publish the code soon in this repository. In the meantime impressions will be shared on our [YouTube channel](https://www.youtube.com/channel/UCNZqFRMi-bBMaDMXC38AehA).

## (DG)-FEM-based multi-phase flow solvers for high-fidelity metal additive manufacturing process simulations

The aim of `MeltPoolDG` is to provide application-oriented, research-driven solvers for modeling the thermo-hydrodynamics in the vicinity of the melt pool during selective laser melting, including melt pool formation and the interaction of the multi-phase flow system (liquid metal/ambient gas/metal vapor). They are and will be based on continuous and discontinuous finite element methods in an Eulerian setting. For interface modeling of the multi-phase flow problem including phase-transition (evporation/melting), diffuse interface capturing schemes such as level set methods and phase-field methods are used. It strongly builds upon the general purpose finite element library [deal.II](https://github.com/dealii/dealii) and the highly efficient, matrix-free two-phase flow solver [adaflo](https://github.com/kronbichler/adaflo). Furthermore, via `deal.II` we access also other important third-party libraries such as `p4est`, `Trilinos` and `Metis`.

`MeltPoolDG`is freely available under the LGPLV2 license. Please find the details in the [LICENSE](https://github.com/MeltPoolDG/MeltPoolDG/blob/master/LICENSE) file.

![alt text](doc/MeltPoolDG.png?raw=true)

## Authors

The principal developers are (in chronological order of entering the project):
* [Magdalena Schreter](https://www.uibk.ac.at/bft/mitarbeiter/schreter.html) [@mschreter](https://github.com/mschreter), University of Innsbruck (AT)/Technical University of Munich (DE)
* [Peter Munch](https://www.mw.tum.de/lnm/staff/peter-munch) [@peterrum](https://github.com/peterrum), Technical University of Munich/Helmholtz Zentrum hereon (DE)
* [Nils Much](https://www.mw.tum.de/lnm/staff/nils-much/) [@nmuch](https://github.com/nmuch), Technical University of Munich (DE)

We gratefully acknowledge the contributions and discussions with Christoph Meier (Technical University of Munich) and Martin Kronbichler (Technical University of Munich/Uppsala University).

## Documentation

The installation procedure is described in the [WIKI](https://github.com/MeltPoolDG/MeltPoolDG/wiki). The documentation can be found under https://meltpooldg.github.io/MeltPoolDG/.
