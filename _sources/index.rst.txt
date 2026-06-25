Welcome to
=====================================

.. image:: ../../logo/logo_text_text.png
  :alt: MeltPoolDG
  :align: center

\

.. raw:: html

   <div class="youtube-slideshow">
   <iframe class="responsive-iframe" src="https://www.youtube.com/embed/9Te9ir4y8e0" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   <iframe class="responsive-iframe" src="https://www.youtube.com/embed/KBPS9vFwqJE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

\

MeltPoolDG is a high-fidelity simulation software for predicting the coupled powder--melt--gas dynamics occurring during metal additive manufacturing such as laser powder bed fusion (LPBF) or binder jetting. It is designed to resolve the complex multi-phase thermo-fluid-structure-contact-interaction phenomena in and around the melt pool, including melting, solidification, evaporation, recoil pressure, capillary forces, interface dynamics, and the interaction between liquid metal, ambient gas, metal vapor, and powder particles.

Rather than providing a general-purpose multiphysics framework, MeltPoolDG focuses on delivering state-of-the-art numerical methods tailored to the extreme physical conditions encountered in metal additive manufacturing. The software combines advanced finite element discretizations with modern immersed-interface techniques, matrix-free operator evaluation, and SIMD-vectorized algorithms to enable highly accurate, scalable, and computationally efficient simulations.

Depending on the problem under consideration, the software supports Eulerian formulations for multiphase thermo-fluid dynamics, Lagrangian formulations for particle transport, and mixed Eulerian--Lagrangian formulations for coupled thermo-fluid--structure--contact phenomena. These formulations are discretized using continuous and discontinuous Galerkin finite element methods as well as discrete element methods.

MeltPoolDG leverages the mature scientific computing ecosystem provided by the general-purpose finite element library `deal.II <https://dealii.org/>`_ and its third-party libraries, including p4est, Trilinos, and METIS. In addition, we build open the highly-efficient matrix-free incompressible flow solver `adaflo <https://github.com/kronbichler/adaflo>`_.

Originally developed as part of an FWF Schrödinger Fellowship research project on high-fidelity melt pool simulations with resolved vapor flow, MeltPoolDG has evolved into a comprehensive software suite for simulating coupled powder--melt--gas dynamics in metal additive manufacturing.

MeltPoolDG is developed under the umbrella of the open-source multiphysics simulation framework `4C <https://github.com/4C-multiphysics/4C>`_, which provides a comprehensive environment for coupled multiphysics simulations.

Content
-------
.. toctree::
   :maxdepth: 2

   installation
   generated/parameters/index
   developer_guide
   publications

MeltPoolDG is an open-source project maintained by a team of principal developers, mainly from the Professorship of Simulation for Additive Manufacturing (`SAM <https://www.epc.ed.tum.de/sam/home/>`_) at the Technical University of Munich (TUM).

Principal Developers
-------
- `Magdalena Schreter-Fleischhacker <https://www.epc.ed.tum.de/sam/team/magdalena-schreter-fleischhacker/>`_
- `Julian Brotz <https://www.epc.ed.tum.de/sam/team/julian-brotz/>`_
- `Andreas Koch <https://www.epc.ed.tum.de/sam/team/andreas-koch/>`_

Former Principal Developers
-------
- `Nils Much <https://www.epc.ed.tum.de/sam/team/nils-much/>`_
- Peter Munch

Contributors
-------
MeltPoolDG has benefited from contributions, feedback, and inspiration from a number of researchers, developers and students. Open-source software thrives through collaboration and the exchange of ideas, and we are grateful to everyone who has contributed code, reported issues, suggested improvements, or inspired new developments.

The following people (in alphabetical order) have contributed directly to the library. We would also like to acknowledge the many individuals whose discussions, feedback, and research have influenced the project. We hope the MeltPoolDG community will continue to grow, and we warmly welcome future contributors.

- Amishga Alphonius
- Bruno Blais
- Anna Frolova
- Martin Kronbichler
- Hélène Papillon Laroche
- Judith Pauen
- Aleksandra Petrovskaia
- Johannes Resch
- Torsten Schmid
- Sandra Seibold
- Lysander Sirach
- Tinh Vo
- Simon Völkl
- Jan Westermann

**Note:** MeltPoolDG builds upon the outstanding work of the `deal.II development team <https://dealii.org/community/team/>`_. The authors of MeltPoolDG gratefully acknowledge that specialized simulation software such as MeltPoolDG would not be possible without the robust foundations provided by deal.II.

Related Open-Source Projects
-------
- `deal.II <https://github.com/dealii/dealii>`_
- `4C <https://github.com/4C-multiphysics/4C>`_
- `lethe <https://github.com/chaos-polymtl/lethe>`_
- `hpsint <https://github.com/hpsint/hpsint>`_

Funding Sources
-------

Development of this project has been partially supported by:

- European Research Council (ERC), Starting Grant ExcelAM (Grant No. 101117579) (2024-2029)
- Austrian Science Fund (FWF), Schrödinger Fellowship J4577 (2022-2024)
- Deutsche Forschungsgemeinschaft (DFG, German Research Foundation), Grant No. 437616465 (XXXX-XXXX)
