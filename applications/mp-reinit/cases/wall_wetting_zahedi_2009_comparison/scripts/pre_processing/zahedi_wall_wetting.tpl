{
  "base": {
    "case name": "wall_wetting",
    "dimension": "2",
    "global refinements": "4",
    "verbosity level": "3",
    "fe": {
      "degree": "1"
    }
  },
  "time stepping": {
    "start time": "0.0",
    "end time": "1.0"
  },
  "output": {
    "directory": "../../meltpooldg_results/CASE_NAME",
    "do user defined postprocessing": "true",
    "paraview": {
      "filename": "wall_wetting",
      "enable": "true",
      "n groups": "2"
    }
  },
  "reinitialization": {
    "linear solver": {
      "do matrix free": "true",
      "preconditioner type": "ILU",
      "abs tolerance": "1e-10"
    },
    "type": "olsson2007",
    "hyperbolic": {
      "pseudo time stepping": {
        "tolerance": "1e-5"
      },
      "Continuous Galerkin": {
        "tangential diffusion factor": "EPSILON_T_FACTOR"
      }
    },
    "interface thickness parameter": {
      "type": "proportional_to_cell_size",
      "val": "EPSILON_N_FACTOR"
    }
  },
  "normal vector": {
    "compute normalized vector": "true",
    "linear solver": {
      "do matrix free": "true",
      "preconditioner type": "ILU"
    }
  },
  "case specific": {
    "contact angle": "STATIC_CONTACT_ANGLE",
    "gamma factor": "GAMMA_FACTOR"
  }
}
