#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: magdalena
"""

##############################################################################
#          USER INPUT
##############################################################################

import sys
import collections.abc
import click
import argparse
import os
from operator import getitem
from functools import reduce
import collections
import json

old_parameter_names = [
    ["laser", "laser center"],
    ["heat", "interpolate rho times cp"],
    ["rte", "problem type"],
    ["rte", "pseudo time stepping", "max n steps"],
    ["rte", "pseudo time stepping", "time step size"],
    ["rte", "verbosity level"],
    ["laser", "laser heat source model"],
    ["laser", "laser impact type"],
    ["laser", "laser power"],
    ["laser", "laser power over time"],
    ["laser", "laser power start time"],
    ["laser", "laser power end time"],
    # and ["laser", "analytical", "absorptivity gas"],
    ["laser", "laser gauss absorptivity gas"],
    # and ["laser", "analytical", "absorptivity liquid"],
    ["laser", "laser gauss absorptivity liquid"],
    ["laser", "laser do move"],
    ["laser", "laser scan speed"],
    # and ["laser", "laser gusarov laser beam radius"],
    ["laser", "laser gauss laser beam radius"],
    ["laser", "laser gusarov reflectivity"],
    ["laser", "laser gusarov extinction coefficient"],
    ["laser", "laser gusarov layer thickness"],
    ["heat", "heat convection coefficient"],
    ["heat", "heat emissivity"],
    ["heat", "heat temperature infinity"],
    ["heat", "heat nlsolve max nonlinear iterations"],
    ["heat", "heat nlsolve field correction tolerance"],
    ["heat", "heat nlsolve residual tolerance"],
    ["heat", "heat nlsolve max nonlinear iterations alt"],
    ["heat", "heat nlsolve field correction tolerance alt"],
    ["heat", "heat nlsolve residual tolerance alt"],
    ["paraview", "paraview directory"],
    ["paraview", "paraview write frequency"],
    ["paraview", "paraview write time step size"],
    ["paraview", "output variables"],
    ["paraview", "do user defined postprocessing"],
    ["paraview", "paraview do output"],
    ["paraview", "paraview filename"],
    ["paraview", "paraview n digits timestep"],
    ["paraview", "paraview print boundary id"],
    ["paraview", "paraview output subdomains"],
    ["paraview", "output material id"],
    ["paraview", "write higher order cells"],
    ["paraview", "paraview n groups"],
    ["paraview", "paraview n patches"],
    ["recoil pressure", "recoil pressure constant"],
    ["recoil pressure", "recoil temperature constant"],
    ["material", "material first conductivity"],
    ["material", "material first capacity"],
    ["material", "material first density"],
    ["material", "material first viscosity"],
    ["material", "material second conductivity"],
    ["material", "material second capacity"],
    ["material", "material second density"],
    ["material", "material second viscosity"],
    ["material", "material solid conductivity"],
    ["material", "material solid capacity"],
    ["material", "material solid density"],
    ["material", "material solid viscosity"],
    ["material", "material solidus temperature"],
    ["material", "material liquidus temperature"],
    ["material", "material boiling temperature"],
    ["material", "material latent heat of evaporation"],
    ["material", "material molar mass"],
    ["material", "material sticking constant"],
    ["material", "material specific enthalpy reference temperature"],
    ["material", "material two phase properties transition type"],
    ["material", "material solidification type"],
    ["heat", "emissivity"],
    ["heat", "convection coefficient"],
    ["heat", "temperature infinity"],
    # 24-02-29
    ["evaporation", "evapor evaporative mass flux"],
    ["evaporation", "evapor formulation evaporative mass flux over interface"],
    ["evaporation", "evapor evaporation model"],
    ["evaporation", "evapor coefficient"],
    ["evaporation", "evapor line integral n subdivisions per side"],
    ["evaporation", "evapor line integral n subdivisions MCA"],
    ["evaporation", "evapor level set source term type"],
    ["evaporation", "evapor formulation source term heat"],
    ["evaporation", "evapor formulation source term continuity"],
    ["evaporation", "evapor do level set pressure gradient interpolation"],
    ["recoil pressure"],
    ["problem specific", "do evaporative heat flux"],
    ["problem specific", "do recoil pressure"],
    ["problem specific", "do evaporative velocity jump"],
    ["heat", "dirac delta function approximation"],
    ["evaporation", "recoil pressure", "model type"],
    # 24-03-06
    ["levelset"],
    ["reinitialization"],
    ["curvature"],
    ["normal vector"],
    ["advection diffusion"],
    ["level set", "advection diffusion", "advec diff diffusivity"],
    ["level set", "advection diffusion", "advec diff time integration scheme"],
    ["level set", "advection diffusion", "advec diff implementation"],
    ["level set", "normal vector", "normal vec damping scale factor"],
    ["level set", "normal vector", "normal vec implementation"],
    ["level set", "normal vector", "normal vec verbosity level"],
    ["level set", "normal vector", "normal vec do narrow band"],
    ["level set", "normal vector", "narrow band threshold"],
    ["level set", "curvature", "curv damping scale factor"],
    ["level set", "curvature", "curv implementation"],
    ["level set", "curvature", "curv verbosity level"],
    ["level set", "curvature", "curv do narrow band"],
    ["level set", "curvature", "narrow band threshold"],
    ["level set", "ls do curvature correction"],
    ["level set", "ls do reinitialization"],
    ["level set", "ls n initial reinit steps"],
    ["level set", "ls n subdivisions"],
    ["level set", "ls do localized heaviside"],
    ["level set", "tol reinit"],
    ["level set", "reinitialization", "reinit max n steps"],
    ["level set", "reinitialization", "reinit modeltype"],
    ["level set", "reinitialization", "reinit implementation"],
    ["Navier-Stokes"],
    ["base", "gravity"],
    ["surface tension"],
    ["darcy damping"],
    ["base", "degree"],
    ["base", "do simplex"],
    ["heat", "degree"],
    ["material", "sticking constant"],
    ["base", "application name"],
    # 24-12-19
    ["advection diffusion", "time integration scheme"],
    ["level set", "advection diffusion", "time integration scheme"]
    # ... add old parameter names
    # ["old", "my age"],
]

new_parameter_names = [
    ["laser", "starting position"],
    ["heat", "use volume-specific thermal capacity for phase interpolation"],
    ["rte", "predictor type"],
    ["rte", "pseudo time stepping", "time stepping", "max n steps"],
    ["rte", "pseudo time stepping", "time stepping", "time step size"],
    ["rte", "rte verbosity level"],
    ["laser", "model"],
    ["laser", "intensity profile"],
    ["laser", "power"],
    ["laser", "power over time"],
    ["laser", "power start time"],
    ["laser", "power end time"],
    ["laser", "absorptivity gas"],
    ["laser", "absorptivity liquid"],
    ["laser", "do move"],
    ["laser", "scan speed"],
    ["laser", "radius"],
    ["laser", "gusarov", "reflectivity"],
    ["laser", "gusarov", "extinction coefficient"],
    ["laser", "gusarov", "layer thickness"],
    ["heat", "convection coefficient"],
    ["heat", "emissivity"],
    ["heat", "temperature infinity"],
    ["heat", "nlsolve", "max nonlinear iterations"],
    ["heat", "nlsolve", "field correction tolerance"],
    ["heat", "nlsolve", "residual tolerance"],
    ["heat", "nlsolve", "max nonlinear iterations alt"],
    ["heat", "nlsolve", "field correction tolerance alt"],
    ["heat", "nlsolve", "residual tolerance alt"],
    ["output", "directory"],
    ["output", "write frequency"],
    ["output", "write time step size"],
    ["output", "output variables"],
    ["output", "do user defined postprocessing"],
    ["output", "paraview", "enable"],
    ["output", "paraview", "filename"],
    ["output", "paraview", "n digits timestep"],
    ["output", "paraview", "print boundary id"],
    ["output", "paraview", "output subdomains"],
    ["output", "paraview", "output material id"],
    ["output", "paraview", "write higher order cells"],
    ["output", "paraview", "n groups"],
    ["output", "paraview", "n patches"],
    ["recoil pressure", "pressure coefficient"],
    ["recoil pressure", "temperature constant"],
    ["material", "gas", "thermal conductivity"],
    ["material", "gas", "specific heat capacity"],
    ["material", "gas", "density"],
    ["material", "gas", "dynamic viscosity"],
    ["material", "liquid", "thermal conductivity"],
    ["material", "liquid", "specific heat capacity"],
    ["material", "liquid", "density"],
    ["material", "liquid", "dynamic viscosity"],
    ["material", "solid", "thermal conductivity"],
    ["material", "solid", "specific heat capacity"],
    ["material", "solid", "density"],
    ["material", "solid", "dynamic viscosity"],
    ["material", "solidus temperature"],
    ["material", "liquidus temperature"],
    ["material", "boiling temperature"],
    ["material", "latent heat of evaporation"],
    ["material", "molar mass"],
    ["material", "sticking constant"],
    ["material", "specific enthalpy reference temperature"],
    ["material", "two phase fluid properties transition type"],
    ["material", "solid liquid properties transition type"],
    ["heat", "radiative boundary condition", "emissivity"],
    ["heat", "convective boundary condition", "convection coefficient"],
    ["heat", "convective boundary condition", "temperature infinity"],
    # 24-02-29
    ["evaporation", "analytical", "function"],
    ["evaporation", "interface temperature evaluation type"],
    ["evaporation", "evaporative mass flux model"],
    ["evaporation", "hardt wondra", "coefficient"],
    ["evaporation", "thickness integral", "subdivisions per side"],
    ["evaporation", "thickness integral", "subdivisions MCA"],
    ["evaporation", "formulation source term level set"],
    ["evaporation", "evaporative cooling", "model"],
    ["evaporation", "evaporative dilation rate", "model"],
    ["evaporation", "do level set pressure gradient interpolation"],
    ["evaporation", "recoil pressure"],
    ["evaporation", "evaporative cooling", "enable"],
    ["evaporation", "recoil pressure", "enable"],
    ["evaporation", "evaporative dilation rate", "enable"],
    ["evaporation", "evaporative cooling", "dirac delta function approximation"],
    ["evaporation", "recoil pressure", "type"],
    # 24-03-06
    ["level set"],
    ["level set", "reinitialization"],
    ["level set", "curvature"],
    ["level set", "normal vector"],
    ["level set", "advection diffusion"],
    ["level set", "advection diffusion", "diffusivity"],
    ["level set", "advection diffusion", "time integration scheme"],
    ["level set", "advection diffusion", "implementation"],
    ["level set", "normal vector", "filter parameter"],
    ["level set", "normal vector", "implementation"],
    ["level set", "normal vector", "verbosity level"],
    ["level set", "normal vector", "narrow band", "enable"],
    ["level set", "normal vector", "narrow band", "level set threshold"],
    ["level set", "curvature", "filter parameter"],
    ["level set", "curvature", "implementation"],
    ["level set", "curvature", "verbosity level"],
    ["level set", "curvature", "narrow band", "enable"],
    ["level set", "curvature", "narrow band", "level set threshold"],
    ["level set", "curvature", "do curvature correction"],
    ["level set", "reinitialization", "enable"],
    ["level set", "reinitialization", "n initial steps"],
    ["level set", "n subdivisions"],
    ["level set", "do localized heaviside"],
    ["level set", "reinitialization", "tolerance"],
    ["level set", "reinitialization", "max n steps"],
    ["level set", "reinitialization", "type"],
    ["level set", "reinitialization", "implementation"],
    ["flow"],
    ["flow", "gravity"],
    ["flow", "surface tension"],
    ["flow", "darcy damping"],
    ["base", "fe", "degree"],
    ["base", "fe", "type"],
    ["heat", "fe", "degree"],
    ["evaporation", "recoil pressure", "sticking constant"],
    ["base", "case name"],
    # 24-12-19
    ["advection diffusion", "time integration", "type"],
    ["level set", "advection diffusion", "time integration", "type"]
    # ... add new parameter names
    # ["new", "new", "my new age"],
]

# Optional: attach lambda function to modify value of new parameter name
new_parameter_names_lambda = [
    (["recoil pressure", "pressure coefficient"], lambda x: x * 1.e-5 / 1.013),
    (["base", "fe", "type"], lambda x: "FE_SimplexP" if x ==
     "true" else "FE_Q" if x == "false" else x),
]

rename_parameter_values = [
    # add new parameter values to be renamed
    # (["parameter", "category"], "old_value", "new_value"),
    (["heat", "use volume-specific thermal capacity for phase interpolation"],
     "smooth", "true"),
    (["recoil pressure", "interface distributed flux type"],
     "continuous", "local_value"),
    (["laser", "model"],
     "interface_projection", "interface_projection_regularized"),
    # 24-02-29
    (["evaporation", "analytical", "function"], "constant", "analytical"),
    (["evaporation", "interface temperature evaluation type"],
     "continuous", "local_value"),
    (["evaporation", "interface temperature evaluation type"],
     "interface value", "interface_value"),
    (["evaporation", "interface temperature evaluation type"],
     "line integral", "thickness_integral"),
    (["evaporation", "formulation source term level set"],
     "interface_velocity", "interface_velocity_local"),
    (["evaporation", "formulation source term heat"], "diffuse", "regularized"),
    (["evaporation", "formulation source term continuity"], "diffuse", "regularized"),
    (["evaporation", "evaporative cooling", "model"], "diffuse", "regularized"),
    (["evaporation", "evaporative dilation rate", "model"], "diffuse", "regularized"),
    (["evaporation", "recoil pressure", "interface distributed flux type"],
     "continuous", "local_value"),
]

delete_parameter_names = [
    # ... add parameter names to be deleted
    ["heat", "interpolate k"],
    ["material", "material melting point"],
    # 24-02-29
    ["evaporation", "evapor evaporative mass flux scale factor"],
    ["evaporation", "evapor ls value liquid"],
    ["evaporation", "evapor ls value gas"],
    ["problem specific", "do evaporation"],
    ["problem specific", "do evaporative mass flux"],
    # 24-03-06
    ["level set", "ls time integration scheme"],
    ["level set", "ls reinit time step size"],
    ["level set", "ls implementation"],
    ["base", "n q points 1d"],
    ["heat", "n q points 1d"],
    ["heat", "n subdivisions"],
]

# extend this function to process special parameter changes


def process_special_parameters(dataDict, nErrors):
    try:
        val = get_nested_value(
            dataDict, ["level set", "reinitialization", "reinit constant epsilon"])
        if float(val) > 0:
            delete_nested_item(
                dataDict, ["level set", "reinitialization", "reinit constant epsilon"])
            set_nested_item(
                dataDict, ["level set", "reinitialization",
                           "interface thickness parameter", "type"], "absolute_value")
            set_nested_item(dataDict, [
                            "level set", "reinitialization",
                            "interface thickness parameter", "val"], val)
            print_success(
                f"RENAME: 'reinit constant epsilon'")
            nErrors += 1
    except BaseException:
        try:
            val = get_nested_value(
                dataDict, ["level set", "reinitialization", "reinit scale factor epsilon"])
            if float(val) > 0:
                delete_nested_item(
                    dataDict, ["level set", "reinitialization", "reinit scale factor epsilon"])
                set_nested_item(
                    dataDict, ["level set", "reinitialization",
                               "interface thickness parameter",
                               "type"],
                    "proportional_to_cell_size")
                set_nested_item(dataDict, [
                                "level set", "reinitialization",
                                "interface thickness parameter", "val"], val)
                print_success(
                    f"RENAME: 'reinit scale factor epsilon'")
                nErrors += 1
        except BaseException:
            pass
    # rename n subdivisisons
    try:
        val = get_nested_value(
            dataDict, ["level set", "n subdivisions"])
        if float(val) > 1:
            delete_nested_item(
                dataDict, ["level set", "n subdivisions"])
            set_nested_item(
                dataDict, ["level set", "fe", "type"],
                "FE_Q_iso_Q1")
            set_nested_item(
                dataDict, ["level set", "fe", "degree"],
                val)
            print_success(
                f"RENAME: 'level set n subdivisions'")
            nErrors += 1
    except BaseException:
        pass
    try:
        val = get_nested_value(
            dataDict, ["level set", "curvature", "verbosity level"])
        if float(val) == 1:
            set_nested_item(
                dataDict, ["level set", "curvature", "verbosity level"],
                "2")
            nErrors += 1
    except BaseException:
        pass
    try:
        val = get_nested_value(
            dataDict, ["level set", "normal vector", "verbosity level"])
        if float(val) == 1:
            set_nested_item(
                dataDict, ["level set", "normal vector", "verbosity level"],
                "2")
            nErrors += 1
    except BaseException:
        pass

    return (dataDict, nErrors)

##############################################################################
#         END OF USER INPUT
##############################################################################


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_error(message):
    print(f"{bcolors.FAIL}ERROR: " + message + f"{bcolors.ENDC}")


def print_success(message):
    print(f"{bcolors.OKGREEN}" + message + f"{bcolors.ENDC}")


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_nested_value(dataDict, mapList):
    """Get item value in nested dictionary"""
    return reduce(getitem, mapList[:-1], dataDict)[mapList[-1]]


def set_nested_item(dataDict, mapList, val):
    """Set item in nested dictionary"""
    # check if tree of new parameter exists
    try:
        get_nested_value(dataDict, mapList)
    # if it does not exist, introduce new tree
    except BaseException:
        print_error("Category does not exist yet")
        tree_dict = create_dict_from_tree_list(mapList)
        update(dataDict, tree_dict)
    reduce(getitem, mapList[:-1], dataDict)[mapList[-1]] = val
    return dataDict


def delete_nested_item(dataDict, mapList):
    """Delete item in nested dictionary"""
    del reduce(getitem, mapList[:-1], dataDict)[mapList[-1]]


def remove_empty_items(d):
    final_dict = {}

    for a, b in d.items():
        if b:
            if isinstance(b, dict):
                final_dict[a] = remove_empty_items(b)
            elif isinstance(b, list):
                final_dict[a] = list(
                    filter(None, [remove_empty_items(i) for i in b]))
            else:
                final_dict[a] = b
    return final_dict


def remove_empty_nested_items(d):
    for i in range(50):  # max 50 nested parameters
        d = remove_empty_items(d)
    return (d)


def create_dict_from_tree_list(tree_list):
    tree_dict = {}
    for key in reversed(tree_list):
        tree_dict = {key: tree_dict}
    return tree_dict


def sanity_check_json(j, old_parameter_names, new_parameter_names,
                      new_parameter_names_lambda,
                      delete_parameter_names, rename_parameter_values, always_yes, write_json, appendix=""):
    assert len(old_parameter_names) == len(new_parameter_names)

    errors = 0

    with open(j, 'r') as f:
        print(70 * "-")
        print("Process file: {:}".format(j))
        datastore = json.load(f, object_pairs_hook=collections.OrderedDict)

        val = datastore

        # process parameters to be renamed
        for i, o in enumerate(old_parameter_names):
            # get value of old parameter
            try:
                val = get_nested_value(datastore, o)

                # TODO: BRUTEFORCE WORKAROUND!!!! [MS]
                # special treatment for new application structure in MeltPoolDG
                if "mp-advec-diff" in j and new_parameter_names[i][0] == "level set":
                    new_parameter_names[i] = new_parameter_names[i][1:]
                if new_parameter_names[i] == old_parameter_names[i]:
                    print("nothing to do")
                    continue

                print_error(
                    f"outdated parameter detected {o}; use new definition {new_parameter_names[i]}")
                errors += 1
                if write_json:
                    if always_yes or click.confirm(
                            'Do you want me to replace the old parameter by the new one?'):
                        # apply lambda function if it exists
                        for key, lambda_fun in new_parameter_names_lambda:
                            if key == new_parameter_names[i]:
                                new_val = lambda_fun(float(val))
                                print_success(
                                    f"CHANGE VALUE: Apply lambda function to change value of {new_parameter_names[i]} "
                                    f"from {val} to {new_val}")
                                val = str(new_val)

                        # replace parameter name and set value from the old
                        # parameter
                        set_nested_item(
                            datastore, new_parameter_names[i], val)

                        # delete the outdated parameter pair
                        delete_nested_item(datastore, o)
                        print_success(
                            f"RENAME: {o} to {new_parameter_names[i]}")

            except BaseException:
                continue

        # process deleted parameters
        for i, o in enumerate(delete_parameter_names):
            try:
                val = get_nested_value(datastore, o)
                print_error(f"non-existing parameter found {o}")
                errors += 1
                if write_json:
                    if always_yes or click.confirm(
                            'Do you want me to delete the outdated parameter?'):
                        delete_nested_item(datastore, o)
                        print_success("DELETE: parameter '{:}'".format(o))
            except BaseException:
                continue

        # process renamed parameter value
        for i, (name, old_val, new_val) in enumerate(rename_parameter_values):
            try:
                val = get_nested_value(datastore, name)
                if (str(val) == str(old_val)):
                    print_error(
                        f"outdated parameter value for {name} detected '{val}'; Use new value '{new_val}'")
                    errors += 1
                    if write_json:
                        if always_yes or click.confirm(
                                'Do you want me to replace the old parameter value by the new one?'):
                            # introduce new parameter name and set value from
                            # the old parameter
                            set_nested_item(datastore, name, new_val)
                            print(f"    RENAME: {name} {val} to {new_val}")
            except BaseException:
                continue
        # process special parmaeters
        datastore, errors = process_special_parameters(datastore, errors)

        # delete potentially empty items
        datastore = remove_empty_nested_items(datastore)

    if write_json:
        new_file = j
        if appendix:
            base = os.path.basename(j).split(".json")[0]
            new_file = os.path.join(
                os.path.dirname(j),
                base + appendix + ".json")

        with open(new_file, 'w') as f:
            print(f"Write file: updated parameters written to {new_file}")
            json.dump(datastore, f, indent=2, separators=(',', ': '))

    else:
        sys.exit(1)
        assert errors == 0, f"ERROR: {errors} errors in the parameter file {j} detected. Use the overwrite option '-w' to fix it."

    print_success(f"Done! {errors} errors fixed.")
    sys.exit(0)


def parseArguments():
    parser = argparse.ArgumentParser(description="Format json file.")

    parser.add_argument('--file', type=str, required=True, help="Input file.")
    parser.add_argument('-nv', action='store_true', help='Set this action to be not verbose.',
                        required=False)
    parser.add_argument('-w', action='store_true', help='Set this action to write files.',
                        required=False)
    parser.add_argument('-y', action='store_true', help='Set this action to automatically overwrite files.',
                        required=False)
    parser.add_argument('--appendix', type=str, required=False,
                        help="Appendix of written output file.")

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parseArguments()

    # block printing
    if args.nv:
        sys.stdout = open(os.devnull, 'w')

    sanity_check_json(args.file, old_parameter_names, new_parameter_names, new_parameter_names_lambda,
                      delete_parameter_names, rename_parameter_values, args.y, args.w, args.appendix)
