#!/bin/env python
# -*- coding: utf-8 -*-

"""Functionality to implement WABgen from the command line."""

import argparse
import os
import time
import random
import string
from wabgen.io import parse_file
from wabgen.core import SG, standardize_Molecule, pick_option
from wabgen.utils import symm
from wabgen.utils.align import read_rotation_dict
from wabgen.utils.profile import start_profiling
from wabgen.utils.allocate import zero_site_allocation, new_site_allocation, convert_new_SA_to_old
from wabgen.core import placement_table, weight_spacegroups, make_perm, add_hash
from multiprocessing import Process


TITLE = """\033[1;36m
 _    _  ___  ______
| |  | |/ _ \\ | ___ \\
| |  | / /_\\ \\| |_/ / __ _  ___ _ __
| |/\\| |  _  || ___ \\/ _` |/ _ \\ '_ \\
\\  /\\  / | | || |_/ / (_| |  __/ | | |
 \\/  \\/\\_| |_/\\____/ \\__, |\\___|_| |_|
                      __/ |
                     |___/
\033[m
WABgen python module used to build MOFs using block alignment at Wycoff positions.

This project was inspired and initiated by the WAM code.

Author: Orlando Villegas
Date: 2024-07-08
----------------
"""


def options():
    """Generate command line interface."""
    parser = argparse.ArgumentParser(
        prog="wabgen",
        usage="%(prog)s [-options]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Enjoy the program!"
    )

    settings = parser.add_argument_group(
        "\033[1;36mInitial settings\033[m")

    settings.add_argument(
        "seed",
        type=str,
        help="seed name to label the ouput files"
    )

    settings.add_argument(
        "-n", "--number",
        type=int,
        default=1,
        help="number of .res files to create for random airss or number per perm if doing exhausti\
ve")

    settings.add_argument(
        "-f", "--form",
        type=str,
        default="res",
        help="format of the output files. Can be cell or res")

    settings.add_argument(
        "-sglb", "--spacegroup_lower_bound",
        type=int,
        default=1,
        help="defines the minimum value of spacegroup starting from 1"
    )

    settings.add_argument(
        "-sgub", "--spacegroup_upper_bound",
        type=int,
        default=230,
        help="defines the maximum value of spacegroup ending with 230."
    )

    settings.add_argument(
        "-fom", "--filter_out_molecules",
        action="store_true"
    )

    settings.add_argument(
        "-ptol", "--filter_ptol",
        type=float,
        default=10.0
    )

    settings.add_argument(
        "--sg_list",
        type=int,
        default=None,
        help="Selection of space groups (1-230).",
        nargs="+"
    )

    settings.add_argument(
        "-stol", "--symmetry_tolerance",
        type=float,
        default=1e-03
    )

    settings.add_argument(
        "--sg_weight",
        default="ranks"
    )

    settings.add_argument(
        "--sg_fs_file",
        default=None
    )

    settings.add_argument(
        "-n_tries",
        type=int,
        default=50
    )

    settings.add_argument(
        "--exact_sg",
        action="store_true",
        default=False
    )

    settings.add_argument(
        "--no_supercells",
        action="store_true",
        default=False
    )

    settings.add_argument(
        "--push_apart",
        type=str,
        default=False,
        help="None or flexible"
    )

    settings.add_argument(
        "--minsep_noise",
        type=float,
        default=0.2
    )

    settings.add_argument(
        "--aenet_relax",
        action="store_true",
        default=False
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False
    )

    parser.add_argument(
        "-nc",
        type=int,
        default=None
    )

    parser.add_argument(
        "--max_options", "-mo",
        type=int,
        default=1e3,
        help="max options before truncating enumeration"
    )

    return vars(parser.parse_args())


def main():
    """Run main function."""
    print(TITLE)
    args = options()
    t_i_total = time.time()

    output_folder = "completed"
    if not os.path.exists(output_folder):
        print(f"The {output_folder} folder does not exist, it will be created.")
        os.mkdir(output_folder)

    cwd = os.getcwd()
    directory = os.path.dirname(__file__)

    seed = args["seed"]
    N = args["number"]
    form = args["form"]
    sgb = [args["spacegroup_lower_bound"], args["spacegroup_upper_bound"]]
    fom = args["filter_out_molecules"]  # TODO: Que hace?
    ptol = args["filter_ptol"]          # TODO: Que hace?
    sg_list = args["sg_list"]
    site_mult_restrictions = {}
    site_rank_restrictions = {}

    print("\t{:<35}{:>20}".format("Input file: ", seed))
    print("\t{:<35}{:>20}".format("Number of files to generate: ", N))
    print("\t{:<35}{:>20}".format("Output format: ", form))
    print("\t{:<35}{:>20}".format("Range of selected spacegroups: ", "-".join([str(i) for i in sgb])))

    # parse in the input file containing the molecules objects
    params = parse_file(seed, args["symmetry_tolerance"], template="True")
    print(params)

    mols = params["mols"]
    V_dist = params["V_dist"]
    min_seps = params["min_seps"]
    target_atom_nums = params["target_atom_nums"]
    cell_abc = params["cell_abc"]
    merge_groups = params["merge_groups"]
    pdict = params["p_dict"]
    gps = params["gulp_potentials"]
    pressure = params["pressure"]
    # premade_mols = params["p_mols"]

    for mol in mols:
        mol.print_symm_info()

    # generate all instances of the spacegroup class here, should speed things up!
    t_i = time.time()
    print("reading in spacegroup representation...")
    SpaceGroups = [[]]
    ConvSpaceGroups = [[]]
    for i in range(1, 231):
        SpaceGroups.append(SG(symm.retrieve_symmetry_group(i, reduce_to_prim=True)))
        ConvSpaceGroups.append(SG(symm.retrieve_symmetry_group(i, reduce_to_prim=False)))

    t_f = time.time()
    execution_time = t_f - t_i
    print("..done in %.3f s" % execution_time)

    # standardize the molecules
    t_i = time.time()
    print("standardiszing molecules")
    temp_fname = directory + "/data/templates.cell"
    Rot_dicts = {}
    for i, mol in enumerate(mols):
        print(i, mol)
        mol.print_symm_info()
        mols[i] = standardize_Molecule(mol, temp_fname, args["symmetry_tolerance"])
        if mol.Otype == "Mol":
            Rot_dicts[mol.name] = read_rotation_dict(mols[i], SpaceGroups)

    t_f = time.time()
    execution_time = t_f - t_i
    print("..done in %.3f s" % execution_time)

    # spacegroup list
    if sg_list is None:
        sg_list = [x for x in range(sgb[0], sgb[1]+1)]

    # p = start_profiling()

    only_atoms = True
    for mol in mols:
        if mol.Otype == "Mol":
            only_atoms = False

    ####
    # TODO
    new_options = {}
    print("generating options...")
    for j, sg_ind in enumerate(sg_list):
        print("\r", j+1, "/", len(sg_list), SpaceGroups[sg_ind].name, end="\n")
        zsa = zero_site_allocation(
            SpaceGroups[sg_ind],
            mols,
            placement_table,
            site_mult_restrictions=site_mult_restrictions,
            site_rank_restrictions=site_rank_restrictions
        )
        print("\r", j+1, "/", len(sg_list), SpaceGroups[sg_ind].name, "finished 0D allocations", end="")

        new_options[sg_ind] = new_site_allocation(
            SpaceGroups[sg_ind],
            mols,
            placement_table,
            zsa,
            msr=site_mult_restrictions,
            srr=site_rank_restrictions,
            max_options=args["max_options"]
        )
    print("\nincluding rotations...")
    full_options = new_options
    zkeys = []
    print("final options are...")
    for sg_ind in full_options:
        print(sg_ind, "/", len(sg_list), SpaceGroups[sg_ind].name, len(full_options[sg_ind]))
        if len(full_options[sg_ind]) == 0:
            zkeys.append(sg_ind)

    for sg_ind in zkeys:
        del full_options[sg_ind]

    # create a list of weighted options to use for the random loop
    sg_inds = [key for key in full_options]

    weighted_options, freq_list = weight_spacegroups(
        sg_inds,
        SpaceGroups,
        weighting=args["sg_weight"],
        filename=args["sg_fs_file"]
    )

    ####

    # set the base file name
    fname = cwd + "/" + seed.replace(".cell", "")
    print(fname)
    i = 0

    # prepare the origins file and add the cell and args and strings
    o_name = "origins_"+seed
    print(o_name)
    o_name = add_hash(o_name, 7) + ".txt"
    print(o_name)

    # random allocation looop
    arg_dict = {
        "min_seps": min_seps,
        "V_dist": V_dist,
        "cell_abc": cell_abc,
        "sg_ind": 1,
        "mols": mols,
        "n_tries": args["n_tries"],
        "target_atom_nums": target_atom_nums,
        "form": form,
        "exact_sg": args["exact_sg"],
        "no_supercells": args["no_supercells"],
        "push_apart": args["push_apart"],
        "gps": gps,
        "pressure": pressure,
        "minsep_noise": args["minsep_noise"],
        "aenet_relax": args["aenet_relax"],
        "o_name": o_name,
        "Rot_dicts": Rot_dicts}

    if args["parallel"]:
        if args["nc"] is None:
            Nc = os.cpu_count()-1
        else:
            Nc = args["nc"]-1
    else:
        Nc = 1

    if Nc < 1:
        Nc = 1

    print("\n\n\n Nc=", Nc, "\n\n\n")

    ###############
    # BUCLE central
    ###############
    procs = []
    Nmade = 0

    print(N-Nmade)
    print(Nc)
    while Nmade < N:
        # 1. Update the number of processes??????
        while len(procs) >= Nc or N-Nmade <= len(procs) and N-Nmade > 0:
            time.sleep(0.1)
            for p in procs:
                if p.exitcode == 0:
                    Nmade += 1
                    print("\r", Nmade, "/", N, end="")
            procs = [p for p in procs if p.exitcode is None]

        # 2. Spawn a new process
        if N-Nmade > len(procs):
            lw0 = len(weighted_options)
            print(lw0)
            if lw0 > 0:
                r = random.randint(0, len(weighted_options)-1)
            else:
                r = 0
            print(r)
            sg_ind = weighted_options[r]
            print(sg_ind)
            arg_dict["sg_ind"] = sg_ind
            arg_dict["sg"] = SpaceGroups[sg_ind]

            full_mols = mols
            print(f"picking perm for sg={sg_ind} from stratified options...")
            new_perm = pick_option(full_options[sg_ind])
            print(new_perm)
            perm = convert_new_SA_to_old(new_perm, SpaceGroups[sg_ind], placement_table, full_mols)
            perm = [-1, perm]  # modification as now expects [dof, perm]

            arg_dict["mols"] = full_mols
            p = Process(target=make_perm, args=(perm, fname, arg_dict))
            p.start()
            procs.append(p)

            # exit()

    ###############
    t_f_total = time.time()
    execution_time = t_f_total - t_i_total
    print("\tElapsed time done in %.3f s" % execution_time)



if __name__ == '__main__':
    main()
