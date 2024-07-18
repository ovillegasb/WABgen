#!/bin/env python
# -*- coding: utf-8 -*-

"""Functionality to implement WABgen from the command line."""

import argparse
import os
import time
import random
from wabgen.io import parse_file, prepare_output_directory
from wabgen.core import SG, standardize_Molecule, pick_option
from wabgen.utils import symm
from wabgen.utils.align import read_rotation_dict
from wabgen.utils.profile import start_profiling
from wabgen.utils.allocate import zero_site_allocation, new_site_allocation, convert_new_SA_to_old
from wabgen.core import placement_table, weight_spacegroups, make_structures, log_memory_usage
from multiprocessing import Process, Manager, Lock


# find the directory
directory = os.path.dirname(__file__)

# local directory
cwd = os.getcwd()


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[91m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


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

Example:

python -m wabgen system.cell -n 100 --push_apart flexible --parallel

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
        default=100
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

    # Getting the variables defined in the options
    seed = args["seed"]
    N = args["number"]
    form = args["form"]
    sgb = [args["spacegroup_lower_bound"], args["spacegroup_upper_bound"]]
    sg_list = args["sg_list"]
    site_mult_restrictions = {}
    site_rank_restrictions = {}
    output_folder = "completed"

    print("\t{:<35}{:>20}".format("Input file: ", seed))

    if args["parallel"]:
        if args["nc"] is None:
            Nc = os.cpu_count() - 1
        else:
            Nc = args["nc"] - 1
    else:
        Nc = 1

    if Nc < 1:
        Nc = 1

    print("\t{:<35}{:>20}".format("Number of cpus: ", Nc + 1))
    print("\t{:<35}{:>20}".format("Number of files to generate: ", N))
    print("\t{:<35}{:>20}".format("Output format: ", form))
    print("\t{:<35}{:>20}".format("Range of selected spacegroups: ", "-".join(
        [str(i) for i in sgb]
    )))
    print("\t{:<35}{:>20}".format("Output folder: ", output_folder))
    print("")
    # Prepare output folder
    prepare_output_directory(output_folder)
    # prepare_output_directory("duplicates")
    # prepare_output_directory("rejected")

    # parse in the input file containing the molecules objects
    params = parse_file(seed, args["symmetry_tolerance"], template="True")

    mols = params["mols"]
    V_dist = params["V_dist"]
    min_seps = params["min_seps"]
    target_atom_nums = params["target_atom_nums"]
    cell_abc = params["cell_abc"]
    gps = params["gulp_potentials"]
    pressure = params["pressure"]
    Z_molecules = params["Z_molecules"]

    # for mol in mols:
    #     mol.print_symm_info()

    # for z in Z_molecules:
    #     print("Z=", z)
    #     mols_z = Z_molecules[z]["MOLS"]
    #     for mol in mols_z:
    #         mol.print_symm_info()

    # generate all instances of the spacegroup class here, should speed things up!
    t_i = time.time()
    print("reading in spacegroup representation")
    SpaceGroups = [[]]
    ConvSpaceGroups = [[]]
    for i in range(1, 231):
        SpaceGroups.append(SG(symm.retrieve_symmetry_group(i, reduce_to_prim=True)))
        ConvSpaceGroups.append(SG(symm.retrieve_symmetry_group(i, reduce_to_prim=False)))

    t_f = time.time()
    execution_time = t_f - t_i
    print("done in %.3f s" % execution_time)

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

    for z in Z_molecules:
        print("Z=", z)
        mols_z = Z_molecules[z]["MOLS"]
        for i, mol in enumerate(mols_z):
            print(i, mol)
            mol.print_symm_info()
            Z_molecules[z]["MOLS"][i] = standardize_Molecule(
                mol, temp_fname, args["symmetry_tolerance"]
            )

    t_f = time.time()
    execution_time = t_f - t_i
    print("done in %.3f s" % execution_time)

    # spacegroup list
    if sg_list is None:
        sg_list = [x for x in range(sgb[0], sgb[1]+1)]

    # TODO
    # p = start_profiling()

    # TODO
    # only_atoms = True
    # for mol in mols:
    #     if mol.Otype == "Mol":
    #         only_atoms = False

    ####
    # TODO
    t_i = time.time()
    new_options = {}
    # generating ooptions for z
    for z in Z_molecules:
        Z_molecules[z]["options"] = {}

    print("Generating options")

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

        # Doing for every Z value
        for z in Z_molecules:
            mols_z = Z_molecules[z]["MOLS"]
            zsa = zero_site_allocation(
                SpaceGroups[sg_ind],
                mols_z,
                placement_table,
                site_mult_restrictions=site_mult_restrictions,
                site_rank_restrictions=site_rank_restrictions
            )

            Z_molecules[z]["options"][sg_ind] = new_site_allocation(
                SpaceGroups[sg_ind],
                mols_z,
                placement_table,
                zsa,
                msr=site_mult_restrictions,
                srr=site_rank_restrictions,
                max_options=args["max_options"]
            )

    print("\nincluding rotations...")
    full_options = new_options
    # print(full_options)
    zkeys = []
    print("final options are...")
    for sg_ind in full_options:
        print(sg_ind, "/", len(sg_list), SpaceGroups[sg_ind].name, len(full_options[sg_ind]))
        if len(full_options[sg_ind]) == 0:
            zkeys.append(sg_ind)

    for sg_ind in zkeys:
        del full_options[sg_ind]

    for z in Z_molecules:
        full_options_Z = Z_molecules[z]["options"]
        zkeys = []

        for sg_ind in full_options_Z:
            if len(full_options_Z[sg_ind]) == 0:
                zkeys.append(sg_ind)

        for sg_ind in zkeys:
            del Z_molecules[z]["options"][sg_ind]

    # create a list of weighted options to use for the random loop
    sg_inds = [key for key in full_options]

    weighted_options, freq_list = weight_spacegroups(
        sg_inds,
        SpaceGroups,
        weighting=args["sg_weight"],
        filename=args["sg_fs_file"]
    )

    for z in Z_molecules:
        sg_inds = [key for key in Z_molecules[z]["options"]]
        Z_molecules[z]["weighted_options"], _ = weight_spacegroups(
            sg_inds,
            SpaceGroups,
            weighting=args["sg_weight"],
            filename=args["sg_fs_file"]
        )

    t_f = time.time()
    execution_time = t_f - t_i
    print("done in %.3f s" % execution_time)

    ####

    # set the base file name
    fname = cwd + "/" + seed.replace(".cell", "")

    # random allocation loop
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
        "Rot_dicts": Rot_dicts}

    ###############
    # BUCLE central
    ###############
    manager = Manager()
    # shared_dict_structure = manager.dict()
    counter = manager.Value('i', 0)
    lock = Lock()
    processes = []
    Nmade = 0
    log_memory_usage("Main process start")
    time.sleep(5)
    for _ in range(N):
        # Choosing a random Z value for each iteration.
        z_val = random.choice(list(Z_molecules.keys()))
        print(f"\tMolecular fomulate taked {z_val}")
        full_options = Z_molecules[z_val]["options"]
        z_weighted_options = Z_molecules[z_val]["weighted_options"]
        lw0 = len(z_weighted_options)
        if lw0 > 0:
            r = random.randint(0, len(z_weighted_options)-1)
        else:
            r = 0

        sg_ind = z_weighted_options[r]
        arg_dict["sg_ind"] = sg_ind
        arg_dict["sg"] = SpaceGroups[sg_ind]
        arg_dict["Z_val"] = z_val
        arg_dict["V_dist"] = Z_molecules[z_val]["V_dist"]
        full_mols = Z_molecules[z_val]["MOLS"]
        print(f"\tpicking perm for sg={sg_ind} from stratified options...")

        new_perm = pick_option(full_options[sg_ind])
        perm = convert_new_SA_to_old(new_perm, SpaceGroups[sg_ind], placement_table, full_mols)
        perm = [-1, perm]  # modification as now expects [dof, perm]
        arg_dict["mols"] = full_mols

        p = Process(
            target=make_structures,
            args=(perm, fname, arg_dict, lock, counter)
        )
        p.start()
        processes.append(p)

        Nmade += 1
        print(f"\033[1;34mStructures submitted: {Nmade:04d}/{N:04d}\033[0m")
    """
    while Nmade < N:
        # 1. Update the number of processes
        print(Nmade)
        time.sleep(5)
        while len(procs) >= Nc or N-Nmade <= len(procs) and N-Nmade > 0:
            time.sleep(0.1)
            for p in procs:
                if p.exitcode == 0:
                    Nmade += 1
                    print("\r", Nmade, "/", N, end="")
            procs = [p for p in procs if p.exitcode is None]

        # 2. Spawn a new process
        if N-Nmade > len(procs):
            # Choosing a random Z value for each iteration
            z_val = random.choice(list(Z_molecules.keys()))

            full_options = Z_molecules[z_val]["options"]
            z_weighted_options = Z_molecules[z_val]["weighted_options"]
            lw0 = len(z_weighted_options)

            if lw0 > 0:
                r = random.randint(0, len(z_weighted_options)-1)
            else:
                r = 0

            sg_ind = z_weighted_options[r]
            arg_dict["sg_ind"] = sg_ind
            arg_dict["sg"] = SpaceGroups[sg_ind]
            arg_dict["Z_val"] = z_val
            arg_dict["V_dist"] = Z_molecules[z_val]["V_dist"]
            full_mols = Z_molecules[z_val]["MOLS"]

            print(f"picking perm for sg={sg_ind} from stratified options...")
            new_perm = pick_option(full_options[sg_ind])
            perm = convert_new_SA_to_old(new_perm, SpaceGroups[sg_ind], placement_table, full_mols)
            perm = [-1, perm]  # modification as now expects [dof, perm]
            arg_dict["mols"] = full_mols

            p = Process(target=make_structures, args=(perm, fname, arg_dict))
            p.start()
            procs.append(p)
            processes.append(p)
    """

    ###############

    for process in processes:
        process.join()

    log_memory_usage("Main process end")
    time.sleep(5)

    # final_structures = 0
    # for hash_comp in shared_dict_structure:
    #     final_structures += len(shared_dict_structure[hash_comp])
    # print('Final shared structures generates:', final_structures)
    print('Final counter value:', counter.value)

    # if final_structures != counter.value:
    #     print(bcolors.WARNING + "WARNING!" + bcolors.ENDC, end=" - ")
    #     print("The generated structures do not match the counted structures")

    print('\nAll computations are done')
    print(f"\t\033[1;32mStructures finished: {counter.value:04d}/{N:04d}\033[0m")

    t_f_total = time.time()
    execution_time = t_f_total - t_i_total
    print("\tElapsed time done in %.3f s" % execution_time)


if __name__ == '__main__':
    main()
