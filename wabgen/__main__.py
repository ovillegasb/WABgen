#!/bin/env python
# -*- coding: utf-8 -*-

"""Functionality to implement WABgen from the command line."""

import argparse
import os
import time
import random
import psutil
import glob
import ase
import copy
import numpy as np
from wabgen.io import parse_file, prepare_output_directory
from wabgen.core import SG, Molecule, GEOM_constraint
from wabgen.utils import symm
from wabgen.utils.align import read_rotation_dict
from wabgen.utils.allocate import zero_site_allocation, new_site_allocation, convert_new_SA_to_old
from wabgen.core import placement_table, weight_spacegroups, log_memory_usage, make_test
from wabgen.core import structure_generator
from wabgen.utils.filter import duplicate_checker
from multiprocessing import Process, Manager, Lock, Event


# find the directory
directory = os.path.dirname(__file__)

# local directory
cwd = os.getcwd()


class colors:
    """
    Colors class.

    Reset all colors with colors.reset; two sub classes fg for foreground and bg for
    background; use as colors.subclass.colorname.

    i.e. colors.fg.red or colors.bg.greenalso, the generic bold, disable, underline, reverse,
    strike through, and invisible work with the main class i.e. colors.bold
    """

    reset = '\033[0m'
    bold = '\033[01m'

    class fg:
        """Foreground colors."""

        red = '\033[31m'
        green = '\033[32m'
        purple = '\033[35m'
        lightgrey = '\033[37m'


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

python -m wabgen system.cell -n 10 --parallel

python -m wabgen system.cell -n 100 --push_apart flexible --parallel --sg_list 1

python -m wabgen system.cell -n 100 --parallel --sg_list 1-50

python -m wabgen system.cell -n 100 --parallel --sg_list 1 5 10 50

python -m wabgen system.cell -n 10 --parallel -nc 1 --push_apart flexible --minsep_noise 0.1

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

    config = parser.add_argument_group(
        "\033[1;36mConfigure structure generation\033[m")

    config.add_argument(
        "--parallel",
        action="store_true",
        default=False
    )

    config.add_argument(
        "-nc", "--n_cpus",
        type=int,
        default=None
    )

    config.add_argument(
        "--max_options", "-mo",
        type=int,
        default=1e3,
        help="max options before truncating enumeration"
    )

    config.add_argument(
        "--test",
        action="store_true",
        default=False
    )

    config.add_argument(
        "--not_reset",
        action="store_true",
        default=False,
        help="Does not delete generated structures contained in the folder “completed”"
    )

    return vars(parser.parse_args())


def pick_option(sg_opts):
    """
    Pick an option from {sg_ind: {total_sites: [options]}}.

    weight to favour using fewer sites. P = 1/2 for fewest sites, P = 1/4 for next fewest etc.
    Could argue this is not aggressive enough...
    could also argue that there will be fewer options for lower numbers of sites and this is
    already exponential!
    """
    n_sites = sorted(list(sg_opts.keys()))
    ws = [1/(2**i) for i in range(len(n_sites))]
    ps = [w/sum(ws) for w in ws]
    n = np.random.choice(n_sites, p=ps)
    opt = np.random.choice(sg_opts[n])
    return opt


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
    not_reset = args["not_reset"]
    site_mult_restrictions = {}
    site_rank_restrictions = {}
    output_folder = "completed"
    n_completed = 0
    print("\t{:<35}{:>20}".format("Input file: ", seed))

    if args["parallel"]:
        if args["n_cpus"] is None:
            Nc = os.cpu_count() - 1
        else:
            Nc = args["n_cpus"] - 1
    else:
        Nc = 1

    if Nc < 1:
        Nc = 1

    print("\t{:<35}{:>20}".format("Number of cpus: ", Nc + 1))
    print("\t{:<35}{:>20}".format("Number of files to generate: ", N))
    print("\t{:<35}{:>20}".format("Output format: ", form))

    # Checking sg_list
    if sg_list is not None:
        try:
            sg_list = [int(sg) for sg in sg_list]
        except ValueError:
            if len(sg_list) == 1:
                sg_i, sg_e = [int(sg) for sg in sg_list[0].split("-")]
                sg_list = list(range(sg_i, sg_e + 1))

        if len(sg_list) == 1:
            sgb = sg_list.copy()

        else:
            sgb = sorted(sg_list.copy())
            sgb = [sgb[0], sgb[-1]]

    # spacegroup list selected.
    if sg_list is None:
        sg_list = [x for x in range(sgb[0], sgb[1]+1)]

    print("\t{:<35}{:>20} ({})".format("Range of selected spacegroups: ", "-".join(
        [str(i) for i in sgb]
    ), len(sg_list)))

    print("\t{:<35}{:>20}".format("Output folder: ", output_folder))
    print("")
    # Prepare output folder
    removed = prepare_output_directory(output_folder, not_reset)
    if not removed:
        n_completed = len(glob.glob(f"./{output_folder}/*.{form}"))
        print("\t{:<35}{:>20}".format("Number of files generated: ", n_completed))

        if n_completed >= N:
            print("Completed files meet the generation criteria increase N or delete files.")
            exit()

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
    geom_constraint = params["geom_constraint"]

    # generate all instances of the spacegroup class here, should speed things up!
    t_i = time.time()
    print("\nReading in spacegroup representation from selected")

    SpaceGroups = {}
    # ConvSpaceGroups = [[]]
    for i in sg_list:
        SpaceGroups[i] = SG(symm.retrieve_symmetry_group(i, reduce_to_prim=True))
        # ConvSpaceGroups.append(SG(symm.retrieve_symmetry_group(i, reduce_to_prim=False)))

    t_f = time.time()
    execution_time = t_f - t_i
    print("done in %.3f s" % execution_time)

    # standardize the molecules? Why? TODO
    t_i = time.time()
    # print("standardiszing molecules")
    # temp_fname = directory + "/data/templates.cell"
    # Rot_dicts? TODO. definition
    Rot_dicts = {}
    if geom_constraint is not None:
        constraint = GEOM_constraint(d=1.0)

    mols = Z_molecules[list(Z_molecules.keys())[0]]["MOLS"]
    for i, mol in enumerate(mols):
        if mol.name in geom_constraint:
            # strategy
            pg = geom_constraint[mol.name]
            geom = constraint.set_point_group(pg)
            geom += ase.Atom(mol.name, (0, 0, 0))
            new_mol = Molecule(
                    geom.get_chemical_symbols(),
                    geom.get_positions(),
                    number=mol.number,
                    Otype="Mol",
                    name=f"coord-{mol.name}",
                    constraint="coordination"
                )
            Rot_dicts[new_mol.name] = read_rotation_dict(new_mol, SpaceGroups)
            new_mol.species = [at.replace("H", "X") for at in new_mol.species]
            mols[i] = new_mol
        else:
            if mol.Otype == "Mol":
                Rot_dicts[mol.name] = read_rotation_dict(mol, SpaceGroups)

    for z in Z_molecules:
        mols_z = Z_molecules[z]["MOLS"]
        for i, mol in enumerate(mols_z):
            if mol.name in geom_constraint:
                new_mol = copy.deepcopy(mols[i])
                new_mol.number = mol.number
                Z_molecules[z]["MOLS"][i] = new_mol

        Z_molecules[z]["ROT_dict"] = Rot_dicts
        Z_molecules[z]["options"] = {}

    t_f = time.time()
    execution_time = t_f - t_i
    print("done in %.3f s" % execution_time)

    # Generation options part
    t_i = time.time()
    # new_options = {}
    print("Generating options")
    n_sg_list = len(sg_list)
    for j, sg in enumerate(sg_list):
        # print("\r", j+1, "/", n_sg_list, SpaceGroups[sg].name, end="\n")
        # zsa = zero_site_allocation(
        #     SpaceGroups[sg],
        #     mols,
        #     placement_table,
        #     site_mult_restrictions=site_mult_restrictions,
        #     site_rank_restrictions=site_rank_restrictions
        # )

        # print("\r", j+1, "/", n_sg_list, SpaceGroups[sg].name, "finished 0D allocations")  # , end=""
        # new_options[sg] = new_site_allocation(
        #     SpaceGroups[sg],
        #     mols,
        #     placement_table,
        #     zsa,
        #     msr=site_mult_restrictions,
        #     srr=site_rank_restrictions,
        #     max_options=args["max_options"]
        # )

        # Doing for every Z value
        for z in Z_molecules:
            mols_z = Z_molecules[z]["MOLS"]
            # print("Val Z =", z)
            # print("\r", j+1, "/", n_sg_list, SpaceGroups[sg].name, end="\n")
            zsa = zero_site_allocation(
                SpaceGroups[sg],
                mols_z,
                placement_table,
                site_mult_restrictions=site_mult_restrictions,
                site_rank_restrictions=site_rank_restrictions
            )

            # print("\r", j+1, "/", n_sg_list, SpaceGroups[sg].name, "finished 0D allocations")  # , end=""
            Z_molecules[z]["options"][sg] = new_site_allocation(
                SpaceGroups[sg],
                mols_z,
                placement_table,
                zsa,
                msr=site_mult_restrictions,
                srr=site_rank_restrictions,
                max_options=args["max_options"]
            )

    print("\nincluding rotations...")
    # full_options = new_options
    # zkeys = []
    print("final options accepted are...")
    # for j, sg in enumerate(full_options):
    #     print(j + 1, "/", n_sg_list, SpaceGroups[sg].name, len(full_options[sg]))
    #     if len(full_options[sg]) == 0:
    #         zkeys.append(sg)

    # for sg in zkeys:
    #     del full_options[sg]

    for z in Z_molecules:
        full_options_Z = Z_molecules[z]["options"]
        zkeys = []

        for sg in full_options_Z:
            if len(full_options_Z[sg]) == 0:
                zkeys.append(sg)

        for sg in zkeys:
            del Z_molecules[z]["options"][sg]

    # print("Z_molecules", Z_molecules)

    # create a list of weighted options to use for the random loop
    # sg_accepted = [key for key in full_options]
    # weighted_options, freq_list = weight_spacegroups(
    #     sg_accepted,
    #     SpaceGroups,
    #     weighting=args["sg_weight"],
    #     filename=args["sg_fs_file"]
    # )

    for z in Z_molecules:
        sg_selected = [key for key in Z_molecules[z]["options"]]
        Z_molecules[z]["weighted_options"], _ = weight_spacegroups(
            sg_selected,
            SpaceGroups,
            weighting=args["sg_weight"],
            filename=args["sg_fs_file"]
        )

    t_f = time.time()
    execution_time = t_f - t_i
    print("done in %.3f s" % execution_time)

    # set the base file name
    fname = cwd + "/" + seed.replace(".cell", "")

    # random allocation loop
    arg_dict = {
        "min_seps": min_seps,
        "V_dist": V_dist,
        "cell_abc": cell_abc,
        "sg_ind": 1,
        "mols": None,
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
        "Rot_dicts": None
    }

    ###############
    # BUCLE central
    ###############
    manager = Manager()
    # shared_dict_structure = manager.dict()
    counter = manager.Value('i', n_completed)
    lock = Lock()
    processes = []
    # Make and run the process to verify duplicates
    result_queue = Manager().Queue()
    stop_event = Event()
    checker = Process(target=duplicate_checker, args=(result_queue, stop_event))
    checker.start()
    checker_process = psutil.Process(checker.pid)
    try:
        checker_process.cpu_affinity([0])
    except OSError:
        pass

    log_memory_usage("Main process start")
    while counter.value <= N:
        if counter.value == N:
            break
        # Print some information
        print(f"{colors.bold}{colors.fg.green}------------------------------------{colors.reset}")
        print(f"{colors.bold}{colors.fg.red}Main BUCLE{colors.reset}")
        print(f"{colors.bold}{colors.fg.red}N Process: {len(processes)}/{Nc}{colors.reset}")
        while len(processes) >= Nc or (N-counter.value <= len(processes) and N-counter.value > 0):
            time.sleep(0.1)
            for p in processes:
                try:
                    pid = psutil.Process(p.pid)
                    cpu_times = pid.cpu_times()
                    if p.exitcode == 0:
                        # Nmade += 1
                        print(f"\033[1;34mStructures submitted: {counter.value:04d}/{N:04d}\033[0m")
                        print(f'\033[1;34m{pid} user time: {cpu_times.user:.3f} seconds, system time: {cpu_times.system:.3f} seconds\033[0m')
                except psutil.NoSuchProcess:
                    # El proceso ya no existe, podemos continuar
                    print(f"\033[1;33mProcess {p.pid} no longer exists.\033[0m")

            # Process running
            processes = [p for p in processes if p.exitcode is None]

        if counter.value < N:
            # Choosing a random Z value for each iteration.
            z_val = random.choice(list(Z_molecules.keys()))
            print(f"Molecular formula (Z) chosen = {z_val}")
            full_options = Z_molecules[z_val]["options"]
            # print("full_options:", full_options)
            z_weighted_options = Z_molecules[z_val]["weighted_options"]
            z_sg_selected = list(z_weighted_options.keys())
            z_sg_chose = random.choice(z_sg_selected)
            print(f"SpaceGroup ind chosen = {z_sg_chose}")

            # Modifying arguments
            arg_dict["sg_ind"] = z_sg_chose
            arg_dict["sg"] = SpaceGroups[z_sg_chose]
            arg_dict["Z_val"] = z_val
            arg_dict["V_dist"] = Z_molecules[z_val]["V_dist"]
            arg_dict["Rot_dicts"] = Z_molecules[z_val]["ROT_dict"]
            full_mols = Z_molecules[z_val]["MOLS"]
            # print(f"Picking perm for sg={z_sg_chose} from stratified options.")
            new_perm = pick_option(full_options[z_sg_chose])
            perm = convert_new_SA_to_old(
                new_perm,
                SpaceGroups[z_sg_chose],
                placement_table,
                full_mols
            )
            # print("perm", perm)
            perm = [-1, perm]  # modification as now expects [dof, perm]
            arg_dict["mols"] = full_mols

            # make test
            if args["test"]:
                p = Process(
                    target=make_test,
                    args=(perm, fname, arg_dict, lock, counter, result_queue)
                )
            else:
                p = Process(
                    target=structure_generator,
                    args=(perm, fname, arg_dict, lock, counter, result_queue)
                )

            if len(processes) < N - counter.value:
                p.start()
                processes.append(p)
                print(f"{colors.fg.green}Sent process -> ({len(processes)}){colors.reset}")

        print(f"{colors.bold}{colors.fg.red}N Structures generated: {counter.value}{colors.reset}")
        memory_info = checker_process.memory_info()
        print(f"\033[95mChecker process memory usage: RSS={memory_info.rss / (1024 * 1024):.2f} MB, VMS={memory_info.vms / (1024 * 1024):.2f} MB\033[0m")
        print(f"{colors.bold}{colors.fg.purple}------------------------------------{colors.reset}")

    # results = []
    for process in processes:
        process.join()

    stop_event.set()
    checker.join()

    log_memory_usage("Main process end")
    print('\nAll computations are done')
    print(f"\t\033[1;32mStructures finished: {counter.value:04d}/{N:04d}\033[0m")

    t_f_total = time.time()
    execution_time = t_f_total - t_i_total
    print("\tElapsed time done in %.3f s" % execution_time)


if __name__ == '__main__':
    main()
