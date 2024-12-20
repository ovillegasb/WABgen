
"""Submodule recording main functions."""

import re
import os
import string
import copy
import math
import random
import psutil
import sys
import time
import multiprocessing
import numpy as np
from ase import Atoms
from func_timeout import FunctionTimedOut
from pymatgen.symmetry.analyzer import PointGroupAnalyzer as PGA
from pymatgen.symmetry.groups import PointGroup as PymatPointGroup
from pymatgen.core.structure import IMolecule
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from wabgen.utils.data import get_atomic_data
from wabgen.utils.align import parallel, calc_R1, mol2site2
from wabgen.utils.cell import UnitCell, check_all_minseps
from wabgen.utils.data import read_db, write_db
from wabgen.utils.group import generate_group, order_of_R
from wabgen.utils import symm
from wabgen.utils.makecell import make_cell
from wabgen.utils.radialpp_bfgs2 import push_apart as flex_push_apart
from wabgen.utils.filter import test_mof_structure
import wabgen.io


# find the directory
directory = os.path.dirname(__file__)


HMPointGroupSymbols = [
    "",
    "1",      # 1
    "-1",     # 2
    "2",      # 3
    "m",      # 4
    "2/m",    # 5
    "222",    # 6
    "mm2",    # 7
    "mmm",    # 8
    "4",      # 9
    "-4",     # 10
    "4/m",    # 11
    "422",    # 12
    "4mm",    # 13
    "-42m",   # 14
    "4/mmm",  # 15
    "3",      # 16
    "-3",     # 17
    "32",     # 18
    "3m",     # 19
    "-3m",    # 20
    "6",      # 21
    "-6",     # 22
    "6/m",    # 23
    "622",    # 24
    "6mm",    # 25
    "-6m2",   # 26
    "6/mmm",  # 27
    "23",     # 28
    "m-3",    # 29
    "432",    # 30
    "-43m",   # 31
    "m-3m"    # 32
]
"""
"D5h",  #33
"C12v", #34
"D12h"  #35
"C_v",  #36
"D_h"]  #37
"""

metals = [
   'Li', 'Be',
   'Na', 'Mg', 'Al',
   'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge',
   'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
   'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
   'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
   'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
   'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]


class GEOM_constraint:
    """Class representing a point group selected to constrain a site."""

    def __init__(self, d=1.0):
        self.d = d

    @property
    def skeleton_X(self):
        d = self.d
        return {
         "D*h": Atoms(
             "XX",
             positions=[(-d/8, 0, 0), (d/8, 0, 0)]
         ),
         "D4h": Atoms(
             "XXXX",
             positions=[(d, 0, 0), (0, d, 0), (-d, 0, 0), (0, -d, 0)]
         ),
         "D2h": Atoms(
             "XXXX",
             positions=[
                 (d*0.86602540, d*-0.5, 0),
                 (d*0.86602540, d*0.5, 0),
                 (d*-0.86602540, d*-0.5, 0),
                 (d*-0.86602540, d*0.5, 0)
             ]
         ),
         "D3h": Atoms(
             "XXX",
             positions=[
                 (0, d, 0),
                 (0, -d*np.sin(np.deg2rad(30.0)), -d*np.cos(np.deg2rad(30.0))),
                 (0, -d*np.sin(np.deg2rad(30.0)), d*np.cos(np.deg2rad(30.0)))
             ]
         ),
         "Td": Atoms(
             "HHHH",
             positions=[
                 (0, 0, d),
                 (0, d*-0.94280904, d*-0.33333333),
                 (d*0.81649658, d*0.47140452, d*-0.33333333),
                 (d*-0.81649658, d*0.47140452, d*-0.33333333)
             ]
         ),
         "D2d***": Atoms(#MODIFICAR TODO
             "XXXX",
             positions=[
                 (d*-0.17022212, d*0.09827779, d*0.98049269),
                 (d*0.17022212, d*-0.89165810, d*-0.41948808),
                 (d*0.85730963, d*0.29841237, d*-0.41948808),
                 (d*-0.85730963, d*0.49496795, d*-0.14151652),
             ]
         ),
         "Oh": Atoms(
             "XXXXXX",
             positions=[
                 (0, 0, d),
                 (0, d, 0),
                 (d, 0, 0),
                 (0, 0, -d),
                 (0, -d, 0),
                 (-d, 0, 0)
             ]
         )
        }

    def set_point_group(self, pg):

        return self.skeleton_X[pg]


def get_cpu_num():
    """Return cpu number."""
    pid = os.getpid()
    p = psutil.Process(pid)
    return p.cpu_num()


def update_structure_list(old_dict, new_dict):
    """Return structure listm useful when the code is running in parallel."""
    for hash_comp in new_dict:
        if hash_comp not in old_dict:
            old_dict[hash_comp] = list()

        old_dict[hash_comp] += new_dict[hash_comp]

    return old_dict


def log_memory_usage(prefix):
    """Log memory used."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    def bytes_to_mb(size_bytes):
        """Convert bytes to megabytes."""
        return round(size_bytes / (1024 * 1024), 2)

    rss = bytes_to_mb(mem_info.rss)
    vms = bytes_to_mb(mem_info.vms)
    sys_mem = psutil.virtual_memory()
    total_mem = bytes_to_mb(sys_mem.total)
    used_mem = bytes_to_mb(sys_mem.used)
    available_mem = bytes_to_mb(sys_mem.available)

    print(f"{prefix} - Memory usage: RSS={rss} MB, VMS={vms} MB")
    print(f"{prefix} - System Memory: Total={total_mem} MB, Used={used_mem} MB, Available={available_mem} MB")


def print_blue(text):
    """Print message in blue."""
    print(f"\033[1;33m{text}\033[0m")


def read_placement_table(fname):
    """Read in the placement table, if it doesn't exist generate it."""
    path = fname
    if os.path.exists(path):
        f = open(fname, "r")
        text = f.read()
        text = text.splitlines()

        PT = []
        PT.append([x for x in text[0].split()])
        for line in text[1:]:
            l = line.split()
            row = [str(l[0])]
            ns = l[1:]
            for n in ns:
                row.append(int(n))
            PT.append(row)
        f.close()

    # else:
    #     print("generating placement table")
    #     PT = gen_base_PT(fname)

    return PT


PT_name = os.path.join(directory, "data/Placement_table.txt")
placement_table = read_placement_table(PT_name)


def add_hash(fname, N):
    """Add hashs."""
    fname += "_"
    hash_Chars = [str(x) for x in range(0, 10)] + [x for x in string.ascii_lowercase] + [x for x in string.ascii_uppercase]

    for i in range(0, N):
        fname += hash_Chars[random.randint(0, 61)]

    return fname


def generate_structure_test():
    """Generate a structure test using pymatgen."""
    STRUCT = [
      Structure.from_spacegroup(
        "P1", lattice=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), species=["Li1"], coords=[[0, 0, 0]]
      ),
      Structure.from_spacegroup(
        "P1", lattice=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), species=["Li1"], coords=[[0, 0, 0]]
      )
    ]

    return random.choice(STRUCT)


def check_min_seps(cell, min_seps=None, tol=0, modify_tags=False):
    """Check min separations, default of 1 angstrom."""
    print("check_min_seps")
    if modify_tags:
        for i, at in enumerate(cell.atoms):
            cell.atoms[i].tag = i

    if min_seps is None:
        els = list(set([at.label for at in cell.atoms]))
        D = {}
        for e1 in els:
            D[e1] = {}
            for e2 in els:
                D[e1][e2] = 1  # defaults 1 angs.
    # parsed minseps
    else:
        els = min_seps[0]
        D = min_seps[1]

    print("min_seps are", D)
    # Check atoms
    for i, at1 in enumerate(cell.atoms):
        if at1.label == "U":
            continue
        elif at1.label == "X":
            continue
        print("i:", i, "atom:", at1, "label:", at1.label)

        for j, at2 in enumerate(cell.atoms[i:]):
            if at2.label == "U":
                continue
            elif at1.label == "X":
                continue

            if at1.tag == "ignore" and at2.tag == "ignore":
                continue

            print("j:", j, "atom:", at2, "label:", at2.label)
            if (at1.tag == at2.tag and at1.key == at2.key)  or j == 0 :
                # if i == j:
                # don't use the shift vector of [0,0,0]
                # check second closest image
                rs = cell.ed_same_mol(at1.fracCoords, at2.fracCoords, all_pos=True)
                rs = sorted(rs, key=lambda x: x[0])
                d = rs[1][0]
            else:
                d = cell.ed2(at1.fracCoords, at2.fracCoords)
                # d = cell.euclidian_distance(at1.fracCoords,at2.fracCoords)

            if d < D[at1.label][at2.label]-tol:
                print("failed minseps with", at1.label, at2.label, d)
                print(at1.key, at2.key)
                """
                print(cell.lengths, 180/np.pi*np.array(cell.angles))
                print(cell.cartBasis)
                """
                return False

        print("passed minseps")
    return True


def make_test(dof_perm, fname, arg_dict, lock=None, counter=0, result_queue=None):
    """Test main boucle."""
    print(60 * "#")
    print_blue(f"Running main function from WAMgen - building structure ... {counter.value}")
    time.sleep(1)
    cpu = get_cpu_num()
    status = random.choice([True, False])
    if status:
        print(f'\033[1;37m{fname} exiting with error code 1\033[0m')
        sys.exit(1)  # Exit with error code 1

    struct = generate_structure_test()
    print(struct)

    response_q = multiprocessing.Manager().Queue()
    result_queue.put((cpu, struct, response_q))
    is_duplicate = response_q.get()
    if is_duplicate:
        print("Pues resulta que si lo es")

    if lock is not None:
        with lock:
            counter.value += 1

    print('\033[1;32mfinished successfully\033[0m')
    print(60 * "#")

    return 0


def structure_generator(dof_perm, fname, arg_dict, lock=None, counter=0, result_queue=None):
    """Build structures."""
    print(60 * "#")
    print_blue("Running main function from WAMgen - building structure ...")
    # 0. conver the arguments to a useable form
    # dof = dof_perm[0]
    perm = dof_perm[1]

    n_tries = arg_dict["n_tries"]
    sg_ind = arg_dict["sg_ind"]
    target_atom_nums = arg_dict["target_atom_nums"]
    # form = arg_dict["form"]
    V_dist = arg_dict["V_dist"]
    cell_abc = arg_dict["cell_abc"]
    mols = arg_dict["mols"]
    min_seps = arg_dict["min_seps"]
    # gulp_ps = arg_dict["gps"]
    pressure = arg_dict["pressure"]
    noise = arg_dict["minsep_noise"]
    # aenet_relax = arg_dict["aenet_relax"]
    Rot_dicts = arg_dict["Rot_dicts"]
    Z_val = arg_dict["Z_val"]
    push_apart = arg_dict["push_apart"]

    np.random.seed(None)

    # Where is the process
    cpu = get_cpu_num()
    print_blue(f"Process is running on CPU {cpu}")
    log_memory_usage(f"{cpu} start")

    sg = arg_dict["sg"]
    # print_blue(f"Spacegroup: {sg.name}")
    # print("sg_ind=", sg_ind, "perm=", perm)

    # check if only have atoms
    only_atoms = True
    for mol in mols:
        if mol.Otype == "Mol":
            only_atoms = False
    if only_atoms:
        print_blue("System with only atoms")

    # 0. add some random noise to the minseps
    dic = min_seps[1]
    for el1, d in dic.items():
        for el2, d2 in d.items():
            dic[el1][el2] += random.uniform(-noise, noise)
            dic[el2][el1] = dic[el1][el2]
    min_seps[1] = dic
    # print("min_seps:", min_seps)

    # 1. make n_tries attempt to make the permutation
    n_SG = 2    # fail count for making a supergroup
    n_SC = 2    # fail count for making a supercell
    n_vol = 15  # total times to fail because volume is too large
    n_tries = arg_dict["n_tries"]
    accept = False
    check_distance = False
    N_supercells = 0
    N_supergroup = 0
    vols = []
    n_try = 0
    P = 0.0
    print_blue(f"N tries: {n_tries}")
    print_blue(f"Push apart: {push_apart}")
    print_blue(f"Z value: {Z_val}")
    add_centre = False

    while not accept and n_try < n_tries and len(vols) < n_vol:
        n_try += 1

        if not push_apart:
            print("NOT PUSHING APART")
            #TODO what doesn't work hore?
            #NOTE sometimes this doesn't seem to work...
            # print("V_dist is", V_dist)
            # print("cell_abc is", cell_abc)
            all_details, cell = make_cell(
               mols,
               sg_ind,
               perm,
               V_dist,
               cell_abc,
               Rot_dicts,
               add_centre
            )
            # print("cell:", cell)
            # print("cell:", dir(cell))
            # print("cell atoms:", cell.atoms)
            # print(all_details)
            if only_atoms:
                accept = check_all_minseps(cell, min_seps=arg_dict["min_seps"])
            else:
                accept = check_min_seps(cell, arg_dict["min_seps"])

            # print("min sep accepted:", accept)
            if not accept:
                continue
            H = 0

        elif push_apart == "flexible":
            overlap_accept = False
            noverlap = 0
            overlap_max = 30

            # hack for volume distribution
            # pick a volume then increase it by 1% each failed overlap attemp
            # use uniform distribution to specify volume exactly
            func, args, kwargs = wabgen.io.line_to_func(V_dist)
            args = [float(x) for x in args]
            for key, item in kwargs.items():
                kwargs[key] = float(item)

            Vtarg = func(*args, **kwargs)
            while not overlap_accept:
                # gradually increase the volume estimate while struggling to make a non-overlapping
                # guess cell. useful for bad volume estimate and saving time!
                Vtarg *= 1.01
                vd = "numpy.random.uniform " + str(Vtarg) + " " + str(Vtarg*1.001)
                all_details, cell = make_cell(mols, sg_ind, perm, vd, cell_abc, Rot_dicts, add_centre)
                ls = set([at.label for at in cell.atoms])
                if "U" in ls:
                    print("U in cell")
                    exit()
                if cell_abc is None:
                    overlap_accept = overlap_check(cell, all_details)
                else:
                    overlap_accept = True
                noverlap += 1
                if noverlap >= overlap_max:
                    print("noverlap is", noverlap)
                    continue

            sg = arg_dict["sg"]
            try:
                accept, cell = flex_push_apart(
                  cell, sg, min_seps, P=pressure, target_atom_nums=target_atom_nums
                )
            except ZeroDivisionError:
                accept = False

            # print("accept and cell are", accept, cell)
            if not accept:
                continue
            cell.sg_num = sg.number
            H = cell.vol/len(cell.atoms)

        # if here then have a good cell and write it out to file
        # create a file name
        f_name = fname + "_" + str(cell.sg_num)
        sg_name = str(sg.name)
        sg_name = re.sub(r"/", "_", sg_name)
        f_name += "_"+sg_name
        f_name += f"_Z_{Z_val}"
        f_name = add_hash(f_name, 8)
        print_blue("system name: {}".format(f_name.split("/")[-1]))

        # Extract ASE atoms
        #>>>>>>ase_struct = cell.get_ase_struct()

        if not only_atoms and check_distance:
            ase_struct = cell.get_ase_struct()
            # Testing filter distances
            print("Checking distances...")
            metal_center = set()
            for at in cell.atoms:
                if at.label in metals:
                    metal_center.add(at.label)

            metal_center = "".join(list(metal_center))

            try:
                rejected = test_mof_structure(
                   ase_struct,
                   metal_center,
                   radius=5.00,
                   cutoff=3.28,
                   dist_min=1.90,
                )
            except FunctionTimedOut:
                print("Function Timed Out. The file will be rejected")
                rejected = True

            if rejected:
                accept = False
                continue

        #>>>>>>># Testing duplicates
        #>>>>>>>print("Checking duplicates...")
        #>>>>>>>pymatgen_struct = AseAtomsAdaptor.get_structure(ase_struct)
        #>>>>>>>response_q = multiprocessing.Manager().Queue()
        #>>>>>>>N_replicates_q = multiprocessing.Manager().Queue()
        #>>>>>>>result_queue.put((cpu, pymatgen_struct, response_q, N_replicates_q))
        #>>>>>>>is_duplicate = response_q.get()
        #>>>>>>>N_duplicates = N_replicates_q.get()
        #>>>>>>>if is_duplicate:
        #>>>>>>>    print(f"\033[95mStructure duplicated - N={N_duplicates}\033[0m")
        #>>>>>>>    accept = False
        #>>>>>>>    continue
        #>>>>>>>else:
        #>>>>>>>    print(f"\033[95mStructure generated is not a duplicate - N={N_duplicates}\033[0m")

        # H should always be define, often 0
        print_blue(f"writing res with P={P} and E={H}")
        wabgen.io.write_res(f_name, cell, E=H, P=P)

        # move completed file to completed
        cmd = "mv " + f_name + "* ./completed/"
        os.system(cmd)

        with lock:
            counter.value += 1

        print('\033[1;32mfinished successfully\033[0m')
        print(60 * "#")
        log_memory_usage(f"{cpu} end")

        return 0

    # if here then failed to make the allocation
    print(f"failed to make the allocation tried {n_try} times from max allowed of {n_tries}" )       #TODO look at this bit!
    # print("perm is", perm)
    if N_supercells >= n_SC and arg_dict["no_supercells"]:
        f_name = "rejected because supercell " + str(sg_ind)

    elif N_supergroup >= n_SG and arg_dict["exact_sg"]:
        f_name = "rejected because supergroup " + str(sg_ind)

    elif len(vols) >= n_vol:
        f_name = "rejected because of volume:"
        # vav = sum(vols)/len(vols)
        v_best = sorted(vols, key=lambda x: x)[0]
        f_name += " V_best = " + str(v_best)
        f_name += " distribution = " + str(V_dist[0]) + " " + str(V_dist[1])
    else:
        f_name = "couldn't make with " + str(n_tries) + " attempts " + str(sg_ind)

    print("WABgen has failed to find convergence for this structure")
    sys.exit(1)


def tol_sort(fops, tol, ind=0):
    """Sorts based on tolerance."""
    if len(fops[1]) == 1:
        # only have original indicies, so lists are indistinquishable, return inds in current order
        return [fop[0] for fop in fops]

    else:
        # still have something to sort on
        # sort based on first element into differnet lists
        el0s = []
        for fop in fops:
            found = False
            for cat in el0s:
                if abs(fop[1][ind]-cat[0][1][ind]) < tol:
                    cat.append(fop)
                    found = True
                    continue
            if not found:
                el0s.append([fop])

        sortd = sorted(el0s, key=lambda x: x[0][1][ind])

        for i, cat in enumerate(sortd):
            if len(cat) > 1:
                sortd[i] = tol_sort(cat, tol, ind+1)
        flat = [x for y in sortd for x in y]
        return flat


def Rnz(n):
    """Rotate around the Z-axis."""
    a = [np.cos(2*np.pi/n), -np.sin(2*np.pi/n), 0]
    b = [np.sin(2*np.pi/n), np.cos(2*np.pi/n), 0]
    c = [0, 0, 1]
    Rnz = np.array([a, b, c])
    return Rnz


def sort_matrices(ops, tol=10**-3):
    """Sort the operators, element differing by less than tol taken to be equal."""
    if len(ops) == 1:
        return ops
    fops = []
    for i, op in enumerate(ops):
        fop = [x for y in op for x in y]
        fops.append([i, fop])
    sfops = tol_sort(fops, tol)

    final = [ops[x[0]] for x in sfops]
    return final


def order_of_op(R):
    """Return the order of a rotation element."""
    # print("R is", R)
    n = 0
    same = False
    while not same:
        n += 1
        Rn = np.linalg.matrix_power(R, n)
        Rnf = Rn.flatten()
        tol = 10**-6
        Id = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        dif = Rnf - Id
        same = True
        for el in dif:
            if abs(el) >= tol:
                same = False
        if n > 32:
            # print("order of element has gone wrong")
            # print(R)
            # print(np.linalg.det(R))
            n = order_of_R(R)
            if n is not None:
                return n
            # print("order of element has gone wrong")
            exit()

    return n


def add_template_molecule(temp_fname, Mol):
    """Add this molcule to the template_molcules_file."""
    # 1. read in the template file
    out = wabgen.io.general_castep_parse(temp_fname, ignoreComments=False)
    # print("out is", out)

    # 2. add the molecule to the positions_abs block
    name = Mol.point_group_sch
    list_name = ["#!", "group", name]
    for i, c in enumerate(Mol.coords):
        lst = [Mol.species[i]] + [x for x in c]
        lst += list_name
        out["positions_abs"].append(lst)

    # 3. write the template_molecules file out again
    with open(temp_fname, "w") as f:
        for key in out:
            l0 = "%block " + key
            lf = "%endblock " + key
            f.write(l0 + "\n")
            for line in out[key]:
                s = ""
                for x in line:
                    s += str(x) + "\t"
                s += "\n"
                f.write(s)
            f.write(lf+"\n")
            f.write("\n")


def standardize_Molecule(Mol, temp_fname, st=1e-3):
    """
    Standardize the molecules orientation, relative to template molecules.

    Uses as template if template not stored already.

    test that the ops always come out inidentical order after standardization
    05.07.2018, tested ops of template and mol always identical after alignment, works well
    """
    # 0. if atom return as is
    if Mol.Otype == "Atom":
        return Mol

    # 1. check if a template with the same point group exists already
    params = wabgen.io.parse_file(temp_fname, st, template=True)
    mols = params["mols"]
    tmol = None
    for mol in mols:
        if mol.name == Mol.point_group_sch:
            tmol = mol
            break

    if tmol is None:
        # no template molecule found so store this molecule as the standard
        print(Mol.name, ":didn't find template molecule for this point group so adding...")
        SMol = Mol

        # now need to loop over all spacegroups and work out the rotational form of the permutations
        # for every perm : yes eurgh its hideous to find them but should be neatish and useful onece
        # its all done!!

        # NOTE no longer using rot_perms so commneted out generation on 17.10.2022
        # generate_rotation_perms(SMol, SpaceGroups, PT)
        add_template_molecule(temp_fname, Mol)

    else:
        # found a template molecule, rotate this molecule to align with the ops of tmol
        print(Mol.name, Mol.point_group_sch, ":found a template molecule, aligning...")
        Ms = mol2site2(Mol, tmol)

        # 1. can pick any of them Ms, doesn't matter
        # print("Ms are", Ms)
        M = Ms[0]

        # for C1
        if M is None:
            M = np.identity(3)
        # if there was only one eigenvector to align then M will be [R,eig] not just R
        elif len(M) == 2:
            M = M[0]

        species = [s for s in Mol.species]
        coords = [np.dot(M, c) for c in Mol.coords]
        SMol = Molecule(species, coords, number=Mol.number, name=Mol.name)

    print("finished alignment, returning Smol")
    return SMol


class Operator:
   """operator type, format, axial eigenvector, ses"""
   def __init__(self, index, R, t=(0,0,0)):
         self.matrix = R
         self.t = np.array(t)
         self.op_symbol = Operator.set_op_symbol(self)
         self.set_eig()
         self.index = index

   def set_op_symbol(self):
      # determines the operators symbol from its order and determinent
      # all + for rotations, - for roto-inversions
      M = self.matrix

      det = np.linalg.det(M)
      Inv = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
      if det < 0:
         M = np.dot(M, Inv)
      return int(round(det * order_of_op(M)))

   def set_eig(self):
      #finds the axial eigenvector of operator
      #if the element is 1 or -1 return [1,0,0]
      if self.op_symbol == 1 or self.op_symbol == -1:

         self.eig = np.array([1, 0, 0])

      if self.op_symbol < 0:
         lamb = -1
      elif self.op_symbol > 0:
         lamb = 1

      R = self.matrix
      l, M = np.linalg.eig(R)
      tol = 0.0001
      for i, l2 in enumerate(l):
         if lamb - tol <= l2 <= lamb + tol:
            eig = np.real(M.T[i])
      #take the real part to prevent potential problems later
      #otherwise it give things like 1+0.j which is a bit odd
      self.eig = eig


def apply_op(op,fp):
   #applys fractional operators to coordinate
   R = op.matrix
   t = op.t
   fp2 = np.dot(R,np.array(fp)) + np.array(t)
   fp3 = wrap_coords(fp2)
   return fp3


def wrap_coords(fp, tol = 10**-6):
   fp2 = [x%1 for x in fp]
   for i in range(0,3):
      if fp2[i] > 1-tol:
         fp2[i] =0

   return fp2


class Molecule:
   """NOT the pymatgen class, modified to be more useful"""
   def __init__(self, species, coords, number=1, Otype="Mol", name=None, st=1e-03, std=True, constraint=None):
      # how many of the Molecule there are
      self.number = number
      self.Otype = Otype

      # contraint
      self.constraint = constraint

      if Otype=="Mol":
         self.std = std
         self.name = name
         self.species = species
         self.coords = coords
         #self.coords = [x for x in coords]
         self.CoM_shift()
         self.set_radius()
         """
         print("mol is")
         for s,c in zip(self.species, self.coords):
            print(s,c)
         """
         self.C_operators = Molecule.get_ops(self, st)
         self.fp = Molecule.set_fp(self,self.C_operators)
         self.unique_eigs = Molecule.set_unique_eigs(self,self.C_operators)
         self.set_Rnorm()
         self.gfp = Molecule.set_gfp(self,self.C_operators, Rnorm=self.Rnorm)
         #gfp2 = Molecule.set_gfp_test(self,self.C_operators, Rnorm=self.Rnorm)
         #print("len of self.Rnorm is", len(self.Rnorm))

         self.symbol = Symbol(self.fp, self)
         #print(self.fp)
         """
         #testing methods
         s1 = str(self.gfp);     s2 = str(gfp2)
         if s1 != s2:
            print(self.symbol.HM)
            pp.pprint(self.gfp)
            pp.pprint(gfp2)
         """
         self.eig_dict = Molecule.set_eig_dict(self, self.C_operators)


         #self.print_symm_info()

      elif Otype=="Atom":
         self.species = species
         self.coords = [0,0,0]
         self.name = species[0]
         self.rmax = 0.1

   def set_eig_dict(self, C_operators):
      """ returns dict. keys are op indexes, items are list of len 2
      of consistently sorted operator eigenvalues"""
      d = {}
      for op in C_operators:
         #ignore 1 and -1 as they have no meaningful direction
         if op.op_symbol in [1,-1]:
            continue

         temp = [[0,np.array(op.eig)],[1, -1*np.array(op.eig)]]
         eigs = [x[1] for x in tol_sort(temp, tol=10**-3)]
         d[op.index] = eigs
      return d



   def set_Rnorm(self):
      """creates list of all cartesian rotation operators"""
      Rs = []

      #add all the rotation operators
      for op in self.C_operators:
         if op.op_symbol > 0.5:
            Rs.append(op.matrix)


      if self.point_group_sch not in ["C_v", "D_h"]:
         self.Rnorm = [Operator(i,R) for i,R in enumerate(Rs)]
         return 0

      #try to read in the Rnorm, always add the Rnorm in the same direction
      #and only used after standardized so this shouln't matter
      fname = directory+"/data/infinite_Rnorms/"+self.point_group_sch+".txt"
      fz = fname + ".gz"
      if not os.path.isfile(fz):
         #generate the normaliser and write it out
         R24 = Rnz(24)

         for i in range(0,24):
            if i % 2 > 0:
               Rs.append(np.linalg.matrix_power(R24,i))

         Rs = generate_group(Rs)
         Rs = [[[x for x in y] for y in R] for R in Rs]
         write_db(Rs, fname)

      else:
         #read Rnorm in from the file
         Rs = read_db(fz)
      self.Rnorm = [Operator(i,R) for i,R in enumerate(Rs)]

   def CoM_shift(self):
      at_info = get_atomic_data()
      RM = np.array([0.0, 0.0, 0.0])
      M = 0
      for i, lab in enumerate(self.species):
         m = at_info[lab]["Mr"]
         r = np.array(self.coords[i])
         RM += m*r
         M += m
      R = RM/M

      #now shift all of the positions by R
      for i, x in enumerate(self.coords):
         self.coords[i] = np.array(x)-R
      self.cart_CoM = R

   def set_radius(self):
      rmax = 0
      for v in self.coords:
         r = np.linalg.norm(v)
         if r > rmax:
            rmax = r
      self.rmax = rmax

   def get_ops_from_template(self):
      """use a template molecule as the starting point for the
      operators of molecules with continuous degrees of freedom"""
      n = 12
      species = []
      coords = []
      r0 = [2.5,0,0]

      #create base template of 12 Sulfur atoms in a ring
      R12 = Rnz(12)
      for i in range(0,n):
         Rn = np.linalg.matrix_power(R12,i)
         species.append("S")
         coords.append(np.dot(Rn,r0))

      #depending on symbol in question append the next two atoms
      coords +=  [np.array([0,0,2]),np.array([0,0,-2])]

      if self.point_group_sch == "D_h":
         species += ["C","C"]
      if self.point_group_sch == "C_v":
         species += ["C","N"]

      #now set the template mol operators
      self.template_mol = Molecule(species, coords)

      #detect the axis of the molecule and rotate it to be parallel to z
      #can only be linear if Dinfh or Cinfv is found!
      r = np.array(self.coords[1])-np.array(self.coords[0])
      #find the centre of mass of the molecule and centre on the centre of mass

      R1 = calc_R1(r, [0, 0, 1])
      new_coords = [np.dot(R1,x) for x in self.coords]
      self.coords = new_coords

      return [Operator(i,op.matrix) for i,op in enumerate(self.template_mol.C_operators)]


   def print_symm_info(self):
      """Print symmetry infomation."""
      if self.Otype == "Mol":
         print("\nsymmetry info is:")
         print("name is", self.name)
         # if "point_group_sch" in self.__dict__.keys():
         print("sch symbol is ", self.point_group_sch)
         print("fingerprints are", self.fp, self.gfp)
         print("HM symbol is", self.symbol.HM)
         print("mol.symbol.ind", self.symbol.ind)

   def get_ops(self, st):
      #set operations to an instance of the Operators clas
      pmgmol = IMolecule(self.species,self.coords)
      pmgPG = PGA(pmgmol, tolerance=st)
      self.point_group_sch = re.sub(r"\*", "_", pmgPG.sch_symbol)
      #print(self.point_group_sch)


      if self.point_group_sch == "C1":
         return [Operator(0, np.identity(3))]

      #support for non-crystallographic ops
      if self.point_group_sch == "D_h":
         return Molecule.get_ops_from_template(self)
      if self.point_group_sch == "C_v":
         return Molecule.get_ops_from_template(self)

      pmg_symm_ops = pmgPG.get_symmetry_operations()
      #symm_ops = PGO(pmgPG.sch_symbol, pmgPG.symmops)
      #assert len(pmg_symm_ops) == len(symm_ops)

      ops = [x.affine_matrix[0:3,0:3] for x in pmg_symm_ops]
      #sort ops in a consistent way to ensure always the same order, if in same orientiation
      #useful for the template molecule, ensures that the ops of the template are always consistent

      ops = sort_matrices(ops)

      Ops = []
      for i,op in enumerate(ops):
         Ops.append(Operator(i, op))
      return Ops

   def set_fp(self, ops):
      #written like this so can pass either type of ops
      fp = [x.op_symbol for x in ops]
      fp = np.sort(fp)
      sfp = ""
      for x in fp:
         sfp += str(x)
      #print("spf is", sfp)
      return sfp

   def set_gfp_test(self, ops, Rnorm=None):
      """use differnet grouping method to check they give the same answer"""
      gfp = dict()
      for op in ops:
         gfp[op.op_symbol] = []

      Rs = [x.matrix for x in ops if x.op_symbol > 0]
      if Rnorm is not None:
         Rs = [R.matrix for R in Rnorm]

      #loop over type of operator testing for S.E.O's
      for optype in gfp.keys():
         ops_list = [op for op in ops if op.op_symbol == optype]
         ops_tup = tuple([x.matrix for x in ops_list])
         tol = 0.001
         seo = []

         while len(ops_list) > 0:
            Cur_op = ops_list.pop(-1)
            equiv = [Cur_op.index]
            equiv_del = []
            eig1 = Cur_op.eig
            for j,M in enumerate(ops_list):
               eig2 = M.eig
               for R in Rs:
                  #if np.linalg.norm(np.cross(eig1,np.dot(R,eig2))) < tol:
                  Mtrans = np.dot(np.linalg.inv(R), np.dot(M.matrix, R))
                  Mtrans_inv = np.linalg.inv(Mtrans)
                  if np.allclose(Mtrans, Cur_op.matrix) or np.allclose(Mtrans_inv, Cur_op.matrix):
                  #if np.linalg.norm(np.cross(eig1,np.dot(R,eig2))) < tol:
                     equiv.append(M.index)
                     equiv_del.append(j)
                     break
            #print("equiv is",equiv_del)
            for i in range(0,len(equiv_del)):
               del ops_list[equiv_del[len(equiv_del)-1-i]]
            seo.append(equiv)

         gfp[optype] = seo

      return gfp

   def set_gfp(self,ops,Rnorm=None):

      gfp = dict()
      for op in ops:
         gfp[op.op_symbol]=[]

      Rs = [x.matrix for x in ops if x.op_symbol > 0]
      if Rnorm is not None:
         Rs = [R.matrix for R in Rnorm]
      #loop over type of operator testing for S.E.O's
      for optype in gfp.keys():
         ops_list = [op for op in ops if op.op_symbol == optype]
         ops_tup = tuple([x.matrix for x in ops_list])
         tol = 0.001
         seo = []

         while len(ops_list) > 0:
            Cur_op = ops_list.pop(-1)
            equiv = [Cur_op.index]
            equiv_del = []
            eig1 = Cur_op.eig
            for j,M in enumerate(ops_list):
               eig2 = M.eig
               for R in Rs:
                  if np.linalg.norm(np.cross(eig1,np.dot(R,eig2))) < tol:
                     equiv.append(M.index)
                     equiv_del.append(j)
                     break
            #print("equiv is",equiv_del)
            for i in range(0,len(equiv_del)):
               del ops_list[equiv_del[len(equiv_del)-1-i]]
            seo.append(equiv)

         gfp[optype] = seo

      return gfp

   def set_unique_eigs(self, ops):
      """returns a dictionary of the unique eigs for each op type"""
      odict = {}
      ##ops = self.C_operators
      for op in ops:
         #skip 1 and -1
         if op.op_symbol in [1,-1]:
            continue
         odict[op.op_symbol] = []
      for op in ops:
         if op.op_symbol in [1,-1]:
            continue
         odict[op.op_symbol].append(op)


      unique_eigs = {}
      for key in odict:
         unique_eigs[key] = []


      for key in odict:
         for op in odict[key]:
            for sign in [-1,+1]:
               new = True
               eig1 = sign*op.eig
               for eig2 in unique_eigs[key]:
                  #print(eig1, eig2)
                  if parallel(eig1, eig2):
                     new = False
                     break
               if new:
                  unique_eigs[key].append(eig1)

      return unique_eigs

   def symm_equiv_eigs(self,ops):
      """creates a dictionary of the symmetry distinct eigs for each operator"""


class Site:
   """class to hold the symmetry infomation about a site of
   interest. Might also want to store the positions and a list of
   operators needed to move between the sites"""
   """
   symbol
   international number
   """
   def __init__(self, index=None, JWSGsite=None, operators=None, C_operators=None,
      offsets=None, Nsites = None, cell=None):
      #JWSGsite is as described
      #operators is a list of instances of the Operators class
      #includes all operators for the space group in question
      #print("JWSGsite is", JWSGsite)

      if index is not None and JWSGsite is not None and operators is not None:
         self.index = index
         Site.set_letter(self, index, Nsites)
         self.offsets = offsets
         self.frac_pos = Site.str_to_list(JWSGsite[0])
         #print("fp is", self.frac_pos)
         Site.set_other_frac_pos(self,JWSGsite)
         #print("other fps are", self.other_frac_pos)
         self.mult = len(JWSGsite)*len(self.offsets)
         Site.set_rank(self)
         Site.set_operators(self,operators,C_operators)
         self.order = len(self.operators)
         self.gfp = Molecule.set_gfp(self,self.C_operators)
         self.unique_eigs = Molecule.set_unique_eigs(self,self.C_operators)
         self.fp = Molecule.set_fp(self, self.C_operators)
         self.symbol = Symbol(self.fp)
         self.Otype = "Mol"
         #NOTE: need to sort the fractional eigs then convert to cart-eigs
         #     because frac_eigs are constant, cart_eigs vary with cell!!
         self.frac_eig_dict = Molecule.set_eig_dict(self, self.operators)
         """
         print("frac_eig_dict is")
         print(self.frac_eig_dict)
         """
         self.cell = cell

   def set_letter(self, index, Nsites):
      """sets the wyckoff letter of the site"""
      lets = string.ascii_lowercase + string.ascii_uppercase
      self.letter = lets[index]

   def irrational_fp(self):
      #converts the x,y,z strings to irrational fractional coordinates
      #eval is to make sure that "x-y+z" is handled correctly
      fp = self.frac_pos
      xi = 2**-0.5
      yi = 2/np.pi
      zi = 3**-0.5
      ir = []
      for u in self.frac_pos:
         u = re.sub('x',str(xi),u)
         u = re.sub('y',str(yi),u)
         u = re.sub('z',str(zi),u)
         u = eval(u)
         u = u % 1.0
         ir.append(u)
      return ir

   def set_site_basis(self):
      """returns set of basis vectors spanning the sites subspace"""
      tol = 10**-4
      #1. find the vecs
      vs = ["x", "y", "z"]
      covecs = []
      for fp in self.frac_pos:
         covec = []
         for v in vs:
            other = [x for x in vs if x != v]
            fp1 = re.sub(v, "1",fp)
            for o in other:
               fp1 = re.sub(o, "0", fp1)
            e = eval(fp1)
            covec.append(e)
         #now compare to previous co vectors
         if np.linalg.norm(covec) < tol:
            continue
         new = True
         for cv in covecs:
            if np.linalg.norm(np.cross(cv, covec)) < tol:
               new = False
               break
         if new:
            covecs.append(covec)

      #2. covecs is now set of vetors repping the cooeficents of x,y,z for ws
      #print("for fp", fp, "covecs are", covecs)
      self.basis_vecs = covecs


   def randomised_fp(self):
      """subs random numbers in for x,y,z to create a randomised fractional position"""
      us = [random.uniform(0,1) for i in range(0,3)]
      site_fp = copy.copy(self.frac_pos)
      rand_site_fp = []

      for u in site_fp:
         for i,l in enumerate(["x","y","z"]):
            u  = re.sub(l,str(us[i]),u)
         u = eval(u)
         u = u % 1.0
         rand_site_fp.append(u)
      rand_site_fp = np.array(rand_site_fp)
      #NOTE this shold be returning a variable list
      return rand_site_fp, None

   def update_Cops(self, cell):
      # after the cell is fixed update the Cartesian operators
      C_ops = []
      for op in self.operators:
         matrix = op.matrix
         Mc = np.dot(cell.cartMat, matrix)
         MC = np.dot(Mc, cell.fracMat)
         tc = np.dot(cell.cartMat, op.t)
         C_ops.append(Operator(op.index, MC,t=tc))
      self.C_operators = C_ops

   def set_operators(self,operators,C_operators):
      #set the operators associated with the sitesymmetry
      #do by testing if the operator maps the site to itself
      #after wrapping the coords ofc!!!
      self.operators = []
      self.C_operators = []
      r1 = Site.irrational_fp(self)
      tol = 10**-5

      #NOTE modified to copying op on 05.03.2019 as it mucking up op indexes in sg
      for i,op_orig in enumerate(operators):
         op = copy.deepcopy(op_orig)
         #apply the operation, wrap coords, then test if same position
         r2 = np.dot(op.matrix,r1) + np.array(op.t)
         r2 = [u % 1.0 for u in r2]
         r3 = []
         for x in r2:
            if x > 1-tol:
               r3.append(0)
            else:
               r3.append(x)
         #print(r3)
         #dif = np.array(r1) - np.array(r3)
         #aw = np.argwhere(abs(dif) > tol)
         if np.allclose(r1,r3):
            op.index = len(self.operators)
            C_operators[i].index = op.index
            self.operators.append(op)
            self.C_operators.append(C_operators[i])

   def str_to_list(string):
      #removes assumed first and last bracket,
      #then splits with commans as delimiters
      return string[1:-1].split(",")

   def set_rank(self):
      #determine the number of degrees of freedom the site has
      #(0,x,x) has 1 etc. CHECK THIS MORE THOUGHOURLY
      free = set()
      lets = ["x", "y", "z"]
      for x in self.frac_pos:
         for l in lets:
            if l in x:
               free.add(l)
         self.rank = len(free)

   def set_other_frac_pos(self,JWSGsite):
      #set the other fractional_positions
      #TODO remove this if its not needed
      self.other_frac_pos = []
      for x in JWSGsite[1:]:
         self.other_frac_pos.append(Site.str_to_list(x))

   def init_from_symbol(self,HMsymb):
      #initialize (poorly) from a HMsymbol
      #will only set a few attributes

      PyPG = PymatPointGroup(HMsymb)
      Pyops = PyPG._symmetry_ops
      self.operators = []
      for i,x in enumerate(Pyops):
         R = x.affine_matrix[0:3,0:3]
         self.operators.append(Operator(i,R))

      self.symbol = HMsymb
      self.fp = Molecule.set_fp(self,self.operators)


class Perm:
   def __init__(self, otr, M=np.identity(3)):
      self.number =  otr["No"]
      self.str_perm = otr["transformed WP"]
      self.set_perm()
      self.symbolic = otr["symbolic"]
      self.set_op(otr, M)

   def set_perm(self):
      d = {}
      for i,l in enumerate(string.ascii_lowercase+string.ascii_uppercase):
         d[l] = i
      self.perm = [d[x] for x in self.str_perm]

   def set_op(self, otr, M):
      m = np.array(otr["matrix"])
      R = m[:3,:3]
      Mi = np.linalg.inv(M)
      R = np.dot(M, np.dot(R,Mi))
      t = m[:,3]
      t = np.dot(M,t)
      #1. but, ops on bilbao tabulated other way around to expected way!!
      #so now find the inverse of this operator
      Rp = np.linalg.inv(R)
      tp = np.dot(Rp, -t)

      self.op = Operator(self.number, Rp,t=tp)


class SG:
    """Similar class to Jw's for Spacegroups."""
    def __init__(self, JWSG, cell=None):
        """Initialize from one of Jamies space group objects."""
        self.number = JWSG["number"]
        self.system = JWSG["system"]
        self.name = JWSG["name"]
        self.M = JWSG["M"]   # jpd47 added this, used for reduction to primitive
        # read in thr passed random cell and enforce symmetry
        self.cell = cell
        if cell is not None:
            symm.enforce_symmetry(self, self.cell)
        self.offsets = JWSG["offsets"]
        SG.set_ops(self, JWSG)
        SG.set_sites(self, JWSG)
        SG.read_normaliser(self)

    def set_signature_dict(self, sd):
        self.signature_dict = sd

    def read_normaliser(self):
        # read in the permutations of wyckoff sites from the affine normaliser
        if self.number in [229, 230]:
            self.norm_perms = [[i for i in range(0, len(self.sites))]]
        else:
            # read in the affine normaliser from data/norm_number.txt.gz
            fname = directory+"/data/normalisers/norm_"+str(self.number)+".txt.gz"
            norm = read_db(fname)
            self.norm_perms = [x for x in norm["permutations"]]
            self.wsets = [x for x in norm["wsets"]]
            # read in ops_table
            self.perms = []
            for otr in norm["Op_table"]:
                self.perms.append(Perm(otr, self.M))
            # print("self.perms is", self.perms[-1].__dict__, self.perms[-1].op.__dict__)

    def set_C_operators(self, JWSG):
        self.C_operators = []
        for i, op in enumerate(self.operators):
            M = np.dot(np.dot(self.cell.cartMat, op.matrix), self.cell.fracMat)
            t = np.dot(self.cell.cartMat, op.t)
            self.C_operators.append(Operator(i, M, t))

    def irational_cell(self, JWSG):
        # creates a cell with irrational lengths and angles
        lengths = [2**1.5, 3**1.5, 5**0.75]
        angles = [np.pi*30,63*2**0.5,49*3**0.5]
        self.cell = UnitCell(angles=angles, lengths=lengths)
        symm.enforce_symmetry(self, self.cell)

    def set_ops(self, JWSG):
        # initalises self.ops to be a list of instances of
        # the Operators class
        self.operators = []
        for i, op in enumerate(JWSG["operators"]):
            M = op[0]
            t = op[1]
            self.operators.append(Operator(i, M, t))
        if self.cell is None:
            SG.irational_cell(self, JWSG)
        SG.set_C_operators(self, JWSG)
        self.rank = len(self.operators)

    def set_sites(self, JWSG):
        # set self.sites to be a list of instances of
        # the Site class
        self.sites = []
        N = len(JWSG["wyckoff"])
        for i, site in enumerate(reversed(JWSG["wyckoff"])):
            self.sites.append(
               Site(i, site, self.operators, self.C_operators, self.offsets, cell=self.cell, Nsites=N)
            )


class Symbol:
   #idea is it has symbols, fingerprints and indicies
   def __init__(self,info,mol=None):
      #ideally could be assigned from any input
      #will assume that PGRef exists, will be created by this
      if type(info) == int:
         self.ind = info
         self.HM = HMPointGroupSymbols[self.ind]
         s = Site()
         s.init_from_symbol(self.HM)
         self.fp = s.fp

      elif type(info)==str:
         done = False
         #could have symbol or fingerprint
         for x in PGref:
            if info == x.fp:
               self.ind = x.ind
               self.HM = x.HM
               self.fp = info
               done = True
               break
         if done == False:
            for x in PGref:
               if info == x.HM:
                  self.HM = info
                  self.ind = x.ind
                  self.fp = x.fp
                  done = True
                  break

         if not done:
            #print("couldn't find a match!!")
            #print("was parsed the info:", info)
            done = self.fromPT(mol)
            if not done:
               #if here then still not done so need to add to placement table
               add2PT(mol, placement_table, PT_name)
               done = self.fromPT(mol)
               if not done:
                  print("something fucked up")
                  exit()

   def fromPT(self,mol):
      """init from PT"""
      syms = [x[0] for x in placement_table]
      for i,x in enumerate(syms):
         if x == mol.point_group_sch:
            self.HM = x
            self.ind = i
            self.fp = mol.fp
            #print("found non-CPG in PT")
            return True
      return False


# Create the points group reference table
PGref = [Symbol(i) for i in range(1, 33)]


def weight_spacegroups(sg_selected, SpaceGroups, filename=None, weighting="ranks"):
    """Return various weighting schemes OR simply read from a file."""
    if filename is None:
        if weighting == "ICSD":
            filename = directory + "data/ICSD_spacegroup_frequencies.txt"
        elif weighting == "CSD":
            filename = directory + "data/CSD_spacegroup_frequencies.txt"

    # get baseline frequencies from somewhere
    if filename is not None:
        fs = []
        with open(filename, "r") as fil:
            for line in fil:
                ls = line.split()
                sg_ind = int(ls[0])
                f = int(ls[1])
                if sg_ind in sg_selected:
                    fs.append([sg_ind, f])
    elif weighting == "uniform":
        fs = [[sg_ind, 1] for sg_ind in sg_selected]
    elif weighting == "ranks":
        fs = []
        for i in sg_selected:
            sg = SpaceGroups[i]
            fs.append([i, (192.0/sg.rank)**0.5])

    # prevent weighted list from being too long if something weird happens
    fmax = max([x[1] for x in fs])
    scale = 10000/fmax
    fs = [[x[0], x[1]*scale] for x in fs]
    # TODO
    # The weighted list approach is a horrible way of doing this...
    # not worth changing it atm, whole code needs a re-write
    weighted_dict = {}
    freq_dict = {}

    for row in fs:
        sg_ind = row[0]
        f = row[1]
        if f > 1e-5:
            f = math.ceil(f)
        else:
            f = 0

        # freq_list = [...[sg_ind, f] ...]
        freq_dict[sg_ind] = f
        temp = [sg_ind] * f
        weighted_dict[sg_ind] = temp

    return weighted_dict, freq_dict
