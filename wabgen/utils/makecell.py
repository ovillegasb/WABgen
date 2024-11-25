
"""."""

import random
import numpy as np
from wabgen.utils import symm
from wabgen.utils.cell import UnitCell
from wabgen.utils.align import n2R
import wabgen.core
import wabgen.io


def scale_cell(cell, V_dist):
   """scales the cell to be in line with the estimated volume"""
   V = np.linalg.det(cell.cartBasis)
   func, args, kwargs = wabgen.io.line_to_func(V_dist)
   args = [float(x) for x in args]
   for key, item in kwargs.items():
      kwargs[key] = float(item)
   st0 = np.random.get_state()
   #print(st0[1][:5])
   Vtarg = float(func(*args, **kwargs))

   print(f" V random is {V} Vtarg is {Vtarg}")
   s = (Vtarg/V)**0.3334
   cell2 = UnitCell(angles=np.array(cell.angles)*180.0/np.pi,lengths=s*np.array(cell.lengths))
   Vfinal = np.linalg.det(cell2.cartBasis)
   #print(f" V final is {Vfinal}")
   return cell2


def gen_rand_cell(cell_abc=[]):
   """generates a random cell if cell_abc is none"""
   if len(cell_abc) > 0:
      #use the specific angles and lengths that have been specified
      lengths = cell_abc[0:3]
      angles = cell_abc[3:6]

   else:
      #1. generate the angles
      theta_min = 15    # smallest angle to make as cell angle

      angles = [random.uniform(theta_min, 180-theta_min) for i in range(0, 2)]
      a_min = abs(angles[0] - angles[1])
      a_max = min([360 - sum(angles), sum(angles)])
      angles.append(random.uniform(a_min, a_max))

      #2. generate the lengths
      r_max = 10  # largest possible ratio of side lengths
      lengths = [random.uniform(1, r_max) for i in range(0, 3)]

   #3. form the cell
   cell = UnitCell(angles=angles, lengths=lengths)
   return cell


def perm2mol_list(perm):
    """Spits perm into symmetry equivalent molecules to be added and then randomise order."""
    mol_list = []
    tag = 0
    for molind, mol_alloc in perm.items():
        for site_ind, rot_inds in mol_alloc.items():
            for ri in rot_inds:
                mol_list.append([molind, site_ind, ri, tag])
                tag += 1

    random.shuffle(mol_list)
    return mol_list


def random_rotation():
   """returns a random rotation matrix"""
   randnums = [random.uniform(0,1) for i in range(0,3)]
   theta = randnums[0] * 2.0 * np.pi
   phi = randnums[1] * 2.0 * np.pi
   z = randnums[2] * 2.0

   r =  np.sqrt(z)
   V = [np.sin(phi) *r,np.cos(phi)*r,np.sqrt(2.0-z)]
   st = np.sin(theta)
   ct = np.cos(theta)

   R = np.array([[ct,st,0],[-st,ct,0],[0,0,1]])
   M = np.dot((np.outer(V,V)-np.identity(3)),R)
   #print("used random rotation")
   return M


def get_rand_R2(u, gam=None):
   u = u/np.linalg.norm(u)
   if gam is None:
      gam = random.uniform(-np.pi,np.pi)
   cos_gam = np.cos(gam)
   sin_gam = np.sin(gam)

   Id = np.identity(3)
   #wiki formula for R2
   ux = np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
   ut = np.tensordot(u,u,axes=0)
   R2 = cos_gam*Id + sin_gam*ux + (1-cos_gam)*ut
   #print("only one alignemnet, randomised over the free angle")
   return R2


def get_R_matrix(sg, mol, site, rot_ind, mol_rot_dict, theta=None):
    """
    Return rotation matrix requiered.

    Retrieves the matrix required to align the molecule with the site and subs in for random
    degrees of freedom.
    """
    Mopt = n2R(sg, mol, site, rot_ind, mol_rot_dict)
    # print("Mopt is", Mopt)

    if Mopt is None:
        M = random_rotation()
    elif len(Mopt) == 2:
        M = Mopt[0]
        site_axis = Mopt[1]
        if theta is None:
            R2 = get_rand_R2(site_axis)  # random rotation about site axis post alignment
        else:
            R2 = get_rand_R2(site_axis, gam=theta)  # random rotation about site axis post alignment
        M = np.dot(R2, M)
    else:
        M = Mopt

    return M, Mopt


def add_molecule(cell, mols, ml, sg, Rot_dicts, add_centre, frac_pos=None, theta=None):
    """
    Add the molecule to the specified site in the specified orientation.

    Sets the centre coords of the mol and adds the cart offset to each atom.
    """
    mol_ind = ml[0]
    mol = mols[mol_ind]
    site_ind = ml[1]
    rot_ind = ml[2]
    tag = ml[3]

    # print(sg.name, sg.number)
    # for n, s in enumerate(sg.sites):
    #     print(n, s.letter)
    # print("mol_ind is", mol_ind)
    # print("site_ind is", site_ind)
    # print("rot_ind is", rot_ind)
    # print("Rot_dicts", Rot_dicts.keys(), len(Rot_dicts))

    # 1. retrieve the site and update the cartesian operators of the site
    # print("sg.sites", sg.sites)
    site = sg.sites[site_ind]
    site.update_Cops(cell)

    # 3. find the operators needed to replicate mol around the cell
    mult = site.mult
    ops = [wabgen.core.Operator(0, np.identity(3), (0, 0, 0))]
    # print("ops:", ops)
    rsfp, vlist = site.randomised_fp()
    if frac_pos is not None:
        rsfp = frac_pos

    # override if manually specified
    sites = [rsfp]

    for op in sg.operators:
        new_site = wabgen.core.apply_op(op, rsfp)
        add = True
        for s in sites:
            if np.allclose(s, new_site):
                add = False
                break
        if add:
            sites.append(new_site)
            ops.append(op)

        if len(ops) == mult:
            break

    # 4a. if atoms add them now
    if mol.Otype == "Atom":
        for i, s in enumerate(sites):
            cell.add_atom(label=mol.species[0], fracCoords=s, tag=tag, molName=mol.name, key=i)
            cell.atoms[-1].set_repeat_op(ops[i])

        details = {
            "site": site,
            "fp": rsfp,
            "vlist": vlist,
            "repeater_ops": ops,
            "species": mol.species,
            "mc_frac": [rsfp],
            "centers": sites,
            "rmax": mol.rmax
        }
        return cell, details

    # 2. get the orientation matrix to apply to the molecule
    # mc_rot contains the offsets of all the atomic coordinates relative to the moleculs centre
    # print("Rot_dicts are", Rot_dicts.keys())
    if mol.std:
        R, Mopt = get_R_matrix(sg, mol, site, rot_ind, Rot_dicts[mol.name], theta=theta)
    else:
        #TODO make this more general for now only works for p1
        assert mol.point_group_sch == "C1"
        R = random_rotation()
        Mopt = None

    mc_rot = np.transpose(np.dot(R, np.transpose(mol.coords)))
    mc_frac = np.transpose(cell.cart_to_frac(np.transpose(mc_rot)))

    # 4. loop over list of operators adding symmetry equivalent molecules
    mc_fo = [x + rsfp for x in mc_frac]

    for j, op in enumerate(ops):
        for i in range(0, len(mc_fo)):
            if mol.constraint is not None:
                if mol.species[i] == "X":
                    continue

            # mc = apply_op(op, mc_fo[i])     switched to no wrapping of fractional coordinates
            mc = np.dot(op.matrix, mc_fo[i]) + op.t
            cell.add_atom(
               label=mol.species[i],
               fracCoords=[0, 0, 0],
               tag=tag,
               molName=mol.name,
               key=j,
               Mopt=Mopt
            )
            cell.atoms[-1].set_position(
               fracCoords=mc,
               modulo=False
            )
            cell.atoms[-1].set_repeat_op(op)
        # if merges are needed add a U atom at the centre of each cell
        if add_centre:
            cell.add_atom(label="U", fracCoords=[0, 0, 0], tag=tag, molName=mol.name, key=j, Mopt=Mopt)
            cell.atoms[-1].set_position(fracCoords = np.dot(op.matrix, rsfp)+op.t, modulo=False)
            cell.atoms[-1].set_repeat_op(op)

        details = {
            "site": site,
            "fp": rsfp,
            "repeater_ops": ops,
            "vlist": vlist,
            "mc_frac": mc_fo,
            "species": mol.species,
            "centers": sites,
            "rmax": mol.rmax
        }

    return cell, details


def make_cell(mols, sg_ind, perm, V_dist, cell_abc, Rot_dicts, add_centre, ntries=None, min_seps=None):
    """Make a cell from the perm and info that has been parsed."""
    # 1. make the unit cell itself. Random if not explicity specified. Enforce Symmetry.
    cell = gen_rand_cell(cell_abc)
    sg = wabgen.core.SG(symm.retrieve_symmetry_group(sg_ind, reduce_to_prim=True), cell=cell)
    # print("cell:", cell)
    # print("sg:", sg)
    # print("sg_ind:", sg_ind)
    if len(cell_abc) == 0:
        cell = scale_cell(cell, V_dist)
    cell.sg_num = sg.number
    # print("cell is", cell)

    all_details = []
    # 2. split the permutation into symmetry equivalent molecules to be added.
    mol_list = perm2mol_list(perm)
    # print("mol_list:", mol_list)

    # printing
    if False:
        for mol in mols:
            print(mol.name, mol.Otype, mol.number)
        # print("perm is")
        # pp.pprint(perm)

        # print("mol_list is")
        # pp.pprint(mol_list)

    # 3. loop over molecules to be added adding them one by one
    for h, ml in enumerate(mol_list):
        # print(h, ml)  # ml: [mol_index, site_index, rot_index, tag]
        cell, details = add_molecule(cell, mols, ml, sg, Rot_dicts, add_centre)
        all_details.append(details)

    return all_details, cell


def cell2conv(cell):
   """converts a cell to the conventional centering
      assumes that
         - input cell has unwrapped fractional coords
         - atoms in same molcule are joined by bonds with shift [0,0,0]
         - symmetry equivalent molecule = same tag
         - same actual molecule same key as well"""

   ds = symm.get_dataset(cell)
   for key in ds:
      print(key, ds[key])

