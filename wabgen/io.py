
"""Submodule registering input/output functions in WABgen."""

import os
import shutil
import numpy as np
from wabgen.utils.castep import general_castep_parse
from wabgen.core import Molecule

# find the directory
directory = os.path.dirname(__file__)


def prepare_output_directory(directory_path):
    """
    Verify if the output directory exists, if not, creates it.

    If it exists, removes all the files in it.

    Args:
    -----
        directory_path (str): The path to the output directory.
    """
    if not os.path.exists(directory_path):
        # Create the directory if it does not exist
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        # Directory exists, remove all files inside it
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        print(f"All files deleted in the directory: {directory_path}")


def str_to_func(s):
   l = s.split(".")
   module = __import__(l[0])
   fs = [module]
   for x in l[1:]:
      fs.append(getattr(fs[-1],x))
   return fs[-1]


def line_to_func(line):
   """splits a line up into function and arguments
   returns [function, args, kwargs]"""
   ls = line.split()
   func = str_to_func(ls[0])
   args = []
   kwargs = {}
   for w in ls[1:]:
      if "=" in w:
         s = w.split("=")
         assert len(s) == 2
         kwargs[s[0]] = s[1]
      else:
         args.append(w)
   return [func, args, kwargs]


def parse_file(fname, st, template=False):
   """Pass the input file and return a list of molecules objects."""
   dsites = None

   out = general_castep_parse(fname, ignoreComments=False)

   # print(out)

   # setup the groups/ligands
   groups = set()
   for line in out["positions_abs"]:
      groups.add(line[6])

   # print("Ligands:", groups)

   # parse in the pressure, use 0.5 GPa if not present to aid convergence
   P = 0.5
   if "pressure" in out:
      for line in out["pressure"]:
         P = float(line[0])
   # print("Pressure:", P, "GPa")

   # parse in the gulp potentials
   gps = []
   if "gulp_potentials" in out:
      for line in out["gulp_potentials"]:
         gps.append(line)
   # print("GULP potentials:\n", gps)

   unit_formular = []
   if "unit_formular" in out:
      for line in out["unit_formular"]:
        if len(line) == 2:
           unit_formular = list(range(line[0], line[-1] + 1))

   # print("Z value:\n", unit_formular)
   Z_molecules = {z: {"MOLS": [], "V_dist": None} for z in unit_formular}
   # print("Nueva estructura:", Z_molecules)

   # parse in the cell
   cell_abc = []
   if "lattice_abc" in out:
      for line in out["lattice_abc"]:
         for x in line:
            cell_abc.append(float(x))
   # print("cell_abc:\n", cell_abc)

   # append the molecules to the mols object
   mols = []
   for group in groups:
      # print(group)
      coords = []
      species = []
      for line in out["positions_abs"]:
         if line[6] == group:
            species.append(line[0])
            coords.append([line[1], line[2], line[3]])
      number = 0     # default if not present
      for line in out["insert"]:
         if line[1] == group:
            number = line[0]
      if dsites is not None:
         if group in dsites:
            print(dsites[group])
      # print("number:", number)
      # print("species:", species)
      mols.append(Molecule(species, coords, number=number, name=group, st=st))
      # Probando la nueva estructura
      for z in Z_molecules:
        # print(number, z, number*z)
        Z_molecules[z]["MOLS"].append(Molecule(species, coords, number=z*number, name=group, st=st))

   p_mols = []

   # add the atoms
   for line in out["insert"]:
      if "pre_made" in line[1]:
         p_mols.append(line)
      elif not line[1] in groups:
         # print(line)
         number = line[0]
         atom_sym = line[1]
         # print(number)
         mols.append(Molecule([atom_sym], [0, 0, 0], number=number, Otype="Atom"))
         # Probando la nueva estructura
         for z in Z_molecules:
           # print(number, z, number*z)
           Z_molecules[z]["MOLS"].append(Molecule([atom_sym], [0, 0, 0], number=z*number, Otype="Atom"))

   # print("Molecules:\n", mols)

   # remove 0 species
   if not template:
      mols = [mol for mol in mols if mol.number > 0]

   # parse in the target volume as a normal distribution
   if "volume_distribution" in out:
      split = out["volume_distribution"][0]
      V_dist = ""
      for x in split:
         V_dist += str(x) + " "

   # print(V_dist)

   # parse in the target volume as a normal distribution
   if "density" in out:
      vol_atom = out["density"][0][0]
      # print("Density per atom [ang]^3 / N atom:", vol_atom)
      for z in Z_molecules:
         # print("Z:", z)
         natoms = 0
         for m in Z_molecules[z]["MOLS"]:
            natoms += m.number * len(m.species)

         # print("Total numer of atoms:", natoms)
         v_centre = vol_atom * natoms
         # print("Volume center:", v_centre)
         v_min = v_centre - 100
         v_max = v_centre + 100
         # print("Volume range:", v_min, v_max)
         Z_molecules[z]["V_dist"] = f"numpy.random.uniform low={v_min} high={v_max}"
   
   #read in the target_atom_numbers from a block
   target_atom_nums = None
   if "target_atom_numbers" in out:
      if len(out) > 0:
         target_atom_nums = {}
      for line in out["target_atom_numbers"]:
         target_atom_nums[line[0]] = int(line[1])


   # read in the min_seps
   min_seps = None
   if "min_seps" in out:
      els = []
      for el in out["min_seps"][0]:
         els.append(el)
      D = {}
      for i,line in enumerate(out["min_seps"][1:]):
            D[els[i]] = {}
            for j,x in enumerate(line):
               D[els[i]][els[j]] = float(x)
      min_seps = [els, D]

      #if minseps are not specified explicitly, this should be the default!
      #use the sum of the known covalent radii as a starting point
   else:

      fname = directory + "/data/covalent_radii.dat"
      with open(fname, "r") as f:
         c_rads = {}
         for line in f:
            ws = line.split()
            c_rads[ws[0]] = float(ws[1])

      #create list of all elements in the molecule
      els = set()
      for mol in mols:
         for at in mol.species:
            els.add(at)
      els = list(els)

      #construct the relavent minseps object in the same style as before
      D = {}
      for el1 in els:
         D[el1] = {}
         for el2 in els:
            D[el1][el2] = c_rads[el1]+c_rads[el2]
      min_seps = [els,D]

   #pdict is perm_dict if a specific perm is specified
   input_params = {}
   input_params["mols"] = sorted(mols, key=lambda m: m.name)

   # print molecules to check they are read correctly
   # if True and not template:
   #    print("type of mols is", type(mols))
   #    print("mols are")
   #    for mol in mols:
   #       print(mol.name, mol.number, mol.Otype)
   #       if mol.Otype == "Mol":
   #          print(mol.symbol.HM)

   input_params["V_dist"] = V_dist
   input_params["min_seps"] = min_seps
   input_params["target_atom_nums"] = target_atom_nums
   input_params["cell_abc"] = cell_abc
   input_params["p_mols"] = p_mols
   input_params["gulp_potentials"] = gps
   input_params["pressure"] = P
   input_params["Z_molecules"] = Z_molecules

   return input_params


def write_perm(perm, cell_name, o_name, sg, Mols):
   """write the cell_name and perm to origins file"""
   pstr = str(sg.number) + " "
   for mol_ind in perm:
      mol_string = Mols[mol_ind].name + ": "
      for si in perm[mol_ind]:
         mol_string += sg.sites[si].letter
         mol_string += str(perm[mol_ind][si])
      mol_string += "  "
      pstr += mol_string

   out_text = cell_name + "\n" + pstr + "\n"
   with open("../"+o_name,"a") as f:
      f.write(out_text)



def write_res(fname,cell, E=None, platon=False, P=0.0):
   #writes a res file from an instance of the UnitCell class
   f = open(fname+".res", 'w')
   #title
   vol = abs(np.linalg.det(cell.cartBasis))
   n = len(cell.atoms)
   titl = "TITL " + fname.split("/")[-1] + " "+str(P)+" " + str(vol)
   if E is None:
      titl += " 0.0"
   else:
      titl += " "+str(E)
   titl += " 0 0 " + str(n)+" (P1) n - 1\n"
   f.write(titl)
   #cell
   f.write("CELL ")
   f.write("  1.54180   ")
   for l in cell.lengths:
      if not platon:
         f.write('{0:.12f}'.format(l)+"   ")
      else:
         f.write('{0:.4f}'.format(l)+"   ")
   for a in cell.angles:
      ad = a*180.0/np.pi
      if not platon:
         f.write('{0:.12f}'.format(ad)+"   ")
      else:
         f.write('{0:.5f}'.format(ad)+"   ")
   f.write("\n")
   if platon:
      f.write("ZERR 1 0 0 0 0 0 0\n")
   #symmetry
   f.write("LATT -1\n")
   #atoms count

   #2. write the types of atoms present
   f.write("SFAC")
   types = set()
   for at in cell.atoms:
      types.add(at.label)
   types = list(types)
   types = sorted(types, key = lambda x: x)
   tdict = {}
   for i,el in enumerate(types):
      tdict[el] = str(i+1)
   #print("tdict is", tdict)

   for el in types:
      f.write(" "+el)
   f.write("\n")

   #atom positions
   for i,at in enumerate(cell.atoms):
      f.write(at.label+"  "*2)
      f.write(tdict[at.label]+"  ")
      for fp in at.fracCoords:
         f.write('{0:.10f}'.format(fp)+"  ")
      f.write("1.0\n")
   f.write("END\n")
