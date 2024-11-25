
"""."""

import os
import json
import gzip
import spglib
import numpy as np
import wabgen.utils

# find the directory
directory = os.path.dirname(__file__)


periodicTable = [None, 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
                 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
                 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
                 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm',
                 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
                 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U',
                 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs',
                 'Mt', 'Ds', 'Rg', 'Cn', 'Uut', 'Fl', 'Uup', 'Lv', 'Uus', 'Uuo']


T_dict = {}
T_dict["I"] = np.array([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])
T_dict["H"] = np.array([[-1, 1, 1], [2, 1, 1], [-1, -2, 1]])/3.0
T_dict["A"] = np.array([[0, 0.5, 0.5], [0, 0.5, -0.5, ], [1, 0, 0]])
T_dict["C"] = np.array([[0.5, 0.5, 0], [0.5, -0.5, 0], [0, 0, 1]])
T_dict["F"] = np.array([[0.5, 0, 0.5], [0.5, 0.5, 0], [0, 0.5, 0.5]])


# alternative form of the spacegroup constraints for symmetric bfgs
restric = {}
restric["triclinic"] = {}
restric["monoclinic"] = {"alpha": 0.5*np.pi, "gamma": 0.5*np.pi}
restric["orthorhombic"] = {"alpha": 0.5*np.pi, "bet": 0.5*np.pi, "gamma": 0.5*np.pi}
restric["tetragonal"] = {"alpha": 0.5*np.pi, "bet": 0.5*np.pi, "gamma": 0.5*np.pi, "b": "a"}
restric["cubic"] = {"alpha": 0.5*np.pi, "bet": 0.5*np.pi, "gamma": 0.5*np.pi, "b": "a", "c": "a"}
restric["trigonal"] = {"alpha": 0.5*np.pi, "bet": 0.5*np.pi, "gamma": 2.0/3*np.pi, "b": "a"}
restric["hexagonal"] = restric["trigonal"]


def enforce_symmetry(sg, cell):
   system = sg.system   
   #print(system)
   if system == 'TRICLINIC':
      pass
   elif system == 'MONOCLINIC':
      cell.angles[0] = 0.5*np.pi
      cell.angles[2] = 0.5*np.pi
   elif system == 'ORTHORHOMBIC':
      cell.angles[0] = 0.5*np.pi
      cell.angles[1] = 0.5*np.pi
      cell.angles[2] = 0.5*np.pi
   elif system == 'TETRAGONAL': # Use |b| = |a|
      cell.angles[0] = 0.5*np.pi
      cell.angles[1] = 0.5*np.pi
      cell.angles[2] = 0.5*np.pi
      cell.lengths[1] = cell.lengths[0]
   elif system == 'CUBIC': # Fix |a| = |b| = |c|
      cell.angles[0] = 0.5*np.pi
      cell.angles[1] = 0.5*np.pi
      cell.angles[2] = 0.5*np.pi
      cell.lengths[1] = cell.lengths[0]
      cell.lengths[2] = cell.lengths[0]
   elif system == 'TRIGONAL' or 'HEXAGONAL': # Use the hexagonal setting
      cell.angles[0] = 0.5*np.pi
      cell.angles[1] = 0.5*np.pi
      cell.angles[2] = (2.0/3.0)*np.pi
      cell.lengths[1] = cell.lengths[0]
   else:
      raise ValueError("Crystal system not recognised")
   
   
   #enforced the original symmetry group
   #use conversion matrix to primitive to transform
   #the basis vectors...
   M = sg.M 

   cell.calc_props(atoms=True, fromCartesian=False)

   old = cell.cartBasis #vecs are rows

   #basis should transform in the inverse way as the fractional coordinates
   Minv = np.linalg.inv(M)

   old_col = np.transpose(old)
   new_col = np.dot(old_col,Minv)
   new = np.transpose(new_col)

   cell.cartBasis = np.array(new)
   cell.calc_props(atoms=True, fromCartesian=True)
   

   cell.calc_props(atoms=True, fromCartesian=False)


def read_space_group_database(fname):
    f = gzip.open(fname, 'rb')
    # added the .decode for the pymat environemnt with newer pymatgen
    dbstr = f.read().decode("utf-8")
    f.close()
    db = json.loads(dbstr)
    for i in range(1, len(db)):
        for key in ['wyckoffOps']:
            try:
                for j in range(0, len(db[i][key])):
                    for k in range(0, len(db[i][key][j])):
                        db[i][key][j][k] = [np.array(x) for x in db[i][key][j][k]]
            except KeyError:
                pass
        for key in ['operators']:
            try:
                for j in range(0, len(db[i][key])):
                    db[i][key][j] = [np.array(x) for x in db[i][key][j]]
            except KeyError:
                pass

        try:
            db[i]["M"] = np.reshape(db[i]["M"], (3, 3))
        except KeyError:
            pass
    return db


PrimdatabaseName = os.path.join(directory, "../", "data/jpd47PrimSpaceGroupDataBase.gz")
prim_db = read_space_group_database(PrimdatabaseName)

databaseName = os.path.join(directory, "../", "data/jpd47SpaceGroupDataBase.gz")
db = read_space_group_database(databaseName)


def retrieve_symmetry_group(number, groupType='space', reduce_to_prim=True):
    """230 space groups, 75 rod groups, 7 frieze groups."""
    assert number > 0
    if groupType == 'space':
        assert number <= 230
        if reduce_to_prim:
            sg = prim_db[number]
        else:
            sg = db[number]
        return sg
    elif groupType == 'rod':
        assert number <= 75
        return db[230+number]
    elif groupType == 'frieze':
        assert number <= 7
        return db[230+75+number]


def cell2conv(cell, to_prim=False, stol=1e-4):
   """converts the cell to the spglib standardized cell"""
   #get the standard cell parameters  
   spgcell = UnitCell2spglibcell(cell)
   
   try:
      l, fp, n = spglib.standardize_cell(spgcell, to_primitive=to_prim, symprec=stol)
   except:
      print("standardizing didn't work!")
      print(spglib.standardize_cell(spgcell, to_primitive=to_prim))
      exit()


   #convert the cell to the standard format
   cell = wabgen.utils.cell.UnitCell(cartBasis=l)
   for i in range(0,len(n)):
      cell.add_atom(label = periodicTable[n[i]], fracCoords = fp[i]) 
   return cell 


def niggli_reduce(cell, to_prim=True, stol=1e-5):
   cell = cell2conv(cell, to_prim=to_prim)

   spgcell = UnitCell2spglibcell(cell)

   nl = spglib.niggli_reduce(spgcell[0], eps=stol)


   old_fp = [[at.label, at.fracCoords] for at in cell.atoms]  
   cell.cartBasis = np.array(nl)
   cell.atoms = []
   for at in old_fp:
      f = basis_change(spgcell[0],nl,Monly=False,t=at[1]) 
      cell.add_atom(label = at[0], fracCoords = np.array(f))

   cell.calc_props(atoms=True, fromCartesian=True)
   return cell


def UnitCell2spglibcell(Ucell):
   #conerst unit cell class to spglib cell
   lattice = Ucell.cartBasis
   positions = [at.fracCoords for at in Ucell.atoms]  
   numbers = [periodicTable.index(at.label) for at in Ucell.atoms]   
   spgcell = (lattice, positions, numbers)
   return spgcell


def basis_change(old_basis,new_basis,Monly=True,t=[0,0,0]):
   #basis specified by the three vectors [a1,a2,a3]
   #change from fractional coordinates in the old basis
   #to fractional coordinates in the new basis
   #and wrap the coordinates

   #t can be a vector or a matrix, dimension will be detected
   #and the appropriate transformation will be applied

   A = np.transpose(new_basis)
   U = np.transpose(old_basis)
   M = np.dot(np.linalg.inv(A),U)

   if Monly:
      return M
   
   t = np.array(t)
   if np.ndim(t) == 1:
      #transfrom vector
      return np.dot(M,t)
   elif np.ndim(t) == 2:
      #transform matrix
      Minv = np.linalg.inv(M)
      return np.dot(M,np.dot(t,Minv))
   else:
      print("can only pass vectors or matrices")


def wrap_fc(fp):
   tol = 10**-5   
   fp2 = [x%1 for x in fp]
   
   #this bit seems to be needed otherwise end up with 0.99999999999 a lot
   #which seems to muck things up :
   for i in range(0,3):
      if fp2[i] > 1-tol:
         fp2[i] =0
   return fp2

