
"""."""

import copy
import math as m
import numpy as np



def crystal_modulo(position, eps=1e-10):
      # Translates the fractional coordinate vector 'position' back inside the unit cell,
      #  via addition of the appropriate lattice vector

      # eps is for fudging to avoid upsetting CASTEP's symmetry detection attempts;
      # it prevents e.g. (1,0,0) being translated to (0,0,0)
      # TODO: check if this is still necessary on newer CASTEP version
      for i in range(0,len(position)):
         position[i] = position[i] - int(position[i]-eps) + 1.0*int(position[i]<0)
         # Handle edge cases of exact equality
         if position[i] == 1.0:
            position[i] -= 1.0
         elif position[i] == -1.0:
            position[i] += 1.0
      return position


class UnitCell:
   # Describes the geometry of a unit cell
   def __init__(self, angles=[90.0, 90.0, 90.0],
             lengths=[1.0, 1.0, 1.0], cartBasis=None):
      # Pass either the lengths and angles, or the cartesian basis vectors
      # In the latter case, pass cartBasis=[[ax,ay,az],[bx,by,bz],[cx,cy,cz]].
      # angles should be passed in degrees and lengths in angstroms
      self.atoms = []
      if cartBasis is not None:
         # Compute angles, [alpha, beta, gamma]
         self.angles = [0, 0, 0]
         for iDim in range(0,3):
            # Compute indices of vectors to dot for the iDim'th angle
            cartBasis[iDim] = np.array(cartBasis[iDim])
         self.cartBasis = cartBasis
         self.calc_props(fromCartesian=True)
      else:
         # Unit cell vector lengths in A
         self.lengths = np.array(lengths)
         # Unit cell vectors' angles in rads (having been given in degrees)
         self.angles = np.array([x*(m.pi)/180.0 for x in angles])
         self.calc_props(fromCartesian=False)
      # Space group imposed on cell
      self.sg_num = None

      # Wyckoff information. self.equivAtoms[i][0] : [space group, index of Wyckoff site]
      #                 self.equivAtoms[i][1] : list of atoms
      #                 self.equivAtoms[i][2] : position s.t. acting with the Wyckoff operators for this particular Wyckoff site generates self.equivAtoms[i][1]
      self.equivAtoms = None

      # Molecules whose atoms are constrained to maintain their relative positions.
      self.groups = None


   def set_Energy(self, E):
      self.E = E

   def atoms_from_dict(self):
      atoms = []
      for key in self.atoms_dict:
         atoms.append(self.atoms_dict[key])
      self.atoms = atoms

   def add_atom(self, **kwargs):
      kwargs['cell'] = self
      self.atoms.append(Atom(**kwargs))
      return self.atoms[-1]

   def init_atoms_dict(self):
      self.atoms_dict = {}
      for i,at in enumerate(self.atoms):
         self.atoms_dict[i] = at
         at.key = i

   def init_bonds(self,merge):
      # loops over the atoms adding bonds to them
      self.init_atoms_dict()

      for key in self.atoms_dict:
         for key2 in self.atoms_dict:
            if key == key2:
               continue
            if self.atoms_dict[key].tag == self.atoms_dict[key2].tag:
               bond_type = "mol"
               natural_length = UnitCell.euclidian_distance(self,self.atoms_dict[key].fracCoords, self.atoms_dict[key2].fracCoords)
               self.atoms[key].add_bond(key2,bond_type,natural_length)
            else:
               if self.atoms_dict[key].label in merge and self.atoms_dict[key].label == self.atoms_dict[key2].label:
                  self.atoms[key].add_bond(key2,bond_type="merge")

   def calc_props(self, fromCartesian=False, atoms=True):
      # (Re)calculates unit cell properties.
      # if fromCartesian is true, then the Cartesian basis is preserved and
      #  lengths and angles are calculated using it; otherwise the converse occurs
      if fromCartesian:
         # Compute angles, [alpha, beta, gamma]
         self.angles = [0, 0, 0]
         for iDim in range(0,3):
            # Compute indices of vectors to dot for the iDim'th angle
            self.cartBasis[iDim] = np.array(self.cartBasis[iDim])
            vecInd = [0,1,2]
            vecInd.pop(iDim)

            dot = np.dot(self.cartBasis[vecInd[0]], self.cartBasis[vecInd[1]])
            dot /= (np.linalg.norm(self.cartBasis[vecInd[0]])\
                  *np.linalg.norm(self.cartBasis[vecInd[1]]))
            self.angles[iDim] = m.acos(dot)
         # Compute lengths
         self.lengths = [np.linalg.norm(self.cartBasis[iDim]) for iDim in range(0,3)]
         # Put the basis into matrix form
         # NB this is the transpose of the way CASTEP likes to represent the matrix
         self.cartMat = np.zeros((3,3))
         for iDim in range(0,3):
            for jDim in range(0,3):
               self.cartMat[iDim][jDim] = self.cartBasis[jDim][iDim]
               #self.cartMat[iDim][jDim] = self.cartBasis[iDim][jDim]
         self.fracMat = np.linalg.inv(self.cartMat)
         self.vol = np.dot(self.cartBasis[0], np.cross(self.cartBasis[1], self.cartBasis[2]))
         if self.vol < 0.0:
            self.vol = -self.vol

      else:
         self.cos = np.cos(self.angles)
         self.sin = np.sin(self.angles)
         # Calculate unit cell volume from frac coords
         self.lengths = np.array(self.lengths)

         x = 1 - self.cos[0]**2 - self.cos[1]**2 - self.cos[2]**2 + 2*self.cos[0]*self.cos[1]*self.cos[2]
         try:

            self.vol = self.lengths.prod()*m.sqrt(1 - self.cos[0]**2 - self.cos[1]**2- self.cos[2]**2 + 2*self.cos[0]*self.cos[1]*self.cos[2])
         except:
            #print("x is", x)
            #print("cell.lengths are", self.lengths)
            #print("cell.angles are", self.angles)
            abc = 1
            csq = 0
            cosprod = 1
            for i in range(0,3):
               abc *= self.lengths[i]
               csq += np.cos(self.angles[i])**2
               cosprod *= np.cos(self.angles[i])
            x = 1 - csq + 2*cosprod
            #print("x is", x)
            #print("lengths are", self.lengths)
            #print("angles are", 180/np.pi * self.angles)
            vol = abc*(x)**0.5
            #print("vol is", vol)


            a= self.lengths[0]*np.array([1,0,0])
            b = self.lengths[1]*np.array([np.cos(self.angles[2]),np.sin(self.angles[2]), 0])
            x = np.cos(self.angles[1])
            y = (np.cos(self.angles[0]) - x* np.cos(self.angles[2]))/np.sin(self.angles[2])
            c = self.lengths[2]*np.array([x,y,(1-x**2-y**2)**0.5])
            self.cartBasis = np.array([a,b,c])
            #print("vol is", np.linalg.det(self.cartBasis))


         # Calculate frac -> cart conversion matrix
         ln = self.lengths
         self.cartMat = np.array([[ln[0], ln[1]*self.cos[2], ln[2]*self.cos[1]],
         [0, ln[1]*self.sin[2], ln[2]*(self.cos[0]-self.cos[1]*self.cos[2])/self.sin[2]],
         [0, 0, ln[2]*self.vol/(self.lengths.prod()*self.sin[2])]])

         # Invert to get cart -> frac conversion matrix
         self.fracMat = np.linalg.inv(self.cartMat)

         # Hence get cartesian basis vectors
         self.cartBasis = np.array([self.cartMat[:,0],
                              self.cartMat[:,1],
                              self.cartMat[:,2]])


      # Recalculate atoms' cartesian positions, fixing their fractional coordinates
      if atoms:
         for iAt in range(0,len(self.atoms)):
            self.atoms[iAt].cartCoords = self.frac_to_cart(self.atoms[iAt].fracCoords)
            #self.atoms[iAt].fracCoords = self.cart_to_frac(self.atoms[iAt].cartCoords)

      #set the D matrix, D=(M)^TM, where M is cart basis
      #self.D = np.dot(np.transpose(self.cartMat), self.cartMat)

   def frac_to_cart(self, x):
      # Converts x=[x_a, x_b, x_c] from fractional to cartesian coordinates
      return self.cartMat.dot(x)

   def cart_to_frac(self, x):
      # Converts x=[x_a, x_b, x_c] from cartesian to fractional coordinates
      return self.fracMat.dot(x)


   #TODO need a big change in the way that euclidean distance is calculated to speed code up!



   def ed2(self,a,b):
      """only calculates the shortest euclidean distance"""
      #test, gives same shortest distance as eueclidean distance but ~40 faster
      dr = [(a[i]-b[i])%1 for i in range(0,3)]
      d2_min = 1000000
      for i in range(-1,2):
         for j in range(-1,2):
            for k in range(-1,2):
               dv = [dr[0]+i,dr[1]+j,dr[2]+k]
               dvec = self.frac_to_cart(dv)
               d2 = np.dot(dvec,dvec)
               if d2 < d2_min:
                  d2_min = d2
      return d2_min**0.5

   def euclidian_distance(self, a, b, all_pos=False, vec=False, shift=False):
      # Returns the minimum possible Euclidian distance
      #  between two FRACTIONAL coordinate vectors
      deltaR = np.array(a) - np.array(b)
      deltaR = np.array([x%1 for x in deltaR])
      ds = []
      vs = []
      shifts = []
      rs = []
      for i in range(-1,2):
         for j in range(-1,2):
            for k in range(-1,2):
               deltaR_shift = deltaR + np.array([i,j,k])
               rCart = self.frac_to_cart(deltaR_shift)
               #ds.append(np.linalg.norm(rCart))
               #ds.append(jm.norm(rCart))
               #vs.append(rCart)
               #shifts.append(np.array([i,j,k]))
               rs.append([np.linalg.norm(rCart), rCart, np.array([i,j,k])])


      if all_pos:
         rs = sorted(rs, key = lambda x: x[0])
         return rs
      else:
         rs = sorted(rs, key = lambda x: x[0])
         return rs[0]


   def ed_same_mol(self, acart, bcart, all_pos=False, vec=False, shift=False, sr=4):
      #cartesian coords have not been wrapped
      #ignore same shift as atoms within same molecule
      deltaR = np.array(acart) - np.array(bcart)
      ds = []
      vs = []
      shifts = []
      rs = []
      for i in range(-sr,sr+1):
         for j in range(-sr, sr+1):
            for k in range(-sr, sr+1):
               if i == 0 and j == 0 and k == 0:
                  continue
               shift =  np.array([i,j,k])
               rshift = self.frac_to_cart(shift)
               rCart = deltaR+rshift
               rs.append([np.linalg.norm(rCart), rCart, np.array([i,j,k])])

      if all_pos:
         rs = sorted(rs, key = lambda x: x[0])
         return rs
      else:
         rs = sorted(rs, key = lambda x: x[0])
         return rs[0]


   def ed3(self, a, b, all_pos=False, vec=False, shift=False):
      # Returns the minimum possible Euclidian distance
      #  between two FRACTIONAL coordinate vectors
      ds = []
      vs = []
      shifts = []
      rs = []
      for i in range(-1,2):
         for j in range(-1,2):
            for k in range(-1,2):
               fab = np.array(a)-np.array(b)+np.array([i,j,k])
               rCart = self.frac_to_cart(fab)
               #ds.append(np.linalg.norm(rCart))
               #ds.append(jm.norm(rCart))
               #vs.append(rCart)
               #shifts.append(np.array([i,j,k]))
               rs.append([np.linalg.norm(rCart), rCart, np.array([i,j,k])])

      if all_pos:
         rs = sorted(rs, key = lambda x: x[0])
         return rs
      else:
         rs = sorted(rs, key = lambda x: x[0])
         return rs[0]


class Atom:
   def __init__(self, cell, label=None, z=1, r=0.8, fracCoords=None,
             cartCoords=None, tag=None, molName=None, key=None, mc_rot_ind=None, Mopt=None):
      # Label of atom (i.e. its element name)
      self.label = label
      self.cell = cell
      # Coordinates of atom - set to (0,0,0) if user fails to specify either
      if fracCoords is None and cartCoords is None:
         fracCoords = np.array([0.0,0.0,0.0])
      # Make sure the coordinates lie inside the unit cell
      self.set_position(fracCoords, cartCoords)
      # TODO: crystal modulo check on entry via cartCoords?
      # Atomic number of atom
      self.z = z
      # Radius of atom
      self.r = r
      # Tag on atom (not its element name!)
      self.tag = tag
      # Wyckoff position index (defined w.r.t. the space group database, symm.db).
      # Format: space group index, wyckoff site index, index of point within wyckoff site,
      #       occupation index for when a single wyckoff site is multiply occupied
      self.wyck = None
      # Whether the atom is locked in place
      self.locked = False
      # Constraints restricting the atom's movement
      self.constraints = []
      # Magnetic moment
      self.spin = 0.0
      # Groups of equivalent molecules
      self.groups = []
      # self.key
      self.key = key
      # Bonds
      self.bonds = []
      self.molName = molName
      self.mcri = mc_rot_ind
      self.Mopt = Mopt

   def set_Force(self, force):
      self.Force = force

   def set_repeat_op(self, op):
      self.repeat_op = op

   def set_hyper_position(self, hyper_pos):
      """sets the hyper_position"""
      r3 = np.array(hyper_pos[0:3])
      rplus = np.array(hyper_pos[3:])
      self.set_position(cartCoords=r3, modulo=False)
      self.hyperCoords = rplus

   def add_bond(self,key,bond_type,natural_length=None, shift=None):
      #shift frac and is added to atom itself before calculating distances
      if shift is None:
         shift = np.array([0,0,0])
      b = [key, bond_type, natural_length, shift]
      self.bonds.append(b)

   def init_force(self):
  #initialises the forces to be 0
      self.force = np.array([0,0,0])


   def set_position(self, fracCoords=None, cartCoords=None, modulo=True):
      # Call with ONE of fracCoords or cartCoords specified; the other
      #  will be calculated
      # Check for both being mistakenly given
      if fracCoords is not None and cartCoords is not None:
         raise ValueError("Incorrect call to set_position with both frac & cart coords given")

      # Check for at least one being given
      if fracCoords is None and cartCoords is None:
         raise ValueError("Incorrect call to set_position with no coordinates given")

      # Fractional coordinates given
      if fracCoords is not None:
            # Make sure the coordinates lie inside the unit cell
            # TODO: handle case where specified cartCoords are outside cell, not just frac
            if modulo:
               crystal_modulo(fracCoords)
            self.fracCoords = copy.deepcopy(fracCoords)
            self.fracCoords = np.array(self.fracCoords)
            self.cartCoords = self.cell.frac_to_cart(self.fracCoords)
      # Cartesian coordinates given
      if cartCoords is not None:
            self.cartCoords = copy.deepcopy(cartCoords)
            self.cartCoords = np.array(cartCoords)
            self.fracCoords = self.cell.cart_to_frac(self.cartCoords)
   def get_frac_coords(self):
      return self.fracCoords

   def get_cart_coords(self):
      return self.cartCoords

   def print_properties(self):
         #print "Type="+str(self.label)+str(", r=")+str(self.r)+str(", mu=")+str(self.spin)
         #print "pos(frac)="+str(self.fracCoords)+", pos(cart)="+str(self.cartCoords)
         print( self.__str__())

   def __str__(self):
      outStr = ''
      outStr += "Type="+str(self.label)+str(", r=")+str(self.r)+str(", mu=")+str(self.spin)+'\n'
      outStr += "pos(frac)="+str(self.fracCoords)+", pos(cart)="+str(self.cartCoords)
      return outStr

