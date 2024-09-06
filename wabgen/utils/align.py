
"""."""

import os
import numpy as np
from wabgen.utils.data import write_db, read_db

# find the directory
directory = os.path.dirname(__file__)


def parallel(eig1, eig2):
    """Check if eig1 is parallel to eig2."""
    tol = 0.00001
    if np.dot(eig1, eig2) > tol:
        if np.linalg.norm(np.cross(eig1, eig2)) < tol:
            return True

    return False


def find_perp(u):
   """find a vector perpendicular to u"""
   tol = 10 ** -4
   vs = [np.array([1,0,0]),np.array([0,1,0])]
   for v in vs:
      a = np.cross(u,v)
      if np.linalg.norm(a) > tol:
         return a


def calc_R1(x1,y1):
   #gives a rotation matrix that will rotate x1 onto y1
   #print("x1 and y1 are", x1, y1)
   Id = np.identity(3)
   u = np.cross(x1,y1)
   tol = 10**-12
   if np.linalg.norm(u) < tol:
      #already parallel/anti-parallel
      if np.dot(x1, y1) > 0:
         return Id
      else:
         #print("currently anti-parallel")
         a = find_perp(y1)
         #rotate by 180 about a =  rotate a onto z, rotate about z then rotate back
         Ra = calc_R1(a,np.array([0,0,1]))
         Rz = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
         Rainv = np.linalg.inv(Ra)
         R1 = np.dot(Rainv,np.dot(Rz,Ra))


         r1 = np.dot(R1, x1)
         assert np.dot(r1, y1) > 0
         return R1

   sin_th = np.linalg.norm(u)/(np.linalg.norm(x1)*np.linalg.norm(y1))
   cos_th = np.dot(x1,y1)/(np.linalg.norm(x1)*np.linalg.norm(y1))
   u = u/np.linalg.norm(u)

   #use wiki formula for R1
   ux = np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
   ut = np.tensordot(u,u,axes=0)
   R1 = cos_th*Id + sin_th*ux + (1-cos_th)*ut

   #temporary check
   r1 = np.dot(R1, x1)
   assert np.dot(r1, y1) > 0

   return(R1)


def find_R2(meig1, meig2, seig1,seig2, tol=10**-4):
   """find a rotation matrix to rotate meigs onto seigs"""
   m1 = meig1/np.linalg.norm(meig1)
   m2 = meig2/np.linalg.norm(meig2)
   m3 = np.cross(m1,m2)
   s1 = seig1/np.linalg.norm(seig1)
   s2 = seig2/np.linalg.norm(seig2)
   s3 = np.cross(s1,s2)
   calpha = np.dot(m1, m2)
   cbeta = np.dot(s1, s2)
   if abs(calpha - cbeta) > tol:
      # print("calpha and cbeta are", calpha, cbeta)
      calpha_clipped = np.clip(calpha, -1, 1)
      cbeta_clipped = np.clip(cbeta, -1, 1)
      alpha = 180/np.pi * np.arccos(calpha_clipped)
      beta = 180/np.pi * np.arccos(cbeta_clipped)
      #print("alpha and beta are", alpha, beta, alpha+beta, alpha-beta, alpha/beta)
      return False, None

   Mp = np.transpose(np.array([s1, s2, s3]))
   M = np.transpose(np.array([m1, m2, m3]))
   Minv = np.linalg.inv(M)
   R2 = np.dot(Mp, Minv)
   assert abs(np.linalg.det(R2) -1) < tol
   return True, R2


def symm_equiv(R1,R2,rots):
   """check if the orientations defined by R1 and R2 are equivalent under rots"""
   R1inv = np.linalg.inv(R1)
   Rp = np.dot(R1inv, R2)
   for r in rots:
      if np.allclose(r,Rp) or np.allclose(Rp,r):
         return True
   return False


def check_alignment(mol, site, R):
   """check that when R is applied to the molecule all operators in site are matched"""
   #1. generate list of eigenvalues of molecule after rotation
   tol = 10 ** -4
   reigs = {}
   for op_type in mol.unique_eigs:
      reigs[op_type] = []
      for meig in mol.unique_eigs[op_type]:
         reigs[op_type].append(np.dot(R,meig))

   #2. check that every operator in the site is satisfied
   for op_type in site.unique_eigs:
      for s_eig in site.unique_eigs[op_type]:
         matched = False
         for m_eig in reigs[op_type]:
            if np.linalg.norm(np.cross(s_eig, m_eig)) < tol:
               matched = True
               break
         if not matched:
            return False
   #if here then every operator in the site is matched so all good!
   return True


def ops2symbol_dict(ops):

   symbol_dict = {}
   for op in ops:
      key = op.op_symbol
      if key in symbol_dict:
         symbol_dict[key].append(op)
      else:
         symbol_dict[key] = [op]

   return symbol_dict


def mol2site2(mol,site):
   """returns list of rotation matrices that align molecule with site
   so that symmetry operators of site are preserved. empty list if not possible"""
   Rs = []
   if mol.Otype == "Atom":
      return [np.identity(3)]

   #1. check if mol has enough operators, if not return [] as hopeless
   site_ops = ops2symbol_dict(site.C_operators)
   mol_ops = ops2symbol_dict(mol.C_operators)

   for key in site_ops:
      if key not in mol_ops:
         return []
      if len(mol_ops[key]) < len(site_ops[key]):
         return []

   #2. loop over the eigenvalues of the site trying to find two which are not paralell

   #a. no operators other than 1 and -1
   if len(site.unique_eigs) == 0:
      #print("no alignment needed")
      return [None]  #TODO might need to change this later

   #b. if here then at least one eig that needs to be oriented!
   op_eigs = []
   for key in site.unique_eigs:
      op_eigs.append([key,site.unique_eigs[key][0]])
      eig1 = site.unique_eigs[key][0]
      break    #so only add one eigenvalue

   two = False
   tol = 10**-4

   for key in site.unique_eigs:
      for eig in site.unique_eigs[key]:
         if np.linalg.norm(np.cross(eig1, eig)) > tol:
            #then not parallel
            op_eigs.append([key, eig])
            two = True
            break
      if two:
         break
   #should now have 1 or two elements in
   if not two:
      assert len(op_eigs) == 1
      #print("one axis to satisfy")
   else:
      assert len(op_eigs) == 2
      #print("two distinct to satisfy")
  # print(op_eigs)

   #now have two possible cases that need to be handled
   #two non-parallel site ops or all parallel

   #3. all parallel
   #pick one op type from site and loop over symm equivalent ops in mol
   if not two:
      op_type = op_eigs[0][0]
      site_eig = op_eigs[0][1]
      R_pos = []
      for g_inds in mol.gfp[op_type]:
         op_ind = g_inds[0]   #index of representative element from symmetry equivalent group
         mol_eig = mol.C_operators[op_ind].eig
         #check if there is a 2 fold rotation axis perpendicular
         #if so then line up once with site eig for one way, if not align + and - for two ways
         both = True
         if 2 in mol.unique_eigs:
            for two_eig in mol.unique_eigs[2]:
               #print("mol_eig and two_eig are", mol_eig, two_eig)
               #print(np.dot(mol_eig, two_eig))
               if abs(np.dot(mol_eig, two_eig)) < tol:
                  #have a perpendicular two fold rotation axis
                  both = False
                  break
         #print("both is", both, site.symbol.HM)
         R1 = calc_R1(mol_eig, site_eig)
         R_pos.append([R1,site_eig])
         if both:

            R2 = calc_R1(-mol_eig, site_eig)
            R_pos.append([R2, site_eig])

      #now have list of all possible alignments that are symmetry distinct
      #for each alignment need to check that it satisfies all of the operators
      for rot_opt in R_pos:
         R = rot_opt[0]
         R_eig = rot_opt[1]
         if check_alignment(mol,site,R):
            Rs.append(rot_opt)
      #print("if here have one eigenvector to align")
      """
      for r in Rs:
         print("r is", r)
      """
      return Rs      #return list distinct alignments where all site ops are matched

   #print("have two distinct eigenvectors to align")

   #4. two distinct
   site_eigs = op_eigs #contains 2* [op_type, eig] to be matched
   Rpos = []
   for mol_eig1 in mol.unique_eigs[site_eigs[0][0]]:
      #R1 = calc_R1(mol_eig1, site_eigs[0][1])

      for mol_eig2 in mol.unique_eigs[site_eigs[1][0]]:
         pos, R2 = find_R2( mol_eig1, mol_eig2,site_eigs[0][1], site_eigs[1][1])
         if pos:
            Rpos.append(R2)
   #print("len of Rpos is", len(Rpos))

   #now have list of all possible rotations
   Rchecked = []
   for R in Rpos:
      if check_alignment(mol, site,R):
         Rchecked.append(R)

   #print("len of Rchecked is", len(Rchecked))

   #reduce Rchecked
   Rch = []
   for R in Rchecked:
      new = True
      for R2 in Rch:
         if np.allclose(R,R2):
            new = False
            break
      if new:
         Rch.append(R)

   #print("len of Rch is", len(Rch))


   #now symmetry reduce over the remaing alignments that work
   #construct list of rotation operators from
   if "Rnorm" in mol.__dict__:
      #print("using Rnorm for mol:", mol.name)
      rots =[R.matrix for R in  mol.Rnorm]
   else:
      #print("not using Rnorm for mol:", mol)
      rots = [op.matrix for op in mol.C_operators if np.linalg.det(op.matrix) > 0.5]

   #print("len of rots is", len(rots))
   Rs = []
   for R1 in Rchecked:
      new = True
      for R2 in Rs:
         if symm_equiv(R1, R2, rots):
            new = False
            break
      if new:
         Rs.append(R1)

   #print("len of Rs is", len(Rs))
   return Rs


def RCI(mol, site, M):
   """takes in R such that R.Mol matches site.
   return canonical image of eigenvector alignment"""
   #1. apply R ops of the mol to the coords
   #  for each one find the eig_alignmnet
   #  save the minimum eig_alignment and return

   def ea1smoller(ea1, ea2):
      """returns true if ea1 is smoller than ea2"""
      #1. both should have the same keys, sort the keys then loop over until one is differnet
      keys = ea1.keys()
      skeys = sorted(ea1, key = lambda x: sum([int(y) for y in x.split("_")]))
      #print("skeys are", skeys)
      #leave ^^ there until there is a better testss


      #2.
      for key in skeys:
         key1 = ea1[key]
         key2 = ea2[key]
         n1 = sum([int(y) for y in key1.split("_")])
         n2 = sum([int(y) for y in key2.split("_")])
         if n1 > n2:
            return False
      return True


   def ea1_smaller(ea1,ea2):
      """returns true if ea1 is smaller than ea2"""
      keys = ea1.keys()
      skeys = sorted(keys, key=lambda x: [x.split("_")[0], x.split("_")[1]])
      smaller = True
      for key in skeys:
         key1 = [int(x) for x in ea1[key].split("_")]
         key2 = [int(x) for x in ea2[key].split("_")]
         if key1 > key2:
            smaller = False
         if key1 < key2:
            break

      return smaller

   if M is None:
      return None
   elif len(M) ==2:
      #print("one aligned")
      R = M[0]
   else:
      R = M

   ea = find_aligned(mol,site,R)
   #print("\n\nea from R is", ea)

   for key in ea:
      if len(ea[key]) == 0:
         return None


   ea_min = ea2min(ea, site)


   for Ri in mol.Rnorm:
      Rp = np.dot(R, Ri.matrix)
      ea = find_aligned(mol, site, Rp)
      ea_0 = ea2min(ea, site)
      #compare the ea's to see which are smoller   #TODO check this, probs the error is here..
      #print("ea_0 is", ea_0)
      #print("ea_min is", ea_min)
      if ea1_smaller(ea_0, ea_min):
         #print(ea_0, "smaller than", ea_min)
         ea_min = ea_0

   return ea_min


def read_rotation_dict(mol, SpaceGroups):
   fname = directory + "/../data/rot_normalisers/" + mol.point_group_sch+"/"+mol.point_group_sch+"_rotdict.txt.gz"

   folname = directory + "/../data/rot_normalisers/" + mol.point_group_sch
   if not os.path.exists(folname):
      os.mkdir(folname)
      print("folname is", folname)
   if not os.path.exists(fname):
      print("couldn't find rotation dict, generating it")
      write_rotation_dict(mol, SpaceGroups)
   else:
      pass

   rot_dict = read_db(fname)
   return rot_dict


def write_rotation_dict(mol, SpaceGroups):
   """takes in a molecule, loops over all spacegroups and all sites
   and write the ea0 for each number to a file"""
   fname = directory + "/../data/rot_normalisers/" + mol.point_group_sch+ "/"+mol.point_group_sch +"_rotdict.txt"
   print("fname is", fname)
   print("mol.symbol is", mol.symbol.HM)
   rot_dict = {}
   for sg in SpaceGroups[1:]:
      print("sg.name is", sg.name)
      rot_dict[sg.number] = {}
      for site in sg.sites:
         #print("site.letter is", site.letter, "site.symbol is", site.symbol.HM)
         rot_dict[sg.number][site.letter] = {}
         Ms = mol2site2(mol, site)
         #if len(Ms) == 0 rot_dict[sg.number][site.letter] will remain empty, not an issue
         for i,M in enumerate(Ms):
            #print("generating rci0 for option", i, "/", len(Ms))
            rci0 = RCI(mol, site, M)
            #TODO remove this assertion it, debug only
            if rci0 is not None:
               R = ea02R(rci0, mol, site)
               rci0_2 = RCI(mol, site, R)
               assert rci0 == rci0_2
            rot_dict[sg.number][site.letter][i] = rci0
   write_db(rot_dict, fname)
   return 0


def n2R(sg, mol, site, rotnum, rot_dict):
   """returns a rotation matrix from a rotnum
      also returns cartesian site axis if there is a free
      rotational degree of freedom!"""
   try:
      ea0 = rot_dict[str(sg.number)][site.letter][str(rotnum)]
   except KeyError:
      print("\n"*5)
      print("CHRONIC ERROR")
      print("not in the rot dict...")
      print("sg number, site letter and rot num are")
      print(sg.number, site.letter, rotnum)
      print("mol and site symbol are")
      print(mol.symbol.HM, site.symbol.HM)
      print("rot dict for sg, site and molecule are")
      print(rot_dict[str(sg.number)][site.letter])
      return False
      exit()

   if ea0 is None:
      return None
   Ropt = ea02R(ea0, mol, site, return_site_cart_eig = True)
   return Ropt


def ea02R(ea0, Mol, site, return_site_cart_eig = False):
   """convert an eig alignment to a carteisan rotation matrix"""
   #print("ea0 is", ea0)
   if ea0 is None:
      print("shouldn't be parsed this case")
      exit()
   if len(ea0) == 1:
      #only one eig to align so this is fine
      for key in ea0:
         #1. find the site eig
         site_splt = key.split("_")
         s_op_ind = int(site_splt[0])
         s_eig_ind = int(site_splt[1])
         site_frac_eig = site.frac_eig_dict[s_op_ind][s_eig_ind]
         site_cart_eig = site.cell.frac_to_cart(site_frac_eig)

         #2. find the mol eig
         mol_splt = ea0[key].split("_")
         m_op_ind = int(mol_splt[0])
         m_eig_ind = int(mol_splt[1])
         mol_cart_eig = Mol.eig_dict[m_op_ind][m_eig_ind]

         #3. find an R to align
         R1 = calc_R1(mol_cart_eig, site_cart_eig)
         if not return_site_cart_eig:
            return R1
         else:
            return [R1, site_cart_eig]


   elif(len(ea0)) == 2:
      #two eigs to align
      meigs = []
      seigs = []
      for key in ea0:
         #1. find the site eig
         site_splt = key.split("_")
         s_op_ind = int(site_splt[0])
         s_eig_ind = int(site_splt[1])
         site_frac_eig = site.frac_eig_dict[s_op_ind][s_eig_ind]
         site_cart_eig = site.cell.frac_to_cart(site_frac_eig)
         seigs.append(site_cart_eig)

         #2. find the mol eig
         mol_splt = ea0[key].split("_")
         m_op_ind = int(mol_splt[0])
         m_eig_ind = int(mol_splt[1])
         mol_cart_eig = Mol.eig_dict[m_op_ind][m_eig_ind]
         meigs.append(mol_cart_eig)
      junk, R2 = find_R2(meigs[0], meigs[1], seigs[0], seigs[1])
      try:
         assert junk == True
      except AssertionError:
         print("junk and R2 are", junk, R2)
         print("meigs are", meigs)
         print("seigs are", seigs)
         print("ea0 is", ea0);
         print("exiting...")
         exit()
      return R2


def ea2min(ea, site):
   """takes in a full alignment
   returns minimum one/two non-parallel site eigs
    with min mol eig aligned for each"""

   def key1_greater(key1, key2):
      """bit of a weird function, only really used by ea2min"""
      s1 = key1.split("_")
      n1 = 10*s1[0] + s1[1]
      s2 = key2.split("_")
      n2 = 10*s2[0] + s2[1]
      if n1 > n2:
         return True
   """
   print("ea is", ea)
   for op in site.operators:
      print("op index and symbol are",  op.index, op.op_symbol)
   """

   #1. find smallest site eigenvalues which have |a^b| != 0
   #TODO this bit is wrong!! doesn't do what it tries to do
   min_eigs = []
   keys = [x for x in ea.keys()]

   #1a sort keys
   skeys = sorted(keys, key=lambda x: [x.split("_")[0], x.split("_")[1]])
   #print("skeys are", skeys)

   #1b loop over sorted keys finding first two that are not parallel

   for key in skeys:

      splt = key.split("_")
      site_op_ind = int(splt[0])
      eig_ind = int(splt[1])
      site_frac_eig = site.frac_eig_dict[site_op_ind][eig_ind]
      site_cart_eig = np.dot(site.cell.cartBasis, site_frac_eig)
      eig_opt = [key, site_frac_eig, site_cart_eig]
      new = True

      tol = 10 ** -4
      for i, eigo in enumerate(min_eigs):
         if np.linalg.norm(np.cross(eigo[2], site_cart_eig)) < tol:
            new = False
      if new:
         min_eigs.append(eig_opt)
      if len(min_eigs) == 2:
         break

   """
   #2. min_eigs will now contain 1 or 2 keys of minimum number of eigenvectors of site
   print("min_eigs is")
   for x in min_eigs:
      print(x)
   exit()
   """


   #now create new dict where each key in min_eigs is matched with smolest key from mol
   ea0 = {}
   for x in min_eigs:
      site_key = x[0]
      mol_key0 = ea[site_key][0]
      #print("x, site_key and mol_key0 are", x, site_key, mol_key0)
      for mol_key in ea[site_key]:
         if key1_greater(mol_key0, mol_key):
            mol_key0 = mol_key
      ea0[site_key] = mol_key0
   """
   print("ea0 is")
   for key in ea0:
      print(key, ea0[key])
   """
   """
   print("ea is")
   for key in ea:
      print(key, ea[key])
   print("ea0 is", ea0)
   """
   #print("ea0 after ea2min is", ea0)
   return ea0


def find_aligned(mol, site, R):
   """R.Mol aligns Mol with site.
    Given R return which of the molecules eigenvectors
   are aligned with which site eigenvectors"""
   #print("mol, site, R", mol, site, R)
   #1. loop over the eigenvectors of the site, finding mol eigenvectors which
   # are exactly parallel, ap not allowed
   ea = {}

   for op_ind in site.frac_eig_dict:
      #print("checking site operator", op_ind, site.operators[op_ind].op_symbol)
      for i,seig in enumerate(site.frac_eig_dict[op_ind]):
         site_cart_eig = site.cell.frac_to_cart(seig)
         key = str(op_ind) + "_" + str(i)
         ea[key] = []
         lst = []
         for moind in mol.eig_dict:
            for j,meig in enumerate(mol.eig_dict[moind]):
               key2 = str(moind) + "_"+str(j)
               R_meig = np.dot(R, meig)
               if parallel(site_cart_eig, R_meig):
                  ea[key].append(key2)
               """
                  print(site_cart_eig, R_meig, "are parallel", mol.C_operators[moind].op_symbol)
               else:
                  print(site_cart_eig, R_meig, "are not parallel", mol.C_operators[moind].op_symbol)
               """
         #check that at least one eigenvector was aligned from molecule
         if len(ea[key]) == 0:


            print(key, "is not aligned")
            print("corresponds tooo", site.operators[op_ind].op_symbol, site.operators[op_ind].index)
            print("mol and site are", mol.symbol.HM, site.symbol.HM)
            exit()


   #print for debugging
   """
   print("ea is")
   for key in ea:
      print(key, ":", ea[key])
   """
   return ea
