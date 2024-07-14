
"""."""

import os
import numpy as np
from numba import jit
from scipy.optimize import minimize
import wabgen.utils.symm as symm
from wabgen.utils.solve_vars import init_atom_vars, check_assignments, init_cell_vars
from wabgen.utils.cell import UnitCell
from wabgen.utils.minimize import steepest_descent as SD


def gen_ms_bonds(cell, min_seps):
   """generates the min_seps distances and bond types, all bond_types are min_seps=2 atm"""
   msd = min_seps[1]
   n = len(cell.atoms)
   bond_dists = np.zeros((n,n))
   bond_types = np.zeros((n,n),dtype=int)
   
   for p, atp in enumerate(cell.atoms):
      for q, atq in enumerate(cell.atoms):
         bond_types[p,q] = 2
         bond_dists[p,q] = msd[atp.label][atq.label]+0.1
   return bond_dists, bond_types 


def gen_mol_bonds(cell, s, u):
   """loop over the atoms generating the molecule bonds"""
   Nbonds = 0
   n = len(cell.atoms)
   bmols = [[None for x in range(0,n)] for y in range(0,n)] 
   for i, at1 in enumerate(cell.atoms):
      fp1 = at1.fracCoords
      ui = u[i]
      assert np.linalg.norm(fp1-(ui+s[i])) < 0.01
      for j, at2 in enumerate(cell.atoms):
         fp2 = at2.fracCoords
         uj = u[j]
         assert np.linalg.norm(fp2-(uj+s[j])) < 0.01
         if i == j:
            bmols[i][j] = None
         elif at1.tag == at2.tag and at1.key == at2.key:
            v1 = cell.frac_to_cart(ui-uj+s[i]-s[j])
            d1 = np.linalg.norm(v1) 
            #print(at1.label, at2.label, d1)

            #check that shifts are all good
            v2 = cell.frac_to_cart(fp1-fp2)
            d2 = np.linalg.norm(v2)
            try:
               assert abs(d1-d2) < 0.01 
               #print("adding bond", at1.label, at2.label, d1)
            except:
               print("distance error", d1, d2)

            bmols[i][j] = [[1, s[i]-s[j], d1]]
            Nbonds += 1
   return Nbonds, bmols


def phi(d, bt, bd):
   """calculate the energy"""
   k = 3000
   K = 10*k
   if bt == 2:
      #minsep
      if d > bd:
         return 0.0, 0.0
      else:
         return K* (bd - d)**4, -4*K*(bd-d)**3
   elif bt == 1:
      #bonded
      return k * (bd-d) ** 4, -4*k*(bd - d )**3
   else:
      #ignore
      return 0.0, 0.0


def check_ci(ci):
   """check that the ci are possible variables"""
   #print("ci are", ci)
   ok = True
   #1. check sum of angles smaller than 360 degrees
   if sum(ci[0:3]) > 2*np.pi:
      ok = False
      #print("angles are too big", 180/np.pi*ci[0:3]) 
   
   #2. check largest angle smaller than sum of smallest two
   sangs = sorted(ci[0:3])
   if sangs[2] > sangs[0] + sangs[1]:
      ok = False
      #print("largest angle bigger than sum of others", 180/np.pi*ci[0:3])
   
   #3. check that all entries are positive
   if min(ci) < 0:
      ok = False
      #print("have a negative angle or length", 180/np.pi*ci[0:3], ci[3:])

   if not ok:
      #print("failed ci check...")
      return False
   else:
      #print("angles are", 180/np.pi*ci[0:3]) 
      return True


def ci_to_Ap(ci, Ap_only=False):
   """takes in the ci and return A', conventional cell basis
   ci = [alpha, beta, gamma, a, b, c] """
   ok = check_ci(ci)
   if not ok:
      return None
 
   #1. get variables out of ci
   alpha = ci[0]
   beta = ci[1]
   gamma = ci[2]
   a = ci[3]
   b = ci[4]
   c = ci[5]

   #2. calculate the cosines and sines that are needed
   cg = np.cos(gamma)
   sg = np.sin(gamma)
   cb = np.cos(beta)
   ca = np.cos(alpha)
   
   #2a. make life easier   
   W = (1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg)**0.5
   #Omega is cell volume
   Omega = a*b*c*W

   #3. the non-zero elements of ap as required
   Ap = np.zeros((3,3))  
   Ap[0,0] = a 
   Ap[1,0] = b * cg
   Ap[1,1] = b * sg
   Ap[2,0] = c * cb    
   Ap[2,1] = c *(ca - cb * cg)/sg
   Ap[2,2] = Omega / (a * b * sg) 

   if Ap_only:
      return Ap
   
   try: 
      assert not np.isnan(Ap).any()
   except:
      print("ERORR IN AP VALUE!!!")
      print("abc is", a, b, c)
      print("alph, beta, gama is", alpha, beta, gamma)
      """
      print("Ap is", Ap)
      print(Omega, a, b, sg) 
      print("W is", W)
      print(ca, cb, cg, ca, cb, cg)
      print(180/np.pi * np.array(ci[:3]), ci[3:])
      """
      exit()


   #4. now calculate the derivatives of Ap with ci
   #returns list of 6 3x3 matrices
   dA22_dalpha = c/sg* np.sin(alpha)*(ca-cb*cg)/W
   dA22_dbeta = c/sg* np.sin(beta)*(cb-ca*cg)/W
   dA21_dgamma = (c/sg) *(-ca*cg/sg + cb/sg)
   dA22_dgamma = c/W * (-ca*cb + cg - (cg/sg**2)*W**2)

   dAp_a = np.array([[1,0,0],[0,0,0],[0,0,0]])
   dAp_b = np.array([[0,0,0],[cg, sg, 0],[0,0,0]])
   dAp_c = np.array([[0,0,0],[0,0, 0],[cb,(ca-cb*cg)/sg, W/sg]])
   dAp_alpha = np.array([[0,0,0],[0,0,0],[0, -c*np.sin(alpha)/np.sin(gamma),dA22_dalpha]])
   dAp_beta = np.array([[0,0,0],[0,0,0],[-c*np.sin(beta), c*np.sin(beta)*cg/sg, dA22_dbeta]])
   dAp_gamma = np.array([[0,0,0],[-b*sg,b*cg,0],[0, dA21_dgamma, dA22_dgamma]])
   
   dAp = [dAp_alpha, dAp_beta, dAp_gamma, dAp_a, dAp_b, dAp_c ]


   #5. also want the derivatives of Omega with respect to the ci 

   dOs= [ a*b*c*np.sin(alpha)*(ca-cb*cg)/W, a*b*c*np.sin(beta)*(cb-ca*cg)/W,  a*b*c*sg*(cg-ca*cb)/W]
   dOs += [b*c*W, a*c*W, a*b*W]
   

   return Ap, dAp, dOs


def check_cell_gen(Q,p,k,T, cell):
   """checks that the generation of the cell is working!"""
   c = np.dot(Q,p) + k
   """
   print("check_cell_gen")
   print("angles are", 180/np.pi * cell.angles)
   print("lengths are", cell.lengths)
   """   
   check_ci(c)

   #print("finished check_cell_gen")
   Ap, dAp, dOs = ci_to_Ap(c)
   A = np.dot(T,Ap)
   
   cell2 = UnitCell(cartBasis = A)
   cl = np.array(cell.lengths)
   cl2 = np.array( cell2.lengths)
   ca = np.array(cell.angles)
   ca2 = np.array( cell2.angles)   
   assert np.allclose(cl, cl2)
   assert np.allclose(ca, ca2)


def build_final_cell(vlist, vloc, atom_t_B, Q, k, T, old_cell, sr=1):
   """calculates the Energy = U + P.V"""
   #1. split vlist into at_vars and cell_vars
   n_cell_vars  = Q.shape[1]
   at_vars = vlist[:len(vlist)-n_cell_vars]
   p = vlist[len(at_vars):]
   #assert(len(p) + len(at_vars) == len(vlist))
   #print(at_vars, p)

   #2. calc cell and atom frac_pos from variables
   c = np.dot(Q,p) + k 
   Ap, dAp, dOs = ci_to_Ap(c) 
   A = np.dot(T, Ap)
   u = at_vars2u(at_vars, vloc, atom_t_B)
   
   #check what the distances are and print them out!  
   r_is = calc_r_is(u, A, sr, wrap_un = True )  

   #3.
   cell = UnitCell(cartBasis = A)
   #print("the ui are")
   for i,ui in enumerate(u):
      lab = old_cell.atoms[i].label
      key = old_cell.atoms[i].key
      tag = old_cell.atoms[i].tag
      cell.add_atom(label=lab, fracCoords = ui, key=key, tag=tag)
   #jpd47 changed this at 21:15 27.03.2020
   #cell.calc_props()
   return cell


def wrap_ti(vlist, vloc, atom_t_B, Q):
   """wraps the fractional coordinates by adjusting the ti
   -returns atBw, wrapped version
   -s, vector of si which are the shift vectors needed for the molecule bonds"""
   #1. construct the ui 
   n_cell_vars  = Q.shape[1]
   at_vars = vlist[:len(vlist)-n_cell_vars]
   u = at_vars2u(at_vars, vloc, atom_t_B)

   #2. loop
   atBw = []
   s = []
   uw = []
   for i,ui in enumerate(u):
      uiw = np.array(symm.wrap_fc(ui))
      uw.append(uiw)
      si = ui - uiw
      s.append(si)
      ti = atom_t_B[i][0]
      Bi = atom_t_B[i][1]  
      tiw = ti -si
      atBw.append([tiw, Bi])
   
   """
   #3. check that new atBw generates the uiw
   uw2 = at_vars2u(at_vars, vloc, atBw)
   for ui, ui2 in zip(uw, uw2):
      assert np.linalg.norm(ui-ui2) < 0.01
   """
   return s, uw,  atBw


def sort_equivs(equivs):
   """sorts the equivs so that for each set of equivalent atoms the lowest numbered atom
   is the representative"""
   es = {}
 
   for key,item in equivs.items():
      srt = sorted(item, key = lambda x: x[0])
      es[srt[0][0]] = srt
   asu_inds = sorted(list(es.keys()))
   return es, asu_inds  


def push_apart(cell, sg, min_seps, scan=False, dft=False, var_range=None, P=1, hyper = {}, target_atom_nums=None):
   verbose = False
   """converts min_seps to pair potentials
      converts cell to variables
      optimises using BFGS to minimise E + PV
      hyper_ds is number of hyper dimensions per atom"""  
   #print("pushing apart with pressure", P)
   pid = os.getpid()
   #print("pid is", pid)
   
 
   #standard variable assignment
   if len(hyper) != 0:
      success, cell = hyper_push_apart(cell, sg, min_seps, P, hyper) 
      return success, cell


   """FROM here on down is normal push_apart not using hyper dimensions"""
   #print("using min_seps", min_seps)
   #0. set image range and pressure
   sr = 1

   #1. initialise the variables
   #asu_inds = [[0,4],...] means 0 is index and has 4 symm equivalents
   success, vlist, vloc, atom_t_B, equivs, atom_wls = init_atom_vars(cell, sg)
   if not success:
      print("failed to initialise the atom_variables")
      return False, None
   equivs, asu_inds = sort_equivs(equivs)
  
   if verbose:
      """
      print("vlist is", vlist)
      print("vloc is", vloc)
      print("atom_t_B is", atom_t_B)
      print("equivs are", equivs)   
      print("asu_inds are", asu_inds)
      """
      for vs, loc, mats in zip(vlist, vloc, atom_t_B):
         print(loc, mats)

   check_assignments(cell, vlist, vloc, atom_t_B)
   Q, p, k, T = init_cell_vars(cell, sg)
   vlist += p
   check_cell_gen(Q, p, k, T, cell)

   #bounds = make_bounds(vlist, p,Q,k)
   #constraints = make_constraints(p,Q,k, vlist)
   #check_constraints(vlist, constraints)

   #2. wrap the atoms, at_t_B has ti modified, store the shifts
   s, u, at_t_B = wrap_ti(vlist, vloc, atom_t_B, Q) 

   #Try to set energy and gradient tolerances based on number of molecule and minsep bonds
   #bond type 1 is bonded, 2 is minseps. Allow overlaps over ~0.1AA
   #heuristic is to ALL bonds and 3 minseps per atom to be 0.1A deformed - issues if this goes all on 1 atom ofc
   Nmb, bmols = gen_mol_bonds(cell, s, u)
   E_bond_tol, E_bond_tol_grad = phi(1.1, 1, 1)
   E_ms_tol, E_ms_tol_grad = phi(0.9, 2, 1)

   Etol = Nmb*E_bond_tol + len(cell.atoms)*E_ms_tol
   gnorm_tol = abs(Nmb*E_bond_tol_grad) - len(cell.atoms) * E_ms_tol_grad
   #print(E_bond_tol, E_ms_tol, E_bond_tol_grad, E_ms_tol_grad)
   print("Nmb=", Nmb, "Etol=", Etol, "gtol=", gnorm_tol)

   ##NOTE temporary inclusion to test mergin
   if target_atom_nums is not None:
      bmols, used = aasbu.add_merge_bonds(cell, bmols, u, s, target_atom_nums, equivs, atom_wls, style="random_all")
   bms_dists, bms_types = gen_ms_bonds(cell, min_seps)
   msd = min_seps[1]
   
   #1. have a go using BFGS, switch to monotonic steepest descent if fails
   gnorm = gnorm_tol + 10
   E = Etol + 10
   ngoes = 0
   opt_dict = {"maxiter":50, "eps":10e-6}
   history = []
   gradient_history = []
   conv = False


   #sanity check on initial structure   
   H0, grad_0 = Energy_symm(vlist, vloc, atom_t_B, Q, k, T, bmols, bms_dists, bms_types, sr, P, asu_inds,equivs, debug=False)
   gnorm0 = np.linalg.norm(grad_0)
   n_cell_vars  = Q.shape[1]
   at_vars = vlist[:len(vlist)-n_cell_vars]
   p = vlist[len(at_vars):]
   c = np.dot(Q,p) + k 
   Ap, dAp, dOs = ci_to_Ap(c) 
   A = np.dot(T, Ap)
   V = abs(np.linalg.det(A))
   E0 = H0 - P*V

   if False:
      print("Initial structure has")
      print("E0=%f, H0=%f, gnorm0=%f" % (E0, H0, gnorm0))
      if E0 > 1000 * Etol:
         f_cell = build_final_cell(vlist, vloc, at_t_B, Q, k, T, cell, sr=sr) 
         write_res("fail_"+str(round(E0, 0)), f_cell) 
         write_res("fail_orig"+str(round(E0, 0)), cell) 
         print("initial cell has ludicrous energy")
         return False, f_cell         


   while ngoes < 100:
      ngoes += 1
      try:
         method = "BFGS"
         params_out = minimize(Energy_symm, vlist, args = (vloc, at_t_B, Q, k, T, bmols, bms_dists, bms_types, sr,P, asu_inds, equivs), method="BFGS", options=opt_dict, jac=True)
         H = params_out.fun  
         H_jac = params_out.jac
         gnorm = np.linalg.norm(H_jac)
         vlist = params_out.x     
      except TypeError:
         method = "SD"
         #vlist, H, gnorm, finished  = SD(vlist, Energy_symm,  fargs = (vloc, at_t_B, Q, k, T, bmols, bms_dists, bms_types, sr,P, asu_inds, equivs), step_method="line_search", N = 10, verbose=False)
         vlist, H, gnorm, finished  = SD(vlist, Energy_symm,  fargs = (vloc, at_t_B, Q, k, T, bmols, bms_dists, bms_types, sr,P, asu_inds, equivs), step_method="adaptive", N = 10, verbose=False)
      #nasty don't leave it like this!
      n_cell_vars  = Q.shape[1]
      at_vars = vlist[:len(vlist)-n_cell_vars]
      p = vlist[len(at_vars):]
      c = np.dot(Q,p) + k 
      Ap, dAp, dOs = ci_to_Ap(c) 
      A = np.dot(T, Ap)
      V = abs(np.linalg.det(A))
      E = H - P*V

      #print update 
      print_vars = [str(round(x, 1)) for x in [H, E, gnorm, V]]
      print_string = str(ngoes) + "\t" +method + "\t"
      max_len = max([len(x) for x in print_vars])
      for s in print_vars:
         while len(s) < max_len + 3: 
            s += " "
         print_string += s + "\t"

      if ngoes == 1:
         print("ngoes    method    H               E                   gnorm         V")
      print(print_string)

      #convergence
      history.append(E)
      gradient_history.append(gnorm)
    

      if len(history) > 5: 
         Etest = history[-1]
         gtest = gradient_history[-1]

         #check gradients
         print("checking tolerances...")
         if  gtest < gnorm_tol:
            conv = True
            print("passed!")
            break
         else:
            print("continuing")
         
         #check for flatline
         if abs(history[-4]-history[-1]) < 1e-3:
            print("flatlined")
            break

         #check for hopless large energy
         if len(history) > 10 and Etest > 100 *Etol:
            print("hopelessly large bond energy, break")
            break
               

         
      """
      if len(history) > 5: 
         if abs(history[-4]-history[-1]) < 1e-3:
            #print("flatlined")
            break
         else:
            conv = True
            for g in gradient_history[-4:]:
               if g > gnorm_tol/100:
                  conv = False
            if conv:
               pass
               #print("converged gradient", gradient_history[-4:])
      if conv:
         break
      """     
      if conv:
         break 

   if E < Etol  and  (gnorm < gnorm_tol or conv):
      f_cell = build_final_cell(vlist, vloc, at_t_B, Q, k, T, cell, sr=sr) 
      return True, f_cell
   else:
      f_cell = build_final_cell(vlist, vloc, at_t_B, Q, k, T, cell, sr=sr) 
      #write_res("fail_"+str(round(E, 0)), f_cell)
      print("exited but E and gradients did not pass")
      print("E=", E, "Etol=", Etol)
      print("gnorm=", gnorm, "gnorm_tol=", gnorm_tol)
      return False, None


def at_vars2u(at_vars, vloc, atom_t_B):
   """converts atom variables to fracional coordinates"""
   n_ats = len(atom_t_B)
   u = np.zeros((n_ats, 3))
   for i, tB in enumerate(atom_t_B):
      t = tB[0];  B = tB[1]
      if vloc[i] is None:
         u[i] = np.array(t)
      else:
         vs = [at_vars[j] for j in vloc[i]]  
         u[i] = np.array(t) + np.dot(B, vs)

   return u 


@jit(nopython=True, cache=False)
def calc_r_is(u, A, sr, wrap_un = False):
   """calculate the cartesian coordinates including shifts"""
   #print("calculating r_is")
   #print("u", type(u), u.dtype)
   #print("A", type(A), A.dtype)
   #print("sr", type(sr))

   
   shifts = []
   M = 2*sr +1
   r_is = np.zeros((len(u),M**3,3)) 
   for n in range(0,len(u)):
      un = u[n]
      if wrap_un:
         unw = np.array(wrap_fc(un))
         shifts.append(un-unw)
      else:
         unw = un
         shifts.append(np.array([0.0,0.0,0.0]))
      for i in range(-sr,sr+1):
         for j in range(-sr,sr+1):
            for k in range(-sr,sr+1):
               ind = M**2*(i+sr)+M*(j+sr)+(k+sr)
               r_is[n, ind] = np.dot(A.T, unw + np.array([i,j,k])) 

   #NOTE r_is[i] + np.dot(A.T, shifts[i]) gives unwrapped position
   return r_is, shifts

@jit(nopython=True, cache=False)
def wrap_fc(fp):
   tol = 10**-5
   fp2 = [x%1 for x in fp]

   for i in range(0,3):
      if fp2[i] > 1-tol:
         fp2[i] =0
   return fp2


@jit(nopython=True, cache=False)
def calc_d_ijk_symm(r_is, sr, asu_inds):
   """calculate the cartesian coordinates including shifts"""
   #print("calculating d_ijs")
   #print("r_is", type(r_is), r_is.dtype)
   #print("sr", type(sr))

   M = sr*2 + 1
   nk = M**3
   zk = int((M**3-1)/2)
   d_ijk = np.zeros((len(r_is),len(r_is),M**3)) 
   #1. loop 1 is over atoms in asu
   for i in asu_inds:
      for j in range(0, len(r_is)):
         for k in range(0, nk):
            d_ijk[i,j,k] = np.linalg.norm(r_is[i,zk]-r_is[j][k])
   return d_ijk


@jit
def phi_symm(d, bt, bd):
   """calculate the energy"""
   k = 3000
   K = 10*k
   if bt == 2:
      #minsep
      if d > bd:
         return [0.0, 0.0]
      else:
         return [K* (bd - d)**4, -4*K*(bd-d)**3]
   elif bt == 1:
      #bonded
      return [k * (bd-d) ** 4, -4*k*(bd - d )**3]
   else:
      #ignore
      return [0.0, 0.0]


def convert_volume_derivatives(dOs, Q):
   """convert the derivatives of the volume with respect to the 
   ci to derivatives with respect to the actual cell variables"""
   dOm_dvc = []
   n = Q.shape[1]
   for k in range(0,n):
      t = 0
      Qi_k = Q[:,k]
      for i, qik in enumerate(Qi_k):
         t += dOs[i]*qik
      dOm_dvc.append(t)
   return dOm_dvc


def Energy_symm(vlist, vloc, atom_t_B, Q, k, T, bmols, bms_dists, bms_types, sr, P, asu_inds,equivs, debug=False):
   """calculates the energy using a fully symmetrised loop over atoms.
      test version: no molecules and no derivatives"""

   #1. split vlist into at_vars and cell_vars
   n_cell_vars  = Q.shape[1]
   at_vars = vlist[:len(vlist)-n_cell_vars]
   p = vlist[len(at_vars):]

   #2. calc cell and atom frac_pos from variables
   c = np.dot(Q,p) + k 

   Ap, dAp_dci, dOs = ci_to_Ap(c) 
   det_T = np.linalg.det(T)
   #NOTE needs to be det T here
   dOs = np.array([x*abs(det_T) for x in dOs])
   dOm = convert_volume_derivatives(dOs, Q)
   A = np.dot(T, Ap)
   ATinv = np.linalg.inv(A.T)
   vol = abs(np.linalg.det(A))
   u = at_vars2u(at_vars, vloc, atom_t_B)
   nats = len(u)
   
   #3. calculate the r_is and the d_ijs for minseps only
   r_is, shifts = calc_r_is(u, A, sr, wrap_un=True)  
   d_ijk = calc_d_ijk_symm(r_is, sr, asu_inds)  

   #5. calculate the phi_ijs for the minseps and the dU_dr
   M = 2*sr + 1
   nk = M**3
   zk = int((M**3-1)/2)
   nij = symm_nij(nats, asu_inds, equivs)
  
   dU_dr = [np.array([0.0,0.0,0.0]) for i in range(0,nats)] 
   dU_dsk = [np.array([0.0,0.0,0.0]) for i in range(0, nk)]
   

   #minseps loop 
   Utot, dU_dr, dU_dsk = loop_sym_energy(dU_dr, dU_dsk,asu_inds, nats, nk, zk,r_is, d_ijk, bms_dists, bms_types, nij)

   #bonding loop
   Utot, dU_dr, dU_dsk = loop_bond_energy(dU_dr, dU_dsk,asu_inds, nats, nk, zk,r_is, d_ijk, bms_dists, bms_types, nij, Utot, bmols, M, A, shifts)


   #6. calculate the dr_dva, derivatives of cartesian position with respect to atom vars
   dr_dVa = calc_dr_dVa(A, atom_t_B, vloc, at_vars, asu_inds)
   
   if debug:
      print("found atomic var derivs")

   #7. calculate the dr_dvcs, derivatives of cartesisan position with respect to cell vectors 
   dAp_dVc = calc_dAp_dVc(dAp_dci, Q)
   assert len(dAp_dVc) == n_cell_vars
  
   dvc_matrix = [np.dot(np.transpose(x), np.dot(np.transpose(T), ATinv)) for x in dAp_dVc]
   dr_dVc = [[np.dot(x,rj) for x in dvc_matrix] for rj in r_is[:,zk]]
   #check_ops(r_is, asu_inds, equivs, Qs, dr_dVc, dr_dVa)
  
   if debug:
      print("finished with zero shift cell derivs")
   rshifts = get_r_shifts(sr, A) 
   drs_dVc = [[np.dot(x,rs) for x in dvc_matrix] for rs in rshifts]


   if debug:
      print("finished cell var derivs")
 
   #NOTE that this won't be correct at the moment because we are ignoring shifts for the derivatives
   #8. combine all 3 into dU_dv
   """
   print("r_is[:,zk] are")
   for ri in r_is:
      print(ri[zk], np.dot(ATinv, ri[zk]))
   """
   dU_dv = combine_derivs(dU_dr, dr_dVa, dr_dVc, drs_dVc, dU_dsk, asu_inds, equivs)      
   if debug:
      print("finished combing derivatives")
   assert len(dU_dv) == len(vlist)


   #7. calculate the pressure contribution and the modified derivatives
   H = Utot + P*vol

   dH_dv = []
   for i in range(0,len(dU_dv)):
      if i < len(at_vars):
         dH_dv.append(dU_dv[i])
      else:
         dH_dv.append( dU_dv[i]+ P*dOm[i-len(at_vars)])

   return H, np.array(dH_dv)


def calc_dr_dVa(A, atom_t_B, vars_loc, at_vars, asu_inds):
   """calculates the derivates of the cartesian positions with respect to the atom variables"""
   dr_dVa = []
   At = np.transpose(A)
   for i, tB in enumerate(atom_t_B):
      ti = tB[0]
      Bi = tB[1]  
      dri_dV = [np.array([0.0,0.0,0.0]) for k in range(0,len(at_vars))]  
      if Bi is not None:
         ABi = np.dot(At, Bi)
         #dri_dV = [np.array([0.0,0.0,0.0]) for k in range(0,len(at_vars))]  
         vinds = vars_loc[i]
         for n, vi in enumerate(vinds):
            ABin = ABi[:,n]
            dri_dV[vi] += ABin
      dr_dVa.append(dri_dV)
   #NOTE dr_dVa[i][j] is a 3 vector which shows how ri varies with vat_j
   return dr_dVa


def calc_dAp_dVc(dAp_dci, Q):
   """converts the derivatives of the conventional cell with respect to the 
   ci into the erivatives with respect to the variables"""
   n_cell_vars  = Q.shape[1]
   dAp_dVc = []
   for i in range(0,n_cell_vars):
      t = []
      for a in range(0,3):
         row = []
         for b in range(0,3):
            count = 0
            for j in range(0, len(dAp_dci)):
               count += dAp_dci[j][a,b]*Q[j,i]
            row.append(count)
         t.append(row)
      dAp_dVc.append(np.array(t))
   return dAp_dVc


def combine_derivs(dU_dr, dr_dVa, dr_dVc, drs_dVc, dU_dsk, asu_inds, equivs):
   """combine these using the chain rule to get the total derivatives"""
   dU_dv = []
   nats = len(dr_dVc) 
   #NOTE dr_dVa[i][j] is a 3 vector which shows how ri varies with vat_j
   #2. derivatives with respect to the atomic variables
   nat_vars = len(dr_dVa[0])
   for i in range(0,nat_vars):
      tot = 0         
      for j in range(0,nats): 
         tot += np.dot(dU_dr[j], dr_dVa[j][i])
      dU_dv.append(tot)


   #1.construct derivatives with respect to the cell variables
   ncell_vars = len(dr_dVc[0])
   for i in range(0, ncell_vars):
      tot = 0
      #contribution from modifying the rj
      for j in range(0,nats):
         dU_drj = dU_dr[j] 
         tot +=  np.dot(dU_drj, dr_dVc[j][i])
   
      #contribution from modifying the shift vectors
      for k, dsk_dVc in enumerate(drs_dVc):
         tot += np.dot(dsk_dVc[i], dU_dsk[k]) 
      dU_dv.append(tot)

   return dU_dv


def sk_to_k(sk_ef, M):
   """converts sk_ef to k"""
   k = int((sk_ef[0]+1)*M**2 + (sk_ef[1]+1)*M + sk_ef[2]+1)
   #print("sk_ef is", sk_ef, k)
   return k


def get_r_shifts(sr,A):
   """get the cartesian shift vectors"""
   r_shifts = []
   for i in range(-sr, sr+1):
      for j in range(-sr, sr+1):
         for k in range(-sr, sr+1):
            r_shifts.append(np.dot(A.T, [i,j,k]))
   return r_shifts


def loop_bond_energy(dU_dr, dU_dsk, asu_inds, nats, nk, zk, r_is, d_ijk, bms_dists, bms_types, nij, Utot, bmols, M, A, shifts):
   """add in the bonding energy and its derivatives"""
   #7. BONDING LOOP
   #bmols contains the shift and bond distance for the unwrapped coordinates
   #bmols[i][j] = [1, s[i]-s[j], d1] 
       
   ki = sk_to_k([1,0,0],M)
   kj = sk_to_k([0,1,0],M)
   kz = sk_to_k([0,0,1],M)
   #print("start of bonding loop")
   for i in asu_inds:
      ri = r_is[i] + np.dot(A.T, shifts[i])
      for j in range(i,nats):
         bonds = bmols[i][j]
         if bonds is None:
            continue
         for bond in bonds:
            #calculate an effective shift vector, subtract off minsep contribution if required
            sk_ef = -shifts[i] +shifts[j]-bond[1]
            rij_kef = r_is[i,zk]-r_is[j,zk]-np.dot(A.T,sk_ef)
            dij_kef = np.linalg.norm(rij_kef)
            
            #convert sk_ef into k
            k_ef = sk_to_k(sk_ef,M)  
            if k_ef >= 0 and k_ef < nk:
               #subtract off minsep contribution and add bond contribution to exisiting
               bd = bms_dists[i,j]
               bt = bms_types[i,j]
               U, Uprime = phi(dij_kef, bt, bd) 
               
               Ub, Ubprime = phi(dij_kef, bond[0], bond[2])
               Ut = Ub-U; Utprime = Ubprime - Uprime
               #add internal energy contribution
               Utot += nij[i,j] * Ut
               
               #add contribution to derivatives
               dif = nij[i,j]*Utprime/dij_kef*rij_kef

               #first term
               dU_dr[i] += dif
               
               #second term
               dU_dr[j] -= dif

               #contriubtion to shift derivatives
               #print("k_ef is", k_ef)
               try:
                  dU_dsk[k_ef] -= dif
               except:
                  print("major error", k_ef, i, j, sk_ef)
            else:
               #need to add on bond contribution but k_ef out of bounds... be sneaky
               #decompose into shifts within range and use linearity...
               #print("k_ef is", k_ef, sk_ef)
               #print("missing terms at the moment...")
         
               U, Uprime = phi(dij_kef, bond[0], bond[2])

               #add internal energy contribution
               Utot += nij[i,j] * U
               
               #add contribution to derivatives
               dif = nij[i,j]*Uprime/dij_kef*rij_kef

               #first term
               dU_dr[i] += dif

               #contriubtion to shift derivatives
               dU_dskef = -dif
               dU_dsk[ki] += sk_ef[0]*dU_dskef
               dU_dsk[kj] += sk_ef[1]*dU_dskef
               dU_dsk[kz] += sk_ef[2]*dU_dskef

   return Utot, dU_dr, dU_dsk


@jit(nopython=True, cache=False)
def loop_sym_energy(dU_dr, dU_dsk, asu_inds, nats, nk, zk, r_is, d_ijk, bms_dists, bms_types, nij):
   """main loop in symmetric energy, factorised out to make this but numbarisable"""
   Utot = 0
   #6. NON-BONDING LOOP
   for i in asu_inds:
      for j in range(i,nats):
         for k in range(0,nk):
            if i == j and k == zk:
               #same atom and no shift
               continue

            #call phi
            d = d_ijk[i,j,k]
            Us = phi_symm(d, bms_types[i,j], bms_dists[i,j])
            U = Us[0]; Uprime = Us[1]
            
            #add energy contribution
            Utot += nij[i,j]* U
            rijk = r_is[i,zk] - r_is[j, k] 

            #first derivative term, p is i
            dif = nij[i,j] * Uprime/d*rijk
            dU_dr[i] += dif
            
            #second derivatiVve term, p is j0
            dU_dr[j] -= dif
            #contriubtion to shift derivatives
            dU_dsk[k] -= dif

   return Utot, dU_dr, dU_dsk


def symm_nij(nats, asu_inds, equivs):
   """calculates the symmetrised ni nj"""
   nij = np.zeros((nats,nats))
   ns = []
   for key,item in equivs.items():
      for x in item:
         ns.append([x, len(item)])
   ns = sorted(ns, key = lambda x: x[0])
   for i in asu_inds:
      for j in range(0,nats):
         if j < i:
            continue
         elif j == i:
            nij[i,j] = ns[i][1]
         elif i < j < i + ns[i][1]:
            nij[i,j] = ns[i][1]/2
         else:
            nij[i,j] = ns[i][1]

   return nij

