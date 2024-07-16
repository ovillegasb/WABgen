
"""."""

import copy
import re
import numpy as np
import wabgen.core
import wabgen.utils.symm as symm
from wabgen.utils.cell import UnitCell
import sympy
from sympy.parsing.sympy_parser import parse_expr


def solve(frac, simfrac):
   """parsed fractional coordinates and symbolic expresssions
   solve for the symbols and return them"""
   #print("frac and simfrac are", frac, simfrac) 

   us = []
   null = parse_expr("0", evaluate=True)

   vars_in = []
   for i,x in enumerate(simfrac):
      t = x+"-"+str(frac[i])
      t = parse_expr(t,evaluate=True) 
      us.append(t)
      if "x" in x or "y" in x or "z" in x:
         vars_in.append(i)
  
   ustring = str(us)
   x, y, z = sympy.symbols('x y z ')      
   
  
   vs = []
   if "x" in ustring:
      vs.append(x)
   if "y" in ustring:
      vs.append(y)
   if "z" in ustring:
      vs.append(z)
   vs = tuple(vs)   
   
   us = pick_eqns(vs, us) 
   #print("us are", us)
   #print("vs are", vs)
   return vs, sympy.linsolve(us,vs)


def group_atoms(cell, sg):
   """groups the atoms into symmetry related pairs using operators of the sg"""
   #operators in sg.operators should relate the fractional coordinates

   #NOTE TODO this number should be related to symmetry tolerance on the molecule and cell size
   atol = 10**-3
   rtol = 10**-3

   Id = wabgen.core.Operator(0, np.identity(3))
   equivs = {}
   for i, ati in enumerate(cell.atoms):
      if i in equivs.keys():
         continue
      match = False
      uiw = wabgen.core.wrap_coords([x for x in ati.fracCoords])
      for j in equivs.keys():
         atj = cell.atoms[j]
         ujw = wabgen.core.wrap_coords([x for x in atj.fracCoords])
         for op in sg.operators:
            ouiw = wabgen.core.apply_op(op, uiw)
            if np.allclose(ouiw, ujw, atol=atol, rtol=rtol):
               match = True
               equivs[j].append([i, op])
               break
         if match:
            break
      if not match:
         equivs[i] = [[i, Id]]
  
   #create the useful quantities
   asu_inds = sorted([x for x in equivs.keys()])  
   eas = []
   for i in asu_inds:
      eas.append([x[0] for x in equivs[i]])
   return eas, equivs 


def eqn2var_vec(eqn, const=False):
   #print("eqn is", eqn, type(eqn))
   seqn = str(eqn)
   vs = ["x", "y", "z"]

   #1. find constant
   beqn = seqn
   for v in vs:   
      beqn = re.sub(v, "0", beqn)
   base = eval(beqn)

   #2. find covec
   covec = []
   for v in vs:
      ceqn = seqn
      other = [x for x in vs if x != v]
      ceqn = re.sub(v,"1",ceqn)
      for o in other:
         ceqn = re.sub(o,"0",ceqn)
      e = eval(ceqn)
      covec.append(e-base)

   if const == True:
      return covec, base

   return covec   


def pick_eqns(vs, us):
   """pick which equations to solve to avoid the wrapping issue e.g. [x,-x,z]"""
   #print("vs are", vs)
   #print("us are", us)
   u_vecs = []   
   for u in us:
      u_vec = eqn2var_vec(u, vs) 
      u_vecs.append(u_vec)

   #print("u_vecs are", u_vecs)
   
   #1.
   n = len(vs)
   eqns = []
  
   for j,u in enumerate(us):     
      match = None
      #ignore expressions that are just constants, nothing to solve
      if np.linalg.norm(u_vecs[j]) < 10**-4:
         continue

      for i,eq in enumerate(eqns):
         ind = eq[0]
         eqn = eq[1]
         #print("j is", j, "u is", u)
         t = np.cross(eqn, u_vecs[j])
         if np.linalg.norm(t) <  10** -4:
            match = i

      if match is None:
         eqns.append([j, u_vecs[j], u])
      else:
         #want smaller coefficents to help with wrapping e.g 2y =0.2, y= 0.6
         if np.linalg.norm(eqns[match][1]) > np.linalg.norm(u_vecs[j]): 
            eqns[match] = [j, u_vecs[j], u]
   #2. convert to eqns only..
   u_li = [x[2] for x in eqns]   #now contains linearly independent eqns
   #print("u_li is", u_li)
   

   #3. check that number of linearly independent equations matches number of variables
    
   if len(u_li) == len(vs):
      return u_li
   elif len(u_li) > len(vs):
      return u_li[:len(vs)]
   else:
      print("more variables than linearly independent variables...")
      exit()


   #TODO this doesn't handle some cases very well at all!! in particular
   #y-z = a, z-y = 1-a, y+z = b is solveable but this doesn't think it is!!
   #NOTE decide if filtering out linearly dependent combinations of variables
   #is a good idea... hint it fucking well will be you cretin!!   



 
   #1 need same number of equations as variables and all variables to be in equations
   u2s = []
   varz = set()
   for i in range(0,3):
      if len(u2s) == len(vs):
         break
      ustr = str(us[i])
      ui_vars = set()
      if "x" in ustr:
         ui_vars.add("x")
      if "y" in ustr:
         ui_vars.add("y")
      if "z" in ustr:
         ui_vars.add("z")
      """ 
      if ui_vars.issubset(varz):       #this doesn't handle x+y=1, x-y=0, 0.5=0.5 say
         continue
      """
      if len(ui_vars) == 0:
         continue 
      #replacement condition
      new_varz = set([x for x in varz])
      for x in ui_vars:
         new_varz.add(x)
      if len(u2s) +1 > len(new_varz):
         #would have more equations than variables
         continue  
      
      else:
         u2s.append(us[i])
         for v in ui_vars:
            varz.add(v)   

   #print("u2s are", u2s)
   return u2s


def test_match(fpos, fsym, vs, sol, debug=False):
   """test if the fsym and solutions matches the actual position"""
   #print("fpos is", fpos,"\n", fsym,"\n", vs,"\n", sol)
   
   #1. convert variables to symbols in sympy
   #2. sub in for variables and determine the numerical expression for subbing in
   #3. compare to eh current ones and try to match each by +-1
   #4 if matchable, return True, with the modified strings for the expressions
      #else return false

   if debug:
      print("fpos:", fpos)
      print("fsym:", fsym)
      print("vs:", vs)
      print("sol:", sol)
   
   #probably need to use np.allclose() or something
   varz = []
   #for v in vs:
   #   print(v)
   for v in vs:
      var = sympy.symbols(str(v))
      varz.append(var) 
   
   #print("varz are", varz)  
   dif = [] 
   for i in range(0,3):
      t = fsym[i]+"-"+str(fpos[i])
      #print("t is", t)
      t = parse_expr(t,evaluate=True)
      #loop over variables in sol substituting for them
      for j, v in enumerate(varz):
         t = t.subs(v, sol[j])

      #print("after subbing t is", t)
      dif.append(float(t))

   #now have list of differences
   wrap_dif = np.array([x%1 for x in dif])
   wrap_dif = np.array([min([abs(x), abs(1-x)]) for x in wrap_dif])
   #wrap_dif = np.array([0 if x > 1-tol else x for x in wrap_dif])
   #print("wrap dif is", wrap_dif) 
  

   #NOTE TODO this number should be related to symmetry tolerance on the molecule and cell size
   atol = 10**-3
   rtol = 10**-3
 
   if np.allclose(wrap_dif,np.array([0,0,0]), atol=atol, rtol=rtol):
      #now have a match with wrapping!!
      new_sym = [x+"-"+str(round(dif[i])) for i,x in enumerate(fsym)]
      
      #print(dif, wrap_dif)
      #print("found a match",new_sym)
      return True, new_sym

   #if here then don't have a match
   return False, []


def init_cell_vars(cell, sg):
   """takes in a cell and the corresponding spacegroup class
   initialises the cell variables and constructs the Q matrix for the
   dependency of primitive cell variables a b c alpha beta gamma on the 
   unrestricted conventional cell variables u v w theta phi chi"""
   """
   print("init_cell_vars cell is")
   print("angles are", 180/np.pi * cell.angles)
   print("lengths are", cell.lengths)
   """

   #1. find out what the centering is
   T = None 
   if "P" in sg.name:
      T = np.identity(3)
   else:
      Td = symm.T_dict     
      cens = Td.keys()
      for let in cens:
         if let in sg.name:
            T = Td[let]
   assert T is not None 

   #2. find the conventional cell
   pcell = copy.deepcopy(cell.cartBasis)
   Ti = np.linalg.inv(T)
   basis = np.dot(Ti, pcell)
   cell2 = UnitCell(cartBasis = basis)
   pvals = cell2.angles + cell2.lengths

   #3. get the variable restrictions     
   res = symm.restric[sg.system.lower()]

   #cs = Q.p + k
   vs = ["alpha", "bet", "gamma", "a", "b","c"]
   pnames = [v for v in vs if v not in res]
  
   p = []
   k = [0] * 6
   Q = []
      
   for i,v in enumerate(vs):
      q = [0] * len(pnames)
      if v in res:
         if type(res[v]) == type(1.21):
            k[i] = res[v]
         elif type(res[v]) == type("s"):
            #value determined by something else 
            q[pnames.index(res[v])] = 1
      else:
         q[pnames.index(v)] = 1
         p.append(pvals[i])
      Q.append(q)
   Q = np.array(Q)
   k = np.array(k)

   #4. check
   cs = np.dot(Q,p) + k
   for j,v in enumerate(vs):
      assert(abs(pvals[j]-cs[j]) < 0.001)
  
   return Q, p, k, T


def site_match(ats, wl, debug=False):
   """takes in a list of symmetry equivalent atoms
   wl = [site.letter, site.mult, site.rank, site symbolic fractional positions]
   and tries to solve for the positions of the atoms in terms of the wyckoff site
   variables"""
   #temp debugging
   if debug:
      print("wl is", wl)
      print("ats are")
      for at in ats:
         print(at.label, at.fracCoords) 

   
   vars_list = []
   vars_dict = {}
   
   #1. setup lists of the symbolic and numerical fractional coordinates
   sfps = wl[3]
   nfps = [at.fracCoords for at in ats] 
   nfp = nfps[0]
    
   #loop over trying to match nfp to one of the sfps  
   success = False
   f_sims = {}
   for i,sfp in enumerate(sfps):
      vs, sol = solve(nfp, sfp) 
      #use sympy to try and solve sfp==nfp for x,y,z
      if len(sol) > 0:
         sol = next(iter(sol))   #convert to list
         try:
            sol = [float(x) for x in sol]
         except:
            #solution contained variables to continue to next sfp
            continue 
      
      #if here then have a sympy solution from 1,2 or 3 of the equations from matching vectors
      #still need to check that the solution satisfies all of the equations though

      ok, new_sym = test_match(nfp,sfp, vs, sol)
      if debug:
         print("have a sympy solution...")
         print("nfp is", nfp)
         print("sfp is", sfp)
         print("vs are", vs)
         print("sol is", sol)
         print("ok is", ok)
         print("new_sym is", new_sym) 

      if not ok:
         #print("solution did not satisfy all of the equations")
         continue
      else:
         #update the sfps to allow for equality with a shift
         sfps[i] = new_sym     
         f_sims[0] = new_sym
      
      #if here then have a good solution, try to match up the other sites
      #shoudln't be possible to match the same sfp to differnet nfps
      for j,Nfp in enumerate(nfps):
         if j == 0:
            #skip nfps[0] == nfp
            continue
         for Sfp in sfps:
            ok, new_sym = test_match(Nfp,Sfp, vs, sol)

            if debug:
               print("testing_match...")
               print("ok is", ok)
               print("Nfp is", nfp)
               print("Sfp is", sfp)
               print("vs are", vs)
               print("sol is", sol)
               print("new_sym is", new_sym) 
            #print("\t",Sfp, ok)

            if ok:
               #matches this one and continue to the next one
               f_sims[j] = new_sym
               break
         if not ok:
            #print("didn't match... next allocation") 
            break
      if not ok:
         #couldn't match one of the others together so continue to next attempt
         continue 

      #if here then group finished correctly, ammend vars_list and vars_dict
      if len(vs) > 0:
         #update the list of variables
         for k,v in enumerate(vs):
            l = len(vars_list)
            vars_list.append(float(sol[k]))
            for at_ind in f_sims:
               if at_ind in vars_dict:
                  vars_dict[at_ind][str(v)] = l
               else:
                  vars_dict[at_ind] = {}
                  vars_dict[at_ind][str(v)] = l
         #update the expressions
         for key in f_sims:
            vars_dict[key]["exp"] = f_sims[key]
      success=True
      break
   
   if debug:
      print("success is", success)
      print("vars_list is", vars_list)
      print("vars_dict is", vars_dict)
   #wl[0] is the wyckoff letter
   return success, vars_list, vars_dict, wl[0]


def init_atom_vars(cell, sg):
   """takes in a cell and instance of the sg class and matches each atom to it wyckoff site and
   the associated variables"""
   #0. check the multiplicities of the sites
   for site in sg.sites:
      try:
         assert site.mult == 1 + len(site.other_frac_pos)
      except:
         print("\n\n\nsite error should never happen")
         print( sg.name, site.letter, site.mult, site.other_frac_pos)
         print(sg.number)
         

   #1. group the atoms onto the sites that they occupy
   eas, equivs = group_atoms(cell, sg)
   """
   print("eas are")
   pp.pprint(eas)
   print("equivs are")
   pp.pprint(equivs)
   """
   #eas, ws, equivs = group_atoms_onto_sites(cell, sg)
  
   #2. sort the wyckoff sites of the spacegroup in order of rank
   # and form a dictionary of their symbolic fractional positions
   site_sfp = []
   for site in sg.sites:
      sfps = [site.frac_pos]
      sfps += site.other_frac_pos
      temp = [site.letter, site.mult, site.rank, sfps]   
      #temp = [site.letter, 1+len(site.other_frac_pos), site.rank, sfps]   
      site_sfp.append(temp)

   #2a. start Bis, wis, vinds, and vlist
   vlist = []
   vloc = {}   #for each atom index where its variables are
   vt_bi = {}   

   #2b convert eas into list of asu atom indicies
   asu_inds = [[x[0], len(x)] for x in eas] 

 
   #3. loop over symmetry distinct atoms testing for matches
   """
   print("eas is", eas)
   print("asu_inds are", asu_inds)
   print("site_sfp is")
   pp.pprint(site_sfp)
   """
   atom_wls = {}
   for ea_list in eas:
      #find W's that have same multiplicity as len of ea_list
      wlist = [x for x in site_sfp if x[1] == len(ea_list)]
      wlist = sorted(wlist, key = lambda x: x[2])
      #loop over the wlist trying to match the atoms to the sites     
      ats = [cell.atoms[i] for i in ea_list]

      matched = False
      for wl in wlist:
         matched, vars_list, vars_dict, letter = site_match(ats, wl)
         if matched: 
            for i in ea_list:
               atom_wls[i] = letter 
            break
      if not matched:
         #error message written out to local log file
         print("solve pos error!!!")
         """TODO
         fname = "solve_pos_errors.txt"
         with open(fname, "a") as f:
            f.write("\n\n\n\n" + "coulnd't match up atoms to wyckoff sites\n")
            f.write(str(sg.number) +"\t"+ str(sg.name)+"\n")
            f.write("atoms at\t")
            for at in ats:
               f.write(at.label + "\t"+str(at.fracCoords) + "\n")  

            f.write("\n")
            f.write("wlist is\n")
            for x in wlist:
               f.write(str(x)+"\n")
            f.write("equivs are\n") 
            f.write(str(equivs) + "\n")
            for key, item in equivs.items():
               f.write(str(key) + ","+str(cell.atoms[key].fracCoords)+"\n")
               for x in item:
                  f.write(str(x[0]) + "," + str(cell.atoms[x[0]].fracCoords)+"\n")
               f.write("\n\n")
            f.write("wyckoff sites are")
            for site in sg.sites:
               f.write(str(site.frac_pos) + str(site.other_frac_pos))
            from wairss import write_res
            #write_res("solve_pos", cell) 
         """
         return False, None, None, None, None, None
  
      #4. if here then have solved for the atoms positions    
      #generate Bi, wi and var_inds for the atom, add to lists

      #same variables for all equivalent atoms so add to vlist now
      if len(vars_list) > 0:
         vs = sort_vars(vars_dict, vars_list)
         vinds = [len(vlist) +i for i,v in enumerate(vs)]
         vlist += vs
         for key in vars_dict:
            #key in vars_dict correspoinds to location in ats
            ti, bi = expr2matrix(vars_dict[key])
            at_ind = ea_list[key] 
            vt_bi[at_ind] = [ti, bi]
            vloc[at_ind] = vinds
      else:
         for at_ind in ea_list:
            vt_bi[at_ind] = [cell.atoms[at_ind].fracCoords, None]


   #5. as final step convert the dicts to lists ordered by atom index
   variable_location = []
   atom_t_B = []
   at_inds = sorted(vt_bi.keys())
   for ai in at_inds:
      if ai in vloc:
         variable_location.append(vloc[ai])
      else:
         variable_location.append(None)
      atom_t_B.append(vt_bi[ai])

   return True, vlist, variable_location, atom_t_B, equivs, atom_wls


def check_assignments(cell, vlist, vloc, atom_t_B):
   """checks the assignments of variables and representation"""
   for i,at in enumerate(cell.atoms):
      #calculuate position from init_vars
      if vloc[i] is not None:
         vs = [vlist[j] for j in vloc[i]]
         fp = np.array(atom_t_B[i][0]) + np.dot(np.array(atom_t_B[i][1]), vs)
      else:
         fp = np.array(atom_t_B[i][0])
      try:
         assert(np.linalg.norm(at.fracCoords-fp)) < 0.01
      except:
         print("fail", at.fracCoords, fp)
   return True 


def sort_vars(vars_dict, vars_list):
   """sort the variables into alphabetical order and return list"""
   fsim = vars_dict[list(vars_dict.keys())[0]]
   vs = [v for v in fsim.keys() if v != "exp"]
   vs = sorted(vs)

   #now have sorted variables into alphabetical order create ordered vlist
   vlist = [vars_list[fsim[v]] for v in vs]
   return vlist


def expr2matrix(fsim):
   """convert the expressions in fsim to matrices
      ti is always a 3 vector
      bi is a rectangular matrix, 3 x no. variables"""
   #1. convert to Bi and ti
   vecs = [eqn2var_vec(eqn, const=True) for eqn in fsim["exp"]]
   ti = [v[1] for v in vecs]
   Bi = np.array([v[0] for v in vecs])

   #delete columns from Bi that are empty
   bi = b = np.array([Bi[:,i] for i in range(0,3) if np.linalg.norm(Bi[:,i]) > 0.01]).T
   #print(fsim["exp"])
   #print(ti, Bi, bi) 
   return ti, bi
