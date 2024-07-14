
"""."""

import copy
import re
import itertools
import random
from collections import defaultdict
import numpy as np
import wabgen.utils.group as group



def int_ps(n, a_s):
   """generates all integer partitions of n using the a_s""" 

   #1. update the stack
   lst = []
   #2. iteratte
   for i, num in enumerate(a_s):
      if n-num == 0:
         lst += [[num]]
      else:
         if n -num > 0:
            l = int_ps(n-num, a_s[i:])
            for x in l:
               lst += [[num] + x]
   return lst


def convert_gen_opt(gen_opt):
   dic = dict()
   for i, tup in enumerate(gen_opt):
      dic[i] = list(tup)
   return dic


def convert_new_SA_to_old(new_SA, sg, PT, mols):
    #1. get the normaliser permutations
    norm_perms = sg.norm_perms
    if len(norm_perms) == 0:
       norm_perms = [[site.index for site in sg.sites]]

    SA = defaultdict(lambda: defaultdict(list))

    #zero sites first
    #same permutation for all zero sites as these may have been symmetry reduced already.
    perm = random.choice(norm_perms)
    dic = new_SA["zs_al"]
    for mol_ind, site_inds in dic.items():
        mol = mols[mol_ind]
        for site_ind in site_inds:
            si = perm[site_ind]
            site = sg.sites[si]
            if mol.Otype == "Mol":
                NR = PT[mol.symbol.ind][site.symbol.ind]
            else:
                NR = 1
            SA[mol_ind][si] += [np.random.randint(low=0, high=NR)]

    #now the other sites
    dic = new_SA["gen_al"]
    #differnt perm for each molecule ind as independently sym reduced.
    for mol_ind, site_inds in dic.items():
        mol = mols[mol_ind]
        print("mol is", mol)
        perm = random.choice(norm_perms)
        for site_ind in site_inds:
            si = perm[site_ind]
            site = sg.sites[si]
            if mol.Otype == "Mol":
                NR = PT[mol.symbol.ind][site.symbol.ind]
            else:
                NR = 1
            SA[mol_ind][si] += [np.random.randint(low=0, high=NR)]

    return default_to_regular(SA)


def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


def new_site_allocation(sg, mols, PT, zsa, msr={}, srr={}, max_options=1e3, max_extra_sites=200):
   """take in the dictionary of zero site allocations and find all symmetry
   distinct ways of finishing each allocation using the remaining sites"

   #TODO also parse max extra sites to single_mol_perms
   max_extra sites is the maxmimu number of additiona wyckoff sites to use over the minimum number
   """

   #sort the keys in zsa by number of molecules to add
   #multiplicity increases with dimensionality (incorrectly called "rank" in this code)
   #so having more moleculs left after zero site allocation is desirable from princpiple of parsimony
   #NOTE Wyckoff sets disrupts this idea a bit... e.g. P-1 add 8 atoms all to special sites is very symmetric
   #ignore this subtlety for now. Believe that build cell works on merging atoms from all being on general position
   #so presumably that would also find it very very hard to make configurations like that.
   temp = [[key, sum(zsk2nmols(key, mols))] for key in zsa]
   sorted_keys = [x[0] for x in sorted(temp, key=lambda x: -x[1])]

   #find out which sites which molcules can go on and form up the multiplicities
   gsite_dic = defaultdict(list)
   for i, mol in enumerate(mols):
      for site in sg.sites:
         if site.rank > 0 and site.mult <= mol.number:
               if mol.Otype == "Mol" and PT[mol.symbol.ind][site.symbol.ind] == 0:
                  continue
               gsite_dic[i].append(site.index)

   mults = [site.mult for site in sg.sites]
   #print("gsite_dic is", gsite_dic)
   #print("mults are", mults)

   #loop different zero site "keys" construcuting different ways of finishing the allocation
   #keys are "how many of each molecule was put on zero sites" e.g. (0,0) means no objects on zero sites
   all_perms = defaultdict(list)
   total = 0
   current_length = sum(zsk2nmols(sorted_keys[0], mols))
   all_mol_options = defaultdict(dict)
   max_unique_sites = None
   for key in sorted_keys:
      new_length = sum(zsk2nmols(key, mols))
      #print(f"key is {key}, and total is {total}")

      cond1 = new_length < current_length and total > max_options
      cond2 = total > 10 * max_options
      if cond1 or cond2:
        print("breaking at", total, "options")
        break
      nmols = zsk2nmols(key, mols)
      #print(f"\nnmols is {nmols} and new_length is {new_length}")
      for mol_ind, n in enumerate(nmols):
         #only form the molecule options if they haven't been formed already
         #this can be done independently for each molecule as the sites can be multiply occupied
         #NOTE single mol_perms are returned INDEPENDENTLY SYMMETRY REDUCED!!
         #THEY MUST BE combined expanded and then symmetry reduced again
         if n not in all_mol_options[mol_ind]:
               #print("generating single mol_perms for mol_ind", mol_ind, "n", n, "max_options", max_options)
               all_mol_options[mol_ind][n] = single_mol_perms(gsite_dic[mol_ind], mults, n, max_options=max_options, norm_perms=sg.norm_perms)
               #print("all_mol_options[mol_ind][n] is", all_mol_options[mol_ind][n])

      #check it's possible to allocate all molecules.
      nopts = [len(all_mol_options[mol_ind][n]) for mol_ind, n in enumerate(nmols) ]
      #print("nopts is", nopts)
      
      #NOTE bugfxed on 16.04.2024 - special case where all objects are on special wyckoff site was causing this to fail
      #solution was get single_mol_perms to return an empty list. 
      if 0 in nopts:
         print("skipping becuase no way of allocating all molecules")
         continue

      #now form appropriate combinations.
      for al in zsa[key]:
         #print("al is", al)
         mol_opts = [all_mol_options[mol_ind][n] for mol_ind, n in enumerate(nmols)]
         nu_z = sum([len(al[mol_ind]) for mol_ind in al])
         if nu_z > max_extra_sites:
            continue

         counter = 0
         for gen_opt in itertools.product(*mol_opts):
            #print("gen_opt is", gen_opt)
            counter += 1
            #count the number of sites used
            nu_q = sum([len(opt) for opt in gen_opt])
            nu = nu_z + nu_q
            if len(all_perms.keys()) > 0:
               max_unique_sites = min(all_perms.keys()) + max_extra_sites
            else:
               max_unique_sites = nu + max_extra_sites
            if nu > max_unique_sites:
                  continue
            all_perms[nu].append({"zs_al":al, "gen_al":convert_gen_opt(gen_opt)})

            #check how many options and delete the ones which use most sites
            if counter % 1000 == 0:
               total = sum([len(x) for x in all_perms.values()])

               if total > 10 * max_options:
                 print(f"breaking because have {total} options")
                 break
         if total > 10 * max_options:
                 break


      #all_perms +=  new_complete_allocation(zsa[key], all_mol_options, nmols, norm_perms=sg.norm_perms, max_options=max_options)
      total = sum([len(x) for x in all_perms.values()])
      current_length = new_length
      #for nu, temp in all_perms.items():
      #   print(f"nu is {nu}, len(temp) is {len(temp)}")
      #print(f"total is {total} and current_length is {current_length}")

   #NOTE prints out final deatils - looks very very sane... 44 atom cells works fine with all spacegroups
   if False:
      total = sum([len(x) for x in all_perms.values()])
      print('finishing with total', total, "ways, before symmetry expansion")
      for key, opts in all_perms.items():
         print(key, len(opts))
         for opt in opts:
            print(opt)

   return all_perms


def single_mol_perms(gsites, mults, n, norm_perms=[], max_options=1000):
    
    #bugfix for when n == 0
    if n == 0:
      return [()]
    
    mset = set([mults[site_ind] for site_ind in gsites])
    ips = int_ps(n, list(mset))
    ips = sorted(ips, key = lambda x: len(x))
    if len(ips) == 0:
        return []

    all_mol_perms = []
    for ip in ips:

        #mdic contains the number of times each multiplicity appears in the ip
        #site_dic contains the site indices for each multiplicity
        mdic = defaultdict(int)
        for m in ip:
            mdic[m] += 1
        site_dic = defaultdict(list)
        for site_ind in gsites:
            site_dic[mults[site_ind]].append(site_ind)

        #now form combinations of the sites for each multiplicity
        combo_dic = dict()
        for m, n in mdic.items():
            #print(f"m={m}, n={n}, site_dic[m]={site_dic[m]}")
            combo_dic[m] = list(itertools.combinations_with_replacement(site_dic[m], n))

        #now form the product of the combinations
        prod = list(itertools.product(*combo_dic.values()))
        #print("len of prod is", len(prod))

        #now add to list above
        #TODO symmetry reduce this list OR reduce once combined with other molceules
        #think the latter will be simpler, possibly faster AND the former is probably wrong....
        #NOTE: sneaky idea... Could independently symm reduce for each each molecule then
        #   - randomly pick from each molcules allocation list
        #   - randomly pick a (different from mol) normaliser perm and apply it to each molcule allocation
        #   - combine allocations, and apply another random perm to the entire alloction
        added = set()
        for x in prod:
            #sym_alloc is list of sites
            sym_alloc = list(itertools.chain.from_iterable(x))

            if len(norm_perms) > 0:
                #find orbitals and "min_image", use min_image so only have to check if one thing is in added.
                orbit, _ = group.find_orbit(sym_alloc, norm_perms)
                orbit = sorted(orbit)
                min_image = orbit[0]
                if min_image not in added:
                    added.add(min_image)
                    all_mol_perms.append(min_image)
            else:
                all_mol_perms.append(sym_alloc)

        if len(all_mol_perms) > max_options:
            break

    return all_mol_perms


def zsk2nmols(key, mols):
   """takes in a zero site allocation key and returns the
   mols that still need allocating"""

   s = re.sub(" ","",key)
   ls = s.split(",")
   return [mol.number - int(ls[i]) for i,mol in enumerate(mols)]


def zero_site_keys(nmols, gsites, mults, key=[]):
   """
   nmols is number of each molecule to add.
   gsites is list of allowed sites indices for each molecule
   mults is list of site multiplicities for every site in spacegroup
   """

   #call recursivley to get possible allocations
   lst = []

   #calculate the plausible numbers that can be added - TODO profile this section and find choke
   npos = find_sums([mults[x] for x in gsites[0]])
   nmax = min(nmols[0], sum(mults)-sum(key))
   npos = [n for n in npos if n <= nmax]

   if len(nmols) == 1:
      for n in npos:
         lst.append(key + [n])
      return lst

   #general case
   for n in npos:
      lst += zero_site_keys(nmols[1:], gsites[1:],mults, key + [n])
   return lst


def find_sums(ints):
   """find list of all possible sums from a set of ints
      includes the trivial case of 0.
      e.g. 1,2,8 would return 0,1,2,3,8,9,10,11"""
   s = set([0])
   for x in ints:
      lst = [x+y for y in s]
      for l in lst:
         s.add(l)
   #TODO consider whether this should be sorted for performance
   return s


def zero_site_allocation(sg, mols, pt, site_mult_restrictions={}, site_rank_restrictions={}):
   """returns a dict containing all zero site allocations of molecules
   for given spacegroup. key is number of each molcule"""
   #1-3a variant where all sites used and form full form of allocation is used

   gsites = []
   for mol in mols:
      gs = []
      for site in sg.sites:
         if site.rank > 0 or site.mult > mol.number:
            continue
         if mol.Otype == "Mol":
            if pt[mol.symbol.ind][site.symbol.ind] == 0:
               continue
         if mol.name in site_mult_restrictions:
            if site.mult != site_mult_restrictions[mol.name]:
               continue
         #added in site rank restrictions
         if mol.name in site_rank_restrictions:
            rs = site_rank_restrictions[mol.name]
            if site.rank > rs[1] or site.rank < rs[0]:
               continue
         #if here then site is a good one!
         gs.append(site.index)
      gsites.append(gs)
   mults = [site.mult for site in sg.sites]
   nmols = [mol.number for mol in mols]
   #gsites is list of allowed sites indices for each molecule
   #mults is list of site multiplicities for every site in spacegroup

   #4. create list of keys of possible zero site allocations
   zsks = zero_site_keys(nmols, gsites, mults)
   #print("zsks are", zsks)

   #4a. get list of permuations on full cell site indicies
   G = sg.norm_perms
   #G = []      #TODO change to this to see full effect of normaliser

   #5. for each number of mols to allocate, add all possibilities to dict
   zsas = {}
   total = 0

   for lkey in zsks:
      key = str(lkey)[1:-1]
      zsp = zero_site_perms(lkey, mults, gsites, G)
      total += len(zsp)
      zsas[key] = zsp
      #print("key is", key, "len(zsp) is", len(zsp))
      #print("zsp is", zsp)
   #print("total number of zsas is", total)
   return zsas


def zero_site_perms(lkey, mults, gsites, S,  perm=None, index=0):
   """return all permutations of allocation specified by key"""
   if perm is None:
      perm = {}
      for i,k in enumerate(lkey):
         perm[i] = []

   #final case when no more molecules to add
   if len(lkey) < 1:
      return [perm]

   #otherwise...
   #calculate which sites are good for the mol and unnocupied
   oc_sites = [s for mol_ind in perm for s in perm[mol_ind] ]
   asites = [s for s in gsites[0] if s not in oc_sites]
   if sum([mults[s] for s in asites]) < lkey[0]:
      #not enough good sites to allocate final mol to this perm
      return None

   if lkey[0] == 0:
      #don't add this mol so perm is unchanged, therfore stabiliser is also unchanged
      fperms = zero_site_perms(lkey[1:], mults, gsites[1:], S, copy.deepcopy(perm), index = index +1)
      return fperms

   #if here then will have perms
   fperms = []
   molperms = zs_mol_perms(asites, mults, lkey[0], S)
   if molperms is None:
      return None

   for mp in molperms:
      p2 = copy.deepcopy(perm)
      for s in mp:
         p2[index].append(s)
      #now need to find the stabiliser of p2 and parse that down...
      stab = full_allocation_stabiliser(p2, S)
      new_perms =  zero_site_perms(lkey[1:], mults, gsites[1:], stab, perm=p2, index = index +1)
      if new_perms is not None:
         fperms += new_perms
   return fperms  #returning a list of fully expanded permutations


def extra_el(l2,l):
   """idea is that l2 contains 1 extra element than l
   and want to return that
   e.g. l2 = [0,1,1,2], l=[0,1,2] want to return 1"""
   l2 = [x for x in l2]
   for x in l:
      try:
         i = l2.index(x)
         del(l2[i])
      except:
         return None
   assert len(l2) == 1
   return l2[0]


def zs_mol_perms(asites, mults, n, S, perm=[]):
   """returns all permutations of n mols across its available sites without
   multiple occupancy of the sites"""
   ps = []
   if n == 0:
      return [perm]

   if len(asites) == 0:
      return None

   else:
      #TODO insert the symmetry reduction in this loop here over sites...
      orbital = set()   #<-- add equivalent sites and then skip in list!
      for i,s in enumerate(asites):
         if s in orbital:
            continue
         if mults[s] <= n:

            s_orb, s_stab = group.find_orbit(perm +[s],S)
            #print(s_orb, perm+[s])
            for orb in s_orb:
               #add the extra element in perm + [s] thats not in perm to orbital
               """orbital.add(orb[-1])"""
               #23.05 modified below bit as well, print s_orb and perm+[s] to see why!
               el = extra_el(orb, perm)
               if el is not None:
                  orbital.add(el)
            #permutation is a list of sites inds
            #p2 = zs_mol_perms(asites[i+1:], mults, n-mults[s], s_stab, perm = perm +[s])
            #^^ 23.05 pass full stabiliser down for same objects
            p2 = zs_mol_perms(asites[i+1:], mults, n-mults[s], S, perm = perm +[s])

            #print("s_stab is",s_stab)

            if p2 is not None:
               ps += p2
      if len(ps) > 0:
         return ps
      else:
         return None

def full_allocation_stabiliser(fa, G):
   """full allocation has form
      {0:[1,1,2], 1:[0,3,4] ...}
      means mol 0 on sites: 1,1 and 2 etc"""

   #set stabiliser to all ops then loop over molcules finding
   #mutual stabililser of each allocation, shrinking stabiliser
   stab = G[:]
   for mol_ind in fa:
      l = fa[mol_ind]
      orb, stab = group.sl_orbit(l, stab)

   return stab
