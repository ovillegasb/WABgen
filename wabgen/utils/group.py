
"""."""

import copy
import numpy as np


def generate_group(gens):
   """Form group from generators, in form of matrices."""
   G = [copy.deepcopy(x) for x in gens]
   done = False
   while not done:
      new_els = [] 
      
      #1. check all elements
      for g1 in G:
         for g2 in G:
            g3 = np.dot(g1,g2)
            new = True
            for g in G+new_els:
               if np.allclose(g3,g):
                  new = False
                  break
            if new:
               new_els.append(g3)      

      #2. see if there was any change 
      if len(new_els) == 0:
         done = True
         print("finished with", len(G), "elements")
      else:
         print("added", len(new_els))
         G += new_els

   return G


def sl_orbit(l,G):
   """s is a set where the elements are permuted under G
      e.g. [0,1,3], could represent Mol A on sites 1,2,4
      example g [1,0,2,3] would change s to [1,0,3] which is a differnet
      ordering BUT same occupancy so want this to be allowed!"""
   #want multiply occupied sites to be noticed so use sorted lists instead
   sl = sorted(l, key = lambda x: x)
   sltup = tuple(sl) 
   stabiliser = []
   orbital = set()
   for g in G:
      sl2 = acton(sl,g)
      sl2s = sorted(sl2, key = lambda x: x)
      orbital.add(tuple(sl2s))
      if sltup == tuple(sl2s):
         stabiliser.append(g)      
   
   #2. check using orbit stabiliser theorem 
   assert len(G) == len(orbital) * len(stabiliser)    
   return orbital, stabiliser


def find_orbit(p, G):
   """find and return the orbit of p under G
      operators parsed as permutations of the wyckoff sites"""
  
   #1. p is list, G is list of operators as string permutations!
   #print("p is", p)
   #print("g0 is", G[0])

   orbital = set()
   ptup = tuple(p)
   stabiliser = []
   for g in G:
      p2 = acton(p,g)
      orbital.add(tuple(p2))
      if tuple(p2) == ptup:
         stabiliser.append(g)

   #2. use orbit stabiliser theorem to check that orders make sense
   assert len(G) == len(orbital)*len(stabiliser) 

   return orbital, stabiliser


def acton(p, g):
   """takes in an element as a string and applies the permutation g"""
   #want g as a list of indicies e.g. [0,1,2] is identity
   # but [1,0,2] swaps the first 2 elements.
   np = [g[i] for i in p]
   return np

