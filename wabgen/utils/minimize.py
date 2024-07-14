
"""."""

import numpy as np

def steepest_descent(x, f, fargs = [], fkwargs={}, N=10, gamma=1, verbose=False, jac=True, step_method = "adaptive", gnorm_tol = 1e-6):
   """steepest descent, 0 < gamma <= 1, controls step size""" 

   #keep appraches entirely separate to keep them "trim"
   gamma = 1e-1
   s = 1e-2
   failure = False

   if step_method == "line_search": 
      #0. main loop 
      for i in range(0,N):
         #1. get derivatives
         if jac:
            val, grad = f(x, *fargs, **fkwargs)
         else:
            #numerical derivatives, two shifts for now
            val, grad = num_wrap(x, f, 1e-5, *fargs, **fkwargs)
         gnorm = np.linalg.norm(grad)
         
         #2. check if converged
         print(i, val, gnorm)
         if gnorm < gnorm_tol:
            return x, val, gnorm, True

         #3. do a line search in the desired direction
         x = line_search(x, f, val, grad, *fargs, **fkwargs)
      return x, val, gnorm, False
         
         

   elif step_method == "adaptive":
      gnorms = []
      for i in range(0,N):
         #1. get derivatives
         if jac:
            val, grad = f(x, *fargs, **fkwargs)
         else:
            #numerical derivatives, two shifts for now
            val, grad = num_wrap(x, f, 1e-5, *fargs, **fkwargs)
         gnorm = np.linalg.norm(grad)
         
         #2. check if converged
         if gnorm < gnorm_tol:
            return x, val, gnorm, True

         #decreasing step sizes until the function decreases, forces monotonic convergence
         accept = False
         while not accept:
  
            xtrial = x - gamma/gnorm *grad
            try:
               if jac:
                  ftrial, grad_trial = f(xtrial, *fargs, **fkwargs)
               else:
                  ftrial =  f(xtrial, *fargs, **fkwargs)
               failure = False
            except:
               failure = True
               ftrial = val + 10
            if ftrial < val:
               accept = True
               x = xtrial
               gamma *= 2
               if verbose:
                  if jac:
                     print(i, "/", N, ftrial, np.linalg.norm(grad_trial), gamma) 
                  else:
                     print(i, "/", N, ftrial, np.linalg.norm(grad), gamma) 
            else:
               gamma *= 0.5
               if verbose:
                  print("decreasing gamma to", gamma)
               if gamma < 1e-10:
                  #gamm getting too small
                  return x, val, gnorm, not failure
      return x, val, gnorm, True
   
   elif step_method == "Barzali-Borwein":
      pass
