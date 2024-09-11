

import os
import json
import gzip


directory = os.path.dirname(__file__)


def write_db(dic, fname):
    """Take in a dictionary and writes it to a gzipped database file."""
    fname = fname.replace(".gz", "")
    dbstr = json.dumps(dic)
    with open(fname, "w") as f:
        f.write(dbstr)
    os.system('gzip -f ' + fname)


def read_db(fname):
    """Read in a json database object and returns a dict."""
    fname = fname.replace(".gz", "") + ".gz"
    f = gzip.open(fname, 'rb')
    dbstr = f.read().decode("utf-8")
    f.close()

    db = json.loads(dbstr)
    return db


def get_atomic_data():
   """reads in the atomic radii, covalent radii and atomic weights
   returns dictionary where keys are atomic symbols e.g. "Na" for sodium"""
   path = os.path.join(directory, "../data")

   crf = path + "/covalent_radii.dat"  # covalent radius
   weight_fname = path + "/atomic_weights.dat"
   metals_fname = path + "/metals.txt"
   #rf = "../../data/atomic_radii.dat"  #atomic radius

   #1. read in weights
   with open(weight_fname, "r") as f:
      data = []
      for line in f:
         data.append(line.split())
   #print("data is", data)

   info = {}
   for line in data:
      lab = line[1]

      w = line[3]

      w2 = ""
      for x in w:
         if x != "(" and x != ")" and x != "[" and x != "]":
            w2 += x
      info[lab] = {"Mr": float(w2)}


   #2. read in covalent radii
   with open(crf, "r") as f:
      for line in f:
         l = line.split()
         lab = l[0]
         r = float(l[1])
         if lab in info:
            info[lab]["rc"]=r
         else:
            info[lab] = {"rc":r}

   #3. add whether metal or not
   metals = []
   with open(metals_fname, "r") as f:
      for line in f:
         ls = line.split()
         lab = ls[1]
         metals.append(lab)
   for lab in info:
      if lab in metals:
         info[lab]["metal"] = True
      else:
         info[lab]["metal"] = False

   info["X"] = {"Mr": 1.0, "rc": 0.0, "metal": False}

   return info
