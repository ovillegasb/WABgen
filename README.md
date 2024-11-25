# Wycoff alignment block generator (WABgen)

*Orlando Villegas* - **2024**

Program adapted and modified from the WAM program built to generate crystals aligned crystals by aligning the atoms and molecules along the wycoff positions for a particular group space.

Planned to be used mainly to generate MOF.

This version keeps the same license (General Public License v3.0) and is modifiable. What I have done is to adapt it to my PostDoc work and share it with the community.

The main use I gave it was as a spacegroup MOF generator. Wycoff alignment block generator (WABgen).


## Installation

You can start from a virtual environment local:

```sh
python -m venv .venv --copies --prompt Wabgen
source .venv/bin/activate
pip install setuptools

git clone https://github.com/ovillegasb/WABgen.git
python setup.py install
```

## Pre-use

To display the help interface at all time:

```sh
wabgen -h
```

WABgen works with an input file (`system.cell`) that specifies the structural parameters.

```
block lattice_cart
10 0 0
0 10 0
0 0 10
%endblock lattice_cart

%block positions_abs
C   -0.0741692  -0.1681704  -0.0048599  #! group lig
C   1.3272307   -0.1682862  -0.0043715  #! group lig
C   2.0280313   1.0453036   -0.0048430  #! group lig
C   1.3274319   2.2590093   -0.0058050  #! group lig
C   -0.0739680  2.2591252   -0.0062929  #! group lig
C   -0.7747685  1.0455353   -0.0058213  #! group lig
H   1.8621539   -1.0949775  -0.0036395  #! group lig
H   3.0980312   1.0452152   -0.0044698  #! group lig
H   -0.6088912  3.1858164   -0.0070262  #! group lig
H   -1.8447685  1.0456238   -0.0061963  #! group lig
C   2.0975424   3.5926246   -0.0063264  #! group lig
C   -0.8442797  -1.5017856  -0.0043414  #! group lig
O   1.4469859   4.7196370   -0.0072232  #! group lig
O   3.3988423   3.5925170   -0.0058704  #! group lig
O   -2.1455796  -1.5016780  -0.0047969  #! group lig
O   -0.1937231  -2.6287980  -0.0034486  #! group lig
%endblock positions_abs

%block insert
1 lig groups
1 Fe atoms
%endblock insert

%block volume_distribution
numpy.random.uniform low=145.0 high=265.0
%endblock volume_distribution

%block min_seps
Fe O C H
4.0 2.0 4.0 4.0
2.0 4.0 4.0 4.0
4.0 4.0 5.0 5.0
4.0 4.0 5.0 5.0
%endblock min_seps

%block unit_formular
1 2 3 4 5
%endblock unit_formular

%block density
15.0
%endblock density
```

## Example of uses

Generation of 100 structures activating the possibility of flexible structures, parallel generation is also activated here:

```sh
wabgen system.cell -n 100 --push_apart flexible --parallel -nc 4
```

This will look to generate 100 structures by randomly exploring the space groups.