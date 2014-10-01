Data
====

This directory contains all of the current data. It is broken down into two main groups (non-optimized structures and optimized structures).
Under the `opt` directory, there are three more directories corresponding to optimizing the structures with B3LYP, CAM-B3LYP, and M06HF respectively.

Each one of these sub folders contains 6 files, and 1 folder. The folder contains all of the final geometries used for this set of calculations in a simple `Elem X Y Z` format. The folders are then broken down into two types. The more raw `*.csv` files contain various properties parsed from the raw Gaussian log files. The `*.txt` files contain a cleaner set of data containing only structure names, HOMO, LUMO, and Band Gap energies. The names of each of these files indicates the final computational method that was used to calculate the properties.


	├── noopt
	│   └── geoms
	│   b3lyp.csv
	│   b3lyp.txt
	│   cam.csv
	│   cam.txt
	│   m06hf.csv
	│   m06hf.txt
	└── opt
	    ├── b3lyp
	    │   └── geoms
	    ├── cam
	    │   └── geoms
	    └── m06hf
	        └── geoms

The CSV files have the following columns (and units):

Full Path, Filename, Exact Name, Feature Vector, Method Used, HOMO (eV), LUMO (eV), HOMO Orbital Number (#), Dipole (au), Total Energy (eV), Band Gap (eV), Time To Calculate (Hours)

The Exact Name and Feature Vector name columns currently have no significance for these structures.


All of the structures in this data set are made up of conjugated systems of only Carbon, Hydrogen, and Oxygen. All of these structures are composed of 4 or fewer ring/aryl backbone parts. Many of the structures are the same structure with a 180 degree dihedral flip between two rings (denoted as a `-` in the name AFTER the ring it affects). These systems also have the restriction that all of the ring units are all the same through the chain.

