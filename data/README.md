Data
====

This directory contains all of the current data. It is broken down into two main groups (non-optimized structures and optimized structures).
Under the `opt` directory, there are three more directories corresponding to optimizing the structures with B3LYP, CAM-B3LYP, and M06HF respectively. (All of these calculations are done with the 6-31g(d,p) basis set.)

Each one of these sub folders contains 10 files, and 1 folder (The noopt dir does not currently have the INDO results due to a bug). The folder contains all of the final geometries used for this set of calculations in two different formats. The first (`out`) is a a simple `Elem X Y Z` format. The second (`mol2`) is a more complex format that includes the bond information. This second format is used for the fingerprint feature vector. The folders are then broken down into two types. The more raw `*.csv` files contain various properties parsed from the raw Gaussian log files. The `*.txt` files contain a cleaner set of data containing only structure names, HOMO, LUMO, and Band Gap energies. The names of each of these files indicates the final computational method that was used to calculate the properties. 

The indo_x.txt files correspond to INDO with (default parameters, parameters fit to B3LYP, parameters fit to CAM-B3LYP, parameters fit to M06HF). NOTE: The INDO data sets are 1 datapoint smaller than the other datasets because for some reason `11` does not work with INDO.


	├── noopt
	│   └── geoms
	│   b3lyp.csv
	│   b3lyp.txt
	│   cam.csv
	│   cam.txt
	│   m06hf.csv
	│   m06hf.txt
	│   indo_default.txt
	│   indo_b3lyp.txt
	│   indo_cam.txt
	│   indo_m06hf.txt
	└── opt
	    ├── b3lyp
	    │   └── geoms
		│    ├── mol2
		│    └── out
	    │   ...
	    ├── cam
	    │   └── geoms
	    │   ...
	    └── m06hf
	        └── geoms
	        ...


The CSV files have the following columns (and units):

Full Path, Filename, Exact Name, Feature Vector, Method Used, HOMO (eV), LUMO (eV), HOMO Orbital Number (#), Dipole (au), Total Energy (eV), Band Gap (eV), Time To Calculate (Hours)

The Exact Name and Feature Vector name columns currently have no significance for these structures.


The TXT files have the following columns (and units):

Structure Name, HOMO (eV), LUMO (eV), Band Gap (eV)


All of the structures in this data set are made up of conjugated systems of only Carbon, Hydrogen, and Oxygen. All of these structures are composed of 4 or fewer ring/aryl backbone parts. Many of the structures are the same structure with a 180 degree dihedral flip between two rings (denoted as a `-` in the name AFTER the ring it affects). These systems also have the restriction that all of the ring units are all the same through the chain.

