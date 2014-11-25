Predicting Chemical Properties Using Machine Learning Methods
=============================================================

Andrew ids: ccollin1, haichenl, zhonghal

Quantum mechanics allows us to predict chemical compounds' properties to a very high accuracy, given that we are able to solve certain nonlinear ill-conditioned partial differential equations. Because of their chaotic nature, these equations are extremely hard and expensive to solve. Linear approximations based on machine learning ideas have the potential to help us circumvent such dilemma; research done by astronomers/biologists have shown that linear methods can be effective in approximating nonlinear problems(CITE), and the corresponding mathematical tools for these methods have been exhaustively studied.

Our specific goal is to predict thermodynamic/spectroscopic properties of a given type of chemical compounds, taking the advantage that they have similar chemical compositions, structures, and behaviors. Empirically, molecules having similar functional groups will have similar properties, even if they differ greatly in size. For small size molecules, high accuracy quantum mechanics calculations can be relatively easily performed, but for large molecules it is typically impossible because most of the quantum mechanics methods scale with O(n^{4~7}) time where n is the number of atoms in a molecule (single structures with dozens of atoms might take on the order of hours to calculate). However, as stated previously, it is known that chemically similar compounds will have highly correlated properties. Thus, there is a chance that some machine learning based regression models can be used to exploit these correlations in a quantitative sense.

Data set:
---------
Our data will mainly come from quantum mechanics calculations through traditional ab initio and density functional theory methods. We will perform high accuracy calculations for small, yet characteristic, molecules to get their thermodynamic/spectroscopic properties. There are mature software packages available for doing quantum mechanics calculations; typically these packages give out text log files as output, and our data pre-processing will be parsing these files to get the numerical values of our desired properties. For our structures we will be looking at conjugated polymeric systems consisting of just Carbon, Hydrogen, Oxygen, and Nitrogen (phenyl, furan, pyrrol, pyridine, vinyl, acetylene).

Performance validation:
-----------------------
Validation will be simply done by comparing numerical values generated by our regression model with those generated by quantum mechanics calculations. From a machine learning perspective, this will be a supervised learning where the expected values are the values calculated using density functional theory. Specifically, for properties, we will be looking at the highest occupied molecular orbital (HOMO), the lowest unoccupied molecular orbital (LUMO), band gap, and total energies.

first step, milestones, midway report:
--------------------------------------
The first step would be to implement methodologies represent the chemical structures such that they could be used for machine learning. After this, a small set of the dataset would be used to determine whether this method would be suitable for predicting molecular properties. If all goes well, this data set would be expanded to include more structures and/or more properties.

minimum to achieve:
-------------------
The minimum goal would be to a certain degree of certainty be able to predict molecular properties of structures similar to the ones from our data set.


Long term goal:
---------------
Chemists have well-defined classifications of chemical compounds, and it has been determined, experimentally, that each class of compounds share a lot of common properties. Eventually, we hope that our method can be applied to every class of chemicals so that for every newly proposed chemical structure we can predict its crucial properties (therefore its potential usage) cheaply and accurately. This will enable computers to help chemists design new materials and drugs far more efficiently.





Usage
-----

	git clone https://github.com/crcollins/ml-class
	# Install the required dependencies
	pip install -r requirements.txt
	sudo apt-get install python-rdkit librdkit1 rdkit-data
	# Since the data is now in a submodule, it requires one extra step to get it loaded when you clone/pull the repo.
	git submodule init
	git submodule update
	# Now run the test script
	python example.py
	# Or run the neural net test script
	python neural.py





