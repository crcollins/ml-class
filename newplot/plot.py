import os

import matplotlib

import matplotlib.pyplot as plt
import numpy


def plot_property_set(calc, exp, title, names, indo, indo_exp, idx=1):
	plt.subplot(2,3,idx)
	calc_error = numpy.matrix([abs(x-y) for x,y in zip(exp,calc)])
	indo_error = numpy.matrix([abs(x-y) for x,y in zip(indo, indo_exp)])

	# norm_errors = (calc_error-calc_error.mean())/calc_error.std()

	# high_error_idx = numpy.where((norm_errors>2).tolist()[0])
	# high_error_names = names[high_error_idx].tolist()
	# high_error_vals = numpy.array(calc)[high_error_idx].tolist()
	# high_error_errors = calc_error[0,high_error_idx].tolist()[0]

	# low_error_idx = numpy.where(numpy.invert(norm_errors>2))
	# low_error_errors = calc_error[low_error_idx]

	calc_error_hist = numpy.histogram(calc_error,bins=10)

	plt.plot(exp, calc, '.', color="#2196F3", alpha=0.2, label="Neural Net")
	plt.plot(indo, indo_exp, '.', color="#F44336", alpha=0.2, label="INDO")

	plt.plot([min(exp), max(exp)], [min(exp), max(exp)], '-')
	plt.title(title)
	plt.legend(loc="best")
	plt.xlabel("DFT Calculated (eV)")
	plt.ylabel("Neural Net/INDO Calculated (eV)")

	plt.subplot(2,3,idx+3)
	bins = numpy.linspace(0, max(calc_error.mean()+calc_error.std()*1.5, indo_error.mean()+indo_error.std()*1.5), 50)
	plt.hist(calc_error.T, bins, alpha=0.5, facecolor="#2196F3", label="Neural Net")
	plt.hist(indo_error.T, bins, alpha=0.5, facecolor="#F44336", label="INDO")

	plt.xlabel("Mean Absolute Error (eV)")
	plt.ylabel("# Samples")
	plt.title("%s Error Distribution" % title)
	plt.legend(loc="best")

	# print "%s: %.4f +/- %.4f" % (title, calc_error.mean(), calc_error.std())
	# print "High Errors"
	# for x in zip(high_error_names, high_error_vals, high_error_errors):
	# 	print "    %s %.2f +/- %.2f" % x
	# print "After Dropping: %.4f +/- %.4f" % (low_error_errors.mean(), low_error_errors.std())
	# print


def main(calc_set, exp_set, names, indo_set, indo_exp_set, name=''):
	plt.clf()
	plt.figure(1, figsize=(11.4*2, 5.75*2))

	calc_homo, calc_lumo, calc_gap = zip(*calc_set)
	exp_homo, exp_lumo, exp_gap = zip(*exp_set)

	####
	indo_homo, indo_lumo, indo_gap = zip(*indo_set)
	indo_exp_homo, indo_exp_lumo, indo_exp_gap = zip(*indo_exp_set)
	####

	plot_property_set(calc_homo, exp_homo, "HOMO", names, indo_homo, indo_exp_homo, idx=1)
	plot_property_set(calc_lumo, exp_lumo, "LUMO", names, indo_lumo, indo_exp_lumo, idx=2)
	plot_property_set(calc_gap, exp_gap, "Band Gap", names, indo_gap, indo_exp_gap, idx=3)

	fig = plt.gcf()
	fig.suptitle("Neural Net Compared to INDO Results", fontsize=20)
	plt.show()
	# plt.savefig(name + ".png", dpi=150)


optsets = []
names = []
with open("Xpaths.txt", 'r') as f:
	for line in f:
		optset, name = line.strip().split()
		optsets.append(optset)
		names.append(name)

names = numpy.array(names)
optsets = numpy.array(optsets)

ypred = numpy.loadtxt("ypred.txt")
ydft = numpy.loadtxt("yDFT.txt")
idxs = numpy.loadtxt("indexes.txt").astype(int)


yindoidx = []
yindo = []
yindodft = []
drop = ["11","13ai13ai-13ai","13ai13ai13ai","13da13da-13da","13da13da13da","13ia13ia-13ia","13ia13ia13ia","13ad13ad-13ad","13ad13ad13ad","13fe13fe-4fe13fe","13ff13ff-13ff13ff-"]

# print zip(optsets[idxs].tolist(), names[idxs].tolist())
USE = dict(zip(zip(optsets[idxs].tolist(), names[idxs].tolist()), idxs))

for optset in ('b3lyp', 'cam', 'm06hf'):
	for atom in ('O', 'N'):
		with open("../mol_data/opt/%s/%s/indo_default.txt" % (optset, atom), 'r') as f:
			for line in f:
				temp = line.strip().split()
				pair = (optset, temp[0])
				if pair in USE:
					yindo.append([float(x) for x in temp[1:]])
					yindoidx.append(USE[pair])

main(ypred[idxs, :], ydft[idxs, :], names[idxs], yindo, ydft[yindoidx, :])

# drop = set(drop)
# pairs = []
# pairsdft = []
# for dataset in ('b3lyp', 'cam', 'm06hf', ):
# 	for optset in ('b3lyp', 'cam', 'm06hf'):
# 		for atom in ('O', 'N'):
# 			with open("../mol_data/opt/%s/%s/indo_default.txt" % (optset, atom), 'r') as f:
# 				for line in f:
# 					temp = line.strip().split()
# 					if temp[0] in drop:
# 						continue
# 					pair = (optset, temp[0])
# 					pairs.append(pair)
# 					# if pair in USE:
# 					yindo.append([float(x) for x in temp[1:]])
# 						# yindoidx.append(USE[pair])
# 			with open("../mol_data/opt/%s/%s/%s.txt" % (optset, atom, dataset), 'r') as f:
# 				for line in f:
# 					temp = line.strip().split()
# 					if temp[0] in drop:
# 						continue

# 					pairsdft.append((optset, temp[0]))
# 					yindodft.append([float(x) for x in temp[1:]])

# # print pairs, set(pairsdft)
# # ydft[yindoidx, :]
# print len(yindo), len(yindodft)
# main(ypred[idxs, :], ydft[idxs, :], names[idxs], yindo, yindodft)

