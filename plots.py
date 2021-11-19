import pickle
import matplotlib.pyplot as plt
import argparse
import os
parser = argparse.ArgumentParser()

parser.add_argument('-file', type = str, default = None, help = 'Filename for which metric-plots to be generated')
parser.add_argument('-folder', type = str, default = None, help = 'Source folder for Results File')
args = parser.parse_args()

file_name = args.file
folder = args.folder

def plot_testacc(folder,filename): 
	filepath = os.path.join(folder, filename) 
	with open(filepath, 'rb') as fobject:
		state = pickle.load(fobject)

	plt.figure(figsize = (15, 10))
	modes = [mode for mode in state.keys()]
	for mode in state.keys():
		plt.plot(state[mode].avgtestacc)
	plt.legend([mode.upper() for mode in modes])
	plt.grid()
	plt.title(filename)
	save_path = '../Results/Figures/' + filename
	plt.savefig(save_path)

plot_testacc(folder, file_name)
