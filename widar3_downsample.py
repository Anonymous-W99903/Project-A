"""
downsample DFS
"""
import numpy as np
import scipy.io as scio
import os


NEW_LEN = 128

def downsample(x : np.ndarray):
	old_len = x.shape[2]
	step = old_len // NEW_LEN
	new_datas = []
	for start in range(step):
			index = list(range(start, old_len, step))[:NEW_LEN]
			new_datas.append(x[:,:,index])
	return new_datas


def main():
	data_dir = r'E:\widar3\20181130_dfs\user5'
	save_dir = r'data\widar3'
	for filename in os.listdir(data_dir):
		dat = scio.loadmat(os.path.join(data_dir, filename))
		dat = dat['doppler_spectrum']
		# save
		np.save(os.path.join(save_dir, filename[:-4]), dat)
		# dowansample and save
		new_dats = downsample(dat)
		for i in range(len(new_dats)):
			new_filename = filename[:-4] + f'-d{i+1}'
			np.save(os.path.join(save_dir, new_filename), new_dats[i])


main()
