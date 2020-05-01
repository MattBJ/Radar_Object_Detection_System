## Matthew Bailey

# FFT simulation of frequency buffers

# Purpose: Take the entire sampled simulation data and run them through 3 FFTs (for each data buffer of 900000 64-bit floating point values)
			# --> use the FFT's to create position buffer (won't worry about dopplar shift for this)

# NEED TO DO BATCH PROCESSING, NOT ENOUGH RAM
			# --> Nested functions/subroutines to clear the RAM

# Final 'global' scope variables: Position buffers, iteration storage for files? Maybe that's just local to the looping

import numpy as np
from scipy import fftpack


def fft_and_conversion(sample_buffer,sample_size,sample_freq):
	# Calculates the fft of the buffer
	# Converts this fft value to the 'best guess' of the position at this time instance
	# print(sample_buffer.shape)
	dt = 1/sample_freq
	fft_sample_freq = fftpack.fftfreq(sample_size,d=dt)
	spectrum = fftpack.fft(sample_buffer)
	# spectrum = np.fft.rfftn(sample_buffer) # sample_size - point fft (ie sample_size number of bins)

	# print(spectrum)

	spectrum = spectrum[:(int(sample_size/2))]

	spectrum *= 2 # truncated, multiplied by 2

	# Find the magnitudes, divide them all by 2048 for averaging

	spectrum = np.absolute(spectrum)

	spectrum /= sample_size

	print(spectrum.shape) # 2048

	# print(spectrum)

	# NOW FIND MAX MAGNITUDE VALUE

	# elementMax = np.amax(spectrum)

	# window 10 bins at a time, add them up, and append them to an array
	# Find maximum window'd element --> Use that for the estimation

	x = len(spectrum) # 1024
	estimation = []
	for i in range(int(x-10)):
		sumate = 0
		for j in range(10):
			sumate += spectrum[i + j]
		estimation.append(sumate)
	estimation = np.array(estimation)


	elementMax = np.where(estimation == np.amax(estimation))
	elementMax = elementMax[0] # gives the index into this estimation

	elementMax += 5 # roughly halfway between estimation bins

	print('Bin with highest magnitude: ',elementMax)

	# Know which frequency is the highest

	bin_size = (sample_freq/sample_size)

	del_f = elementMax * bin_size # element number * bin size
	del_f -= (bin_size/2) # average of the bin === midway between bins

	print('frequency component: ',del_f)

	# Convert it to position

	del_t = (del_f)/(6e11) # 600 GHz frequency modulation rate --> 

	distance = (3e8) * del_t # speed of light times change in time

	return distance

def read_elements(buffer_file,it_offset,sample_size,sample_freq):
	# Take 'sample_size' number of floating point elements from the 'buffer_file', using offset 'iteration'
	# Then call a nested function that calculates the fft of the temporary buffer
	# Final return should be: iteration offset (previous iteration offset + sample_size), and position value
	file = open(buffer_file,"r") # read only priveleges

	# read each value with delimiter ','
	temp_buffer = []
	with open(buffer_file,'r') as f:
		for i, line in enumerate(f.readlines(),0):
			if((i >= it_offset) and (i < (it_offset + sample_size))): # for example: From line 0 -> sample_size - 1
				# Allocate to the buffer
				temp_buffer.append(float(line))
	temp_buffer = np.array(temp_buffer)
	distance = fft_and_conversion(temp_buffer,sample_size,sample_freq)
	it_offset += sample_size
	file.close() # minimize the RAM usage as much as possible.. maybe it's bad to keep opening and closing as a large file anyway? Idk
	return it_offset, distance

file_1_name = "RX1_buffer.txt"

N = 9e4

sample_size = 2048

sample_freq = 100000 # 100ksps

pos_buffer = []
it_offset = 0

print(int(N/sample_size))

for i in range(int(N/sample_size)): # rounds down
	print(i)
	it_offset, tmp_dist = read_elements(file_1_name,it_offset,sample_size,sample_freq)
	pos_buffer.append(tmp_dist)

pos_buffer = np.array(pos_buffer)

print('iteration offset: ',it_offset)
print(pos_buffer)

pos_buffer = np.array(pos_buffer)