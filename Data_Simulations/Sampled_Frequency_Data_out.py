# Matthew Bailey

# Simulation python code:

# Purpose: Generate 3 sampled frequency data sets that represent a moving ordinance and 2 stationary objects in a 3D Area

# Secondary purpose: Ditch the stationary objects (explain how it would've worked) - Use only x,x',y, and y' and assume stationary Z, as everything would've been stationary

# Vision: We have 1 transmitting radar, and 3 receivers. The receivers are set up in an equilateral triangle configuration, with the transmitter in the center.
			# There, hopefully, 2 objects that are stationary (represented as single points), that will contribute constant frequency information
			# There will be an ordinance traveling from the near edge of the field to a close point next to the radar dish

# Important information: Bandwidth is 100 MHz, the chirp is 100 microseconds (10 KHz)
							# Chirp waveform: Triangulat (so from 0 modulation to 100 MHz is 50uS)
							# Delta Frequency = 600 GHz
							# Time per freq. change = 1.666 pico seconds
							# Arbitrary distance constraint: 6 meters --> Roughly 25KHz difference (there and back = 12 meters)

# Constraints: C (Speed of light) = 3e8
# Arbitrary distance max = 6 meters (from any receiver)

# The distance to any object to any receiver is calculated by the following:
	# D_TX  = sqrt((TX_x - Obj_x)^2 + (TX_y - Obj_y)^2 + (TX_z - Obj_z)^2)
	# D_RXn = sqrt((RX_x - Obj_x)^2 + (RX_y - Obj_y)^2 + (RX_z - Obj_z)^2)
	# D_Tot = D_TX + D_RXn

# Converting distance to frequency information:
	# (D_tot / c) * 600 GHz = Frequency_Data

# For the stationary objects, this can simply be calculated over the entirety of the large sample set

# For the ordinance, the distance will be dynamically changing
# For each sample in time, a new distance must be calculated and thus the corresponding frequency changes (should steadily get lower)

# Each data set will be represented by a vector of scalars which are set equal to the 3 CURRENT frequency components multiplied by time

# FIRST CHALLENGE: Create a varying frequency dataset (1 HZ to 1KHz, changes once per ____ samples)
	# make the sampling extremely high frequency (1 nanosecond)
	# figure out how to change frequencies dynamically over time WITHOUT jumping around

# after 1000 samples, change frequency

# np.cos() requires an array of data (time set), then outputs an array of equal size. will always take that last element and append it

import numpy as np
import matplotlib.pyplot as plt
import copy

def freq_data(freq,sampling_freq,sample_num): # Takes the frequency, sampling frequency, and the time instance of the data required
	# t = np.arange(0,(sample_num+1)+(1/sampling_freq),(1/sampling_freq))
	# print(t)
	out = np.cos(2*(np.pi)*(freq)*(sample_num / sampling_freq))
	# print(out)
	# return out[(sample_num)*(sampling_freq)] # the last element
	return out

# THE CODE BELOW WAS ME TESTING DYNAMICALLY CHANGING FREQUENCY OF A SIGNAL AS THE SIGNAL IS BEING SAMPLED

# frequency_jumps = np.arange(50,151)
# print(frequency_jumps.shape)
# print(frequency_jumps)
# print(frequency_jumps[999])
# j = 0

# data_buffer = []

# sampling_freq = 1000 # 1ksps

# N = 2224

# freq_jump_prev = 0
# freq_jump_next = (2/50) * 1000

# print(int(52.6))

# for i in range(N):
	# print(i)
	# if(freq_jump_next == i):
		# j += 1
		# freq_jump_prev = freq_jump_next
		# freq_jump_next = freq_jump_prev + (2/frequency_jumps[j]) * 1000
		# freq_jump_next = int((freq_jump_next + 0.5))
		# print(int(freq_jump_next))
		# print('Frequency: ',frequency_jumps[j])
	# data_buffer.append(freq_data(frequency_jumps[j],1000,i))

# data_buffer = np.array(data_buffer)

# Now display the sinusoids over the time

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)

# plt.plot(np.arange(0,N),data_buffer,'r')

# plt.show()

# First attempt, all 1's...

# Generate the 2 object frequencies for the 3 receivers --> Will probably get rid of it

# For now let's just do the ordinance frequencies

# nf_pos = 0.1

TX_pos = np.array([0,1,0])

RX1_pos = np.array([0,1,0])
RX2_pos = np.array([0,0,((4/3)**(1/2))/2])
RX3_pos = np.array([0,0,-((4/3)**(1/2))/2])

x_0 = 4
x_f = 1

sampling_freq = 100000
samples = int(sampling_freq*.9)
print(samples)

# dt = 0.0000001 # let's make it real as possible --> 10 Msps
dt = 1/sampling_freq
dx_0 = (x_f - x_0)/(dt*samples)


y_0 = 5
y_f = 3
# dt = 1/sampling_freq
dy_0 = 2.18777777777 # per second

def generate_y_data(y_0, v_0, nf_pos, nf_vel,dt,samples): # count is in milliseconds
	# time step will be 1 millisecond
	a = -9.8 # meters per second
	# a = -9.8e-3 # per milisecond
	v_0_mili = copy.copy(v_0)
	v_0_mili = v_0_mili/1000
	# v_0_mili *= 1e-3 # get in velocity per milisecond
	count = (-2*v_0)/a # count is in MILISECONDS
	count = int(round(count*1000))
	# return np.array([np.random.randn()*nf_pos + y_0 + v_0*i/1000 + (1/2)*a*((i/1000)**2) for i in range(count)]), count, np.array([y_0 + v_0*i/1000 + (1/2)*a*((i/1000)**2) for i in range(count)]), np.array([np.random.randn()*nf_vel + v_0 + (a)*(i/1000) for i in range(count)]), np.array([v_0 + (a)*(i/1000) for i in range(count)])
	return np.array([y_0 + v_0 * (i*dt) + (1/2)*a*((i*dt)**2) for i in range(samples)])

actual_x = np.array([x_0 + (dx_0)*(i*dt) for i in range(samples)])

actual_y = generate_y_data(y_0,dy_0,0,0,dt,samples)

# actual_x = np.ones(samples) * x_0
# actual_y = np.ones(samples) * y_0 # let's just keep it stationary

# Try and work with the generation using stationary object!

print(actual_x.shape)
print(actual_y.shape)

TX_x = np.ones(samples) * 0
TX_y = np.ones(samples) * 0.5

RX1_x = RX1_pos[0] * np.ones(samples)
RX1_y = RX1_pos[1] * np.ones(samples)
RX2_x = RX2_pos[0] * np.ones(samples)
RX2_y = RX2_pos[1] * np.ones(samples)
RX3_x = RX3_pos[0] * np.ones(samples)
RX3_y = RX3_pos[1] * np.ones(samples)

dist_TX = ((TX_x - actual_x)**2+(TX_y - actual_y)**2)**(1/2)

dist_RX1 = ((RX1_x - actual_x)**2 + (RX1_y - actual_x)**2)**(1/2)
dist_RX2 = ((RX2_x - actual_x)**2 + (RX2_y - actual_x)**2)**(1/2)
dist_RX3 = ((RX3_x - actual_x)**2 + (RX3_y - actual_x)**2)**(1/2)

# Now have all the distance information from the separate RX's
# Now convert the 3 buffers to frequency data! --> Total distance * the frequency rate change

F_1 = (dist_TX + dist_RX1)/(3e8) * 6e11
F_2 = (dist_TX + dist_RX2)/(3e8) * 6e11
F_3 = (dist_TX + dist_RX3)/(3e8) * 6e11

# print(D_1)

# Turn the 3 buffers from frequency data into an actual signal

print(int(1/dt))

k = 0 # Responsible for 'batch programming'
print('total samples: ',samples)
print('sampling freq: ',sampling_freq)

# PROBLEM: running out of RAM space with these 900K - 64-bit (double precision) floating point arrays of frequency information

# Solution: Somehow, call these routines as FUNCTIONS (subroutines), upon return the STACK is cleared (RAM)


# 'Garbage collection doesn't happen in line' --> Need to create a function and call it (kinda like a stack!)
# With this function, return 2 things:
	# 1) Return the string to append to the original string
	# 2) Return the ITERATION to continue the calculation!

text_file_buf1 = open("RX1_buffer.txt","w")
text_file_buf2 = open("RX2_buffer.txt","w")
text_file_buf3 = open("RX3_buffer.txt","w")

# samples_calc = 100 # just do batches of 100
# s_sampled_1 = 'float32_t sampled_RX1[] = {'
# s_sampled_2 = 'float32_t sampled_RX2[] = {'
# s_sampled_3 = 'float32_t sampled_RX3[] = {'

s_sampled_1 = ''
s_sampled_2 = ''
s_sampled_3 = ''


# text_file_buf1.write(s_sampled_1)
# text_file_buf2.write(s_sampled_2)
# text_file_buf3.write(s_sampled_3)

def calc_write(text_file,freq,sampling_freq,sample_num):
	sample_value = freq_data(freq,sampling_freq,sample_num)
	s = str(sample_value) + '\n'
	text_file.write(s)

for q in range(samples):
	print('text_file writing, iteration: ', q, 'Frequency at q: ', F_1[q])
	calc_write(text_file_buf1,F_1[q],sampling_freq,q)
	calc_write(text_file_buf2,F_2[q],sampling_freq,q)
	calc_write(text_file_buf3,F_3[q],sampling_freq,q)

sample_size = 512 # or 2024

true_RX1Dist_fname = "RX1_TruDist.txt"

file_true = open(true_RX1Dist_fname,'w')

for x in range(int(samples/sample_size)):
	print(F_1[x*sample_size]/(6e11))

for x in range(samples):
	s = str(F_1[x]/(6e11)) + '\n'
	file_true.write(s)

file_true.close()

print('final x value of ordinance: ',actual_x[-1])
print('final y value of ordinance: ',actual_y[-1])

# text_file_buf1.close()


# Write a function that does the computation AND creates the string, THEN appends it to the text files

# somehow, for the last 3 files delete the last ',' then write '};' and be done!


# write the large datasets to txt files

# text_file_buf1.write()
text_file_buf1.close()

# text_file_buf2.write()
text_file_buf2.close()

# text_file_buf3.write()
text_file_buf3.close()


print(s_sampled_1)
# {number,number,number,number,....,number,number, + '};'... get rid of LAST ','
