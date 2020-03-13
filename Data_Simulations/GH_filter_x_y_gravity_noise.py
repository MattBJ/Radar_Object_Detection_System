# Generate synthetic noisy data for radar
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

from filterpy.gh import GHFilter

def generate_y_data(y_0, v_0, noise_factor): # count is in milliseconds
	# time step will be 1 millisecond
	a = -9.8 # meters per second
	# a = -9.8e-3 # per milisecond
	v_0_mili = copy.copy(v_0)
	v_0_mili = v_0_mili/1000
	# v_0_mili *= 1e-3 # get in velocity per milisecond
	count = (-2*v_0)/a # count is in MILISECONDS
	count = int(round(count*1000))
	return np.array([np.random.randn()*noise_factor + y_0 + v_0*i/1000 + (1/2)*a*((i/1000)**2) for i in range(count)]), count, np.array([y_0 + v_0*i/1000 + (1/2)*a*((i/1000)**2) for i in range(count)])

#count = 2000

noise_factor = .1 # 'normal' white distribution

# randn vs normal:
# randn = mean 0 variance 1
	# 'univariate' 'normal' (gaussian) distribution
# normal takes more parameters but is essentially the same

y_0 = 4 # 4 meters above the ground
dy_0 = 2.4 # 2 m/s

# Make count such that it returns to y_0

# y_f = y_0, displacement = y_0 + dy * /\t + (1/2)a*(/\t)^2
# --> 0 = dy(t) + (1/2)a(t^2)
# --> 0 = 1 + (1/2)a(t)/dy
# --> (-1*dy*2)/a = t

y_data, count, actual_y = generate_y_data(y_0 = y_0, v_0 = dy_0, noise_factor = noise_factor)


x_0 = 0
dx_0 = 2 # .3 meters/s

x_data = np.array([np.random.randn() * noise_factor + x_0 + (dx_0*i)/1000 for i in range(count)])

actual_x = np.array([x_0 + (dx_0*i)/1000 for i in range(count)])

time = np.arange(count)

# Axes3D.plot(xs=x_data,yx=y_data,zs=time,zdir='z')

# ax.plot3D(x_data,time,y_data, 'red')
# ax.scatter3D(x_data,time,y_data)# cmap = 'Greens')

# plt.show()

####################################################
# NOTE - try to retain 'actual' data for comparison
####################################################

####################################################
# GH/Alpha betta filter - 2D attempt
####################################################

init_pos_guess = np.array([x_0,y_0])
init_vel_guess = np.array([1,1])

flag = True
while(flag):
	# g,h = np.random.uniform(0,0.2),np.random.uniform(0,0.01)
	g, h = 0.020282406381970187, 0.0003965804370818338
	# Just noticed those were good, I'll keep the random uniform distributions up for tests later
	print('g =',g, '\th =',h)
	GH_air = GHFilter(x= init_pos_guess, dx = init_vel_guess, dt = 0.001, g = g, h= h) # g = .1, h = .005
	Data = []
	for i in range(count):
		# print(i)
		GH_air.update(z = np.array([x_data[i], y_data[i]])) # x, y
		Data.append(GH_air.x)

	# LATER, try and use GH_air.batch_filter([[],[],[],...,[]]) to update a whole batch of data instead of for loops

	Data = np.array(Data)

	print(count)

	# print(Data)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	# ax.plot3D(x_data,time,y_data, 'red')
	ax.scatter3D(x_data,time,y_data, alpha = 0.1, c = 'b') # alpha changes transparency

	ax.plot3D(actual_x,time,actual_y, alpha = 0.8, c = 'g')

	print(Data.shape)
	ax.plot3D(Data[:,0],time,Data[:,1],'r')

	ax.set_xlabel('X_position (m)'); ax.set_ylabel('Time (ms)'); ax.set_zlabel('Y_position (m)')

	plt.show()

	choice = input('Terminate? y/n')
	if(choice == 'y'):
		flag = False

	# BEST COEFFICIENTS TO DATE
	# 1) G = 0.012163461057957537, H = 0.0010950138756964133
	#
	# 2) G = 0.020282406381970187, H = 0.0003965804370818338
	#
	#