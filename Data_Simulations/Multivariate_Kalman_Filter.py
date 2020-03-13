# Multivariate Kalman filter
# Just going to use 4 states (x, x', y, y')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

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
# From this function, will get the noisy Y posisition data, the number of iterations (in milliseconds), and the ACTUAL track/y data

noise_factor = 0.1 # 'normal' white distribution

y_0 = 4 # 4 meters above the ground
dy_0 = 2.4 # 2 m/s

y_data, count, actual_y = generate_y_data(y_0 = y_0, v_0 = dy_0, noise_factor = noise_factor)

x_0 = 0
dx_0 = 2 # .3 meters/s

x_data = np.array([np.random.randn() * noise_factor + x_0 + (dx_0*i)/1000 for i in range(count)])

actual_x = np.array([x_0 + (dx_0*i)/1000 for i in range(count)])

time = np.arange(count) # Miliseconds

dt = 0.001 # 1 millisecond

# FROM HERE WE HAVE:
# actual_x,y for the 'REAL' data... which is wrong with wind, but won't worry
actual_x = np.array([actual_x]); actual_x = actual_x.T
actual_y = np.array([actual_y]); actual_y = actual_y.T
# x,y_data for the DATA WITH ADDED NOISE --> Assumed to be MEASUREMENT NOISE
x_data = np.array([x_data]); x_data = x_data.T
y_data = np.array([y_data]); y_data = y_data.T

zs = np.column_stack((x_data,y_data))
track = np.column_stack((actual_x,actual_y))

# dt = 1 millisecond

from scipy.linalg import inv
from filterpy.common import Saver # Go back and see if we need it?

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise # Function with params: (Dimensions =, dt = time step, Variance = Q)

from scipy.linalg import pinv # for doing pseudo inverses on non square matrices

H = np.array([[1,0,0,0],[0,0,1,0]])
# HIDDEN VARIABLES -> vELOCITY

R = np.diag([.75,.75]) # x, y measurement noise covariance
# 95% of data is between 1-2 meters from actual

F = np.array([[1, dt, 0, 0],[0, 1, 0, 0],[0, 0, 1, dt],[0, 0, 0, 1]])
# State transition matrix/Function


P = np.diag([R[0,0],9,R[1,1],9])
# R0, R3
# Vx max = maybe 3 meters/second?
# Vy max = maybe 3 meters/second?

Q_var = 0.5625# Process noise, from drag
# Q_var = 0.015
# 95% of actual is around 1-1.5 centimeters from prediction
Q = Q_discrete_white_noise(dim=4,dt=dt,var=Q_var)

def multivariate_pos_vel_filter(x,P,R,Q,dt,F,H):
	kf = KalmanFilter(dim_x = 4, dim_z = 2)
	kf.x = np.array([x[0],x[1],x[2],x[3]])
	kf.F = F
	kf.H = H
	kf.R = R
	kf.Q = Q
	return kf

def run_filter(x0,P,R,Q,dt,F,H,track,zs,count):
	if x0 is None:
		temp = np.array([zs[0,:]])
		# print(pinv(H).shape)
		x0 = np.dot(pinv(H),temp.T) #... but I thought he said 'uninvertable'
		# x0 = np.array([0,0,4,0])
	kf = multivariate_pos_vel_filter(x=x0,P=P,R=R,Q=Q,dt=dt,F=F,H=H)

	# print(H.shape)
	# print(P.shape)
	# print(R.shape)
	print(np.dot(H,P).dot(H.T).shape)

	xs, cov = [],[]
	x = x0
	for z in zs:
		z = np.array([z]).T
		kf.predict()
		kf.update(z)

		# predict
		# x = np.dot(F,x)
		# P = np.dot(F,P).dot(F.T) + Q

		# update
		# S = np.dot(H,P).dot(H.T) + R
		# print(np.dot(H,P).dot(H.T).shape)
		# print(R.shape)
		# print(R)
		# print(S)
		# K = np.dot(P,H.T).dot(inv(S))
		# y = z - np.dot(H,x)
		# print(y.shape)
		# x += np.dot(K,y)
		# P = P - np.dot(K,H).dot(P)

		xs.append(kf.x)
		cov.append(kf.P)

	xs, cov = np.array(xs), np.array(cov)
	return xs, cov, kf

# print(zs[0,:])
# print(zs[0,:].shape)
# print(np.array([zs[0,:]]))
# print(np.array([zs[0,:]]).shape)


xs, cov, kf = run_filter(x0=None,P=P,R=R,Q=Q,dt=dt,F=F,H=H,track=track,zs=zs,count=count)
# print(zs)

# Print results

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot3D(track[:,0],time,track[:,1], alpha = 0.8, c = 'g') # actual x, time, actual y
ax.scatter3D(zs[:,0],time,zs[:,1], alpha = 0.1, c = 'b') # alpha changes transparency
# Noisy data

print(xs.shape)

x = xs[:,0]
y = xs[:,2]


# x_new, y_new = [],[]
# for data in x:
	# x_new.append(x)
	# y_new.append(y)

print(x.shape)
# print(x_new.shape)

ax.plot3D(np.ndarray.flatten(x),time,np.ndarray.flatten(y),'r') # filter output
# print(xs[:,2])

ax.set_xlabel('X_position (m)'); ax.set_ylabel('Time (ms)'); ax.set_zlabel('Y_position (m)')

plt.show()

print(kf.K)
print("Filter-Real residual: (x,y)",track[489,0]-x[489],track[489,1]-y[489])