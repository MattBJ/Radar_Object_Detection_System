# Multivariate Kalman filter
# Just going to use 4 states (x, x', y, y')

# USING 4 STATE MEASUREMENT SPACE

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
	return np.array([np.random.randn()*noise_factor + y_0 + v_0*i/1000 + (1/2)*a*((i/1000)**2) for i in range(count)]), count, np.array([y_0 + v_0*i/1000 + (1/2)*a*((i/1000)**2) for i in range(count)]), np.array([np.random.randn()*noise_factor + v_0 + (a)*(i/1000) for i in range(count)]), np.array([v_0 + (a)*(i/1000) for i in range(count)])
# BIG NOTE:
# Position measurements - Take initial Y, then, using time in miliseconds, can multiply fractions of seconds by METERS/SECOND(SQUARED) to get final position at each point
		# use y0, dy0/dt, (1/2)d^2y/dt^2 = (1/2) a (dt)^2
		# y0 is in meters
		# dt is in miliseconds, but v0 is in meters per second.. m/s * 0.001 second = curent m/s
		# a is in m/s^2 --> A(m/s^2) * (.001s^2) contributes to position
# VELOCITY MEASUREMENTS - Take initial dy/dt, then use the d^2y/dt^2 to get final velocity:
		# need to use dt = .001, so need to find dy over .001 seconds NOT per second
# FINAL NOTE:
		# see if we need to get velocity PER MILISECONDS or velocity PER SECOND
		# earlier we were just using position in our measurements, so wasn't a big deal

# From this function, will get the noisy Y posisition data, the number of iterations (in milliseconds), and the ACTUAL track/y data

noise_factor = .1 # 'normal' white distribution

y_0 = 4 # 4 meters above the ground
dy_0 = 2.4 # 2 m/s

y_data, count, actual_y, y_vel_data, actual_y_vel = generate_y_data(y_0 = y_0, v_0 = dy_0, noise_factor = noise_factor)

x_0 = 0
dx_0 = 2 # .3 meters/s

x_data = np.array([np.random.randn() * noise_factor + x_0 + (dx_0)*(i/1000) for i in range(count)])

actual_x = np.array([x_0 + (dx_0)*(i/1000) for i in range(count)])

x_vel_data = np.array([np.random.randn()*noise_factor + dx_0 for i in range(count)])

actual_x_vel = np.array([dx_0 for i in range(count)])
# print(actual_x_vel[100])
# print(actual_x_vel[0])
# print(actual_x_vel[489])

# NOTE: VELOCITIES ARE IN REFERENCE TO MILISECONDS

time = np.arange(count) # Miliseconds

dt = 0.001 # 1 millisecond

# FROM HERE WE HAVE:
# actual_x,y for the 'REAL' data... which is wrong with wind, but won't worry
actual_x = np.array([actual_x]); actual_x = actual_x.T
actual_y = np.array([actual_y]); actual_y = actual_y.T
# actual_x_vel = np.array([actual_x_vel]).T
# actual_y_vel = np.array([actual_y_vel]).T
# DONT NEED TO PLOT TRACKING OF VELOCITY, JUST POSITION


# x,y_data for the DATA WITH ADDED NOISE --> Assumed to be MEASUREMENT NOISE
x_data = np.array([x_data]); x_data = x_data.T
y_data = np.array([y_data]); y_data = y_data.T
x_vel_data = np.array([x_vel_data]).T
y_vel_data = np.array([y_vel_data]).T

zs = np.column_stack((x_data,x_vel_data,y_data,y_vel_data))
track = np.column_stack((actual_x,actual_y)) # NOT GOING TO TRACK VELOCITY ON A PLOT

# dt = 1 millisecond

from scipy.linalg import inv
from filterpy.common import Saver # Go back and see if we need it?

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise # Function with params: (Dimensions =, dt = time step, Variance = Q)

from scipy.linalg import pinv # for doing pseudo inverses on non square matrices

# First, going to keep 4 state, and just add the addition of acceleration in manually

# H = np.array([[1]]) # FULL MEASUREMENT SPACE
# H = np.diag([1,1,1,1]) # identity matrix
H = np.array([[1,0,0,0],
			  [0,1,0,0],
			  [0,0,1,0],
			  [0,0,0,1]])
# H = 1 # scalar I guess
# HIDDEN VARIABLES -> vELOCITY

R = np.diag([.75,.75,.75,.75]) # x, y measurement noise covariance
# 95% of data is between 1-2 meters from actual
# ADDED FOR VELOCITIES, so play with it

F = np.array([[1,dt,0,0],
			  [0,1,0,0],
			  [0,0,1,dt + (-9.8)*(1/2)*(dt**2)],
			  [0,0,0,1 + (-9.8)*dt]])

print(F)

# State transition matrix/Function
# remember, this is using millisecond for dT


# P = np.diag([R[0,0],9,R[1,1],9])
P = np.diag([R[0,0],
			 R[1,1],
			 R[2,2],
			 R[3,3]])
# PLAY WITH THIS

# R0, R3
# Vx max = maybe 3 meters/second?
# Vy max = maybe 3 meters/second?


#... noise is not as bad e-5
Q_var = 0.5625# Process noise, from drag
# Q_var = 0.015
# 95% of actual is around 1-1.5 centimeters from prediction
Q = Q_discrete_white_noise(dim=4,dt=dt,var=Q_var)

def multivariate_pos_vel_filter(x,P,R,Q,dt,F,H):
	kf = KalmanFilter(dim_x = 4, dim_z = 4)
	kf.x = np.array([x[0],x[1],x[2],x[3]])
	kf.F = F
	kf.H = H
	kf.R = R
	kf.Q = Q
	return kf

def run_filter(x0,P,R,Q,dt,F,H,track,zs,count):
	if x0 is None:
		temp = np.array([zs[0,:]])
		# print(temp.T)
		# print(pinv(H).shape)
		x0 = np.dot(pinv(H),temp.T) #... but I thought he said 'uninvertable'
		# x0 = np.dot(np.linalg.inv(H),temp.T)
		x0 = temp.T
		print(x0)
		print(F)
		# x0 = np.array([0,0,4,0])
	kf = multivariate_pos_vel_filter(x=x0,P=P,R=R,Q=Q,dt=dt,F=F,H=H)

	# print(H.shape)
	# print(P.shape)
	# print(R.shape)
	# print(np.dot(H,P).dot(H.T).shape)

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

# print(zs.shape)

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
ax.scatter3D(zs[:,0],time,zs[:,2], alpha = 0.1, c = 'b') # alpha changes transparency
# print(zs[:,0])
# Noisy data



# print(xs.shape)

x = xs[:,0]
y = xs[:,2]

# print(x)

# x_new, y_new = [],[]
# for data in x:
	# x_new.append(x)
	# y_new.append(y)

# print(x.shape)
# print(y)
# print(x_new.shape)

ax.plot3D(np.ndarray.flatten(x),time,np.ndarray.flatten(y),'r') # filter output
# print(xs[:,2])

ax.set_xlabel('X_position (m)'); ax.set_ylabel('Time (ms)'); ax.set_zlabel('Y_position (m)')

plt.show()

print(kf.K)
print(kf.R)
print(kf.P)
print(kf.Q)