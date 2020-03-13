from filterpy.stats import norm_cdf
from filterpy.stats import gaussian
import numpy as np
from filterpy.stats import plot_gaussian_pdf

u1, u2 = 10, 32

var1, var2 = 25, 100


xs = u1

G1 = gaussian(xs,u1,var1)

xs = u2


G2 = gaussian(xs, u2, var2)

print(G1,'\n',G2)

xs = np.arange(-100,100,.1)

sum1, sum2 = 0,0
sum3 = 0
for i in xs:
	sum1 = sum1 + gaussian(i,u1,var1)
	sum2 = sum2 + gaussian(i,u2,var2)
	sum3 = sum3 + gaussian(i,u1,var1)*gaussian(i,u2,var2)

# Sum 3 is like nothing. Multiplication is not correct, have to change u's and var's myself

print('\n\t',sum1,'\n\t',sum2,'\n\t',sum3)

print(norm_cdf((-100,100),u1,var1))

print(norm_cdf((-100,100),u2,var2))

# here comes the update stage (likelihood * priori ... + normalization?)

# x3 = 
# var3 = 

# norm_cdf((-1000000,1000000),u,var)*100 # 100 for percent.. should equal 100 percent not over