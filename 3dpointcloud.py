import numpy as np
import matplotlib.pyplot as plt
import os
# from mpl_toolkits.mplot3d import Axes3D

dir = os.listdir("./myfed_normal_save")
data = np.loadtxt("./myfed_normal_save/"+dir[-1])
points = []

for i in range(len(data)):
    for j in range(i, len(data)):
        points.append([i, j, abs(data[i][j])])

x=[point[0] for point in points]
y=[point[1] for point in points]
z=[point[2] for point in points]

fig=plt.figure(dpi=120)
ax=fig.add_subplot(111,projection='3d')
plt.title('point cloud')
ax.scatter(x,y,z,c='b',marker='.',s=2,linewidth=0,alpha=1,cmap='spectral')

#ax.set_facecolor((0,0,0))
ax.axis('auto')
# ax.xaxis.set_visible(False)
# ax.yaxis.set_visible(False)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()