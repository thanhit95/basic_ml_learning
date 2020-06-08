import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# make up data points
points = np.random.rand(15,2)

# add 4 distant dummy points
points = np.append(points, [[999,999], [-999,999], [999,-999], [-999,-999]], axis=0)

# compute Voronoi tesselation
vor = Voronoi(points)

# plot
voronoi_plot_2d(vor, show_vertices=False)

# colorize
for region in vor.regions:
    if not -1 in region:
        polygon = [vor.vertices[i] for i in region]
        plt.fill(*zip(*polygon))

# fix the range of axes
plt.xlim([0,1]), plt.ylim([0,1])

plt.show()
