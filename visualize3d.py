import time

import numpy as np
import open3d


class Visualize3D:
    def __init__(self, n, m, save_path, mmpp):
        self.n, self.m = n, m
        self.init_open3D()
        self.cnt = 212
        self.save_path = save_path
        pass

    def init_open3D(self):
        x = np.arange(self.n)  # * mmpp
        y = np.arange(self.m)  # * mmpp
        self.X, self.Y = np.meshgrid(x, y)
        Z = np.sin(self.X)

        self.points = np.zeros([self.n * self.m, 3])
        self.points[:, 0] = np.ndarray.flatten(self.X)  # / self.m
        self.points[:, 1] = np.ndarray.flatten(self.Y)  # / self.n

        self.depth2points(Z)

        self.pcd = open3d.geometry.PointCloud()
        self.pcd.points = open3d.utility.Vector3dVector(self.points)
        # self.pcd.colors = Vector3dVector(np.zeros([self.n, self.m, 3]))
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window(width=640, height=480)
        self.vis.add_geometry(self.pcd)

    def depth2points(self, Z):
        self.points[:, 2] = np.ndarray.flatten(Z)

    def update(self, Z):
        self.depth2points(Z)
        dx, dy = np.gradient(Z)
        dx, dy = dx * 0.5, dy * 0.5
        np_colors = dx + 0.5
        np_colors[np_colors < 0] = 0
        np_colors[np_colors > 1] = 1
        np_colors = np.ndarray.flatten(np_colors)
        colors = np.zeros([self.points.shape[0], 3])
        for _ in range(3):
            colors[:, _] = np_colors

        self.pcd.points = open3d.utility.Vector3dVector(self.points)
        self.pcd.colors = open3d.utility.Vector3dVector(colors)
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

        # SAVE POINT CLOUD TO A FILE
        if self.save_path != "":
            open3d.io.write_point_cloud(
                self.save_path + "/pc_{}.pcd".format(self.cnt), self.pcd
            )

        self.cnt += 1

    def save_pointcloud(self):
        open3d.io.write_point_cloud(
            self.save_path + "pc_{}.pcd".format(self.cnt), self.pcd
        )

if __name__ == "__main__":
    vis3d = Visualize3D(328,242,'out_video.mp4',None)
    while True:
        vis3d.update(np.zeros([328,242]))
        time.sleep(0.01)
