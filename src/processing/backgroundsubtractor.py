import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree

from .pcdframeparser import cart2sph

class BackgroundSubtractor():
    @staticmethod
    def factory(method, **kwargs):
        if method == "kd_tree":
            return KDTreeSubtractor(**kwargs)
        elif method == "octree":
            return OctreeSubtractor(**kwargs)
        elif method == "element_comparison":
            return ElementComparisonSubtractor(**kwargs)
        else:
            ValueError(method)

class OctreeSubtractor():
    def __init__(self, bg_cloud, resolution):
        self.bg = bg_cloud
        self.bgTree = BallTree(bg_cloud.to_array(), leaf_size=10)
        p = pcl.PointCloud()
        self.cd = p.make_octreeChangeDetector(resolution)
    
    def getArray(self,x,y,z):
        # this operation belogs to FrameStream thread
        data = np.vstack((x,y,z)).astype(np.float32).T
        return np.unique(data.round(decimals=4), axis=0)

    def subtract(self, x, y, z):
        data = self.getArray(x,y,z)
        p = pcl.PointCloud(data)
        self.cd.delete_tree()

        self.cd.set_input_cloud(self.bg)
        self.cd.add_points_from_input_cloud()
        self.cd.switchBuffers()
        self.cd.set_input_cloud(p)
        self.cd.add_points_from_input_cloud()
        res = self.cd.get_PointIndicesFromNewVoxels()
        xNew, yNew, zNew = data[res].T
        return xNew,yNew,zNew

    def get_config(self):
        return {
            "method": "octree", 
            "params": {
                "background": "",
                "resolution": self.resolution
                }
            }

class KDTreeSubtractor():
    def __init__(self, bg_cloud=None, search_radius=0.1):
        self.bg_cloud = bg_cloud
        
        self.bgTree = None
        if self.bg_cloud is not None:
            self.bgTree = BallTree(bg_cloud, leaf_size=10)

        self.search_radius = search_radius

    def set_background2(self, bg_cloud):
        self.bg_cloud = bg_cloud
        self.bgTree = cKDTree(bg_cloud)

    def set_background(self, bg_cloud):
        self.bg_cloud = bg_cloud
        self.bgTree = BallTree(bg_cloud, leaf_size=10)

    def subtract2(self, arr):
        arrayTree = cKDTree(arr)
        res = arrayTree.query_ball_tree(
            self.bgTree, r=self.search_radius)
        idx = [i for i,x in enumerate(res) if not x]
        return arrayTree.data[idx]

    def subtract(self, arr):
        if arr.size == 0:
            return arr
        else:
            counts = self.bgTree.query_radius(arr,
                r=self.search_radius, count_only=True)
            return arr[counts==0]

    def get_config(self):
        return {
            "method": "kd_tree", 
            "params": {
                "search_radius": self.search_radius
                }
            }

class ElementComparisonSubtractor():
    def __init__(self, bg_cloud=None, search_radius=0):
        self.bg_cloud = bg_cloud
        
        self.bgTree = None
        # if self.bg_cloud is not None:
        #     self.bgTree = BallTree(bg_cloud, leaf_size=10)

    def set_background(self, bg_cloud):
        self.bg_cloud = bg_cloud

    def set_range_thrld_matrix(self, range_thrld_matrix):
        self.range_thrld_matrix = range_thrld_matrix

    def subtract(self, arr):
        if arr.size == 0:
            return arr
        else:
            total_grid = 2048
            beams_elevation = np.deg2rad([
                -0.67, -1.03, -1.37, -1.71, -2.09, -2.42, -2.78, -3.13, -3.51, -3.84,
                -4.19, -4.55, -4.9, -5.24, -5.6, -5.95, -6.33, -6.65, -7, -7.35, -7.7,
                -8.06, -8.39, -8.75, -9.11, -9.44, -9.78, -10.12, -10.49, -10.83, -11.16,
                -11.5, -11.85, -12.19, -12.52, -12.85, -13.22, -13.54, -13.88, -14.2,
                -14.57, -14.89, -15.2, -15.52, -15.89, -16.21, -16.54, -16.84, -17.19,
                -17.52, -17.82, -18.14, -18.48, -18.8, -19.1, -19.42, -19.75, -20.05,
                -20.36, -20.66, -21, -21.3, -21.6, -21.89
            ])
            # range_thrld_matrix = np.sqrt(self.bg_cloud[:, 0] ** 2 + self.bg_cloud[:, 1] ** 2 + self.bg_cloud[:, 2] ** 2).reshape(64, 2048)
            range_thrld_matrix = self.range_thrld_matrix
            AZIMUTH_UNIT = 360.0 / total_grid

            # pcd = crop_pointcloud(
            #     pcd, x_limits, y_limits, z_limits
            # )  # TODO: Think about removing this as filter_roi(...) takes care of it
            # pcd_roi_filtered = filter_roi(pcd, roi_mask)

            x_roi_filtered = arr[:, 0]
            y_roi_filtered = arr[:, 1]
            z_roi_filtered = arr[:, 2]

            azimuths, elevations, ranges_from_sensor = cart2sph(
                x_roi_filtered, y_roi_filtered, z_roi_filtered)

            degree = np.rad2deg(azimuths)
            degree[degree < 0] = degree[degree < 0] + 360
            grids_idx = np.mod(np.floor(degree / AZIMUTH_UNIT) + 1, total_grid)
            grids_idx[grids_idx == 0] = total_grid - 1
            grids_idx[np.isnan(grids_idx)] = 0
            grids_idx = grids_idx.astype(int)

            back_idxes = np.zeros(x_roi_filtered.shape[0], dtype=int)

            for pnt in range(elevations.shape[0]):
                range_from_sensor = ranges_from_sensor[pnt]
                elevation = elevations[pnt]
                grid_idx = grids_idx[pnt]

                if grid_idx == 0:
                    back_idxes[pnt] = 1
                    continue

                channel_num = np.argmin(np.abs(beams_elevation - elevation))
                distance_flag = range_from_sensor >= range_thrld_matrix[channel_num,
                                                                        grid_idx]
                if distance_flag:
                    back_idxes[pnt] = 1

            foreground_indices = np.squeeze(np.argwhere(back_idxes == 0))

            return arr[foreground_indices]

    def get_config(self):
        return {
            "method": "element_comparison", 
            "params": {
                }
            }
