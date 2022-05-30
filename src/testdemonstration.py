import argparse
import os
import json
from typing import List, Tuple

import numpy as np
import open3d as o3d

def plot(geometry_list: List[o3d.geometry.PointCloud],
         dataset_folder_path: str,
         file_name: str,
         dataset: str = '',
         labels: bool = False,
         point_size: float = 1.5) -> None:
    """ Customized visualization """

    if labels:
        geometry_list += load_bounding_boxes(file_path_labels=os.path.join(
            dataset_folder_path, '_labels', file_name + '.json'),
                                             dataset=dataset)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=file_name, left=50)

    for geometry in geometry_list:
        vis.add_geometry(geometry)

    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1))

    # Apply settings after adding geometries to not be overwritten
    vis.get_render_option().background_color = [0.1, 0.1, 0.1]
    vis.get_render_option().point_size = point_size
    vis.get_render_option().light_on = False
    # vis.get_render_option().line_width = 10.0
    if dataset == 'R1_S4-5':
        vis.get_view_control().set_zoom(0.14)
        vis.get_view_control().set_front([-0.796, 0.097, 0.597])
        vis.get_view_control().set_lookat([14.947, -5.645, -5.557])
        vis.get_view_control().set_up([0.592, -0.079, 0.802])
        # vis.get_view_control().set_up([1, 0, 0])
    else:
        vis.get_view_control().set_zoom(0.14)
        vis.get_view_control().set_front([-0.920, 0.094, 0.380])
        vis.get_view_control().set_lookat([24.590, 0.831, -6.376])
        vis.get_view_control().set_up([0.386, 0.055, 0.921])
    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    pcd = o3d.geometry.PointCloud()
    points = np.load("/home/marcel/bg.npy")
    pcd.points = o3d.utility.Vector3dVector(points)
    # pcd = o3d.io.read_point_cloud("/home/marcel/bg_test.pcd")
    plot([pcd],
        dataset_folder_path=os.path.dirname('/home/marcel/Repositorys/a9_dataset_r01_s04/_points/'),
        file_name='r01_s05_sensor_data_s110_lidar_ouster_north_1646667395.048232462.pcd',
        dataset='R1_S4-5',
        labels=False,
        point_size=1.5)