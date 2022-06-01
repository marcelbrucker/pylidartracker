import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from controller import Controller
from processing.lidarprocessor import LidarProcessor
from app import LidarView
import dpkt

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    view = LidarView()
    model = LidarProcessor()
    ctrl = Controller(view=view, model=model)
    file = ["/home/marcel/Downloads/pylidartracker_test_data/street.pcap"]
    file = ["/home/marcel/Repositorys/a9_dataset_r01_s04/_points/r01_s04_sensor_data_s110_lidar_ouster_north_1646667314.041079126.pcd.pcd",
            "/home/marcel/Repositorys/a9_dataset_r01_s04/_points/r01_s04_sensor_data_s110_lidar_ouster_north_1646667315.741349297.pcd.pcd",
            "/home/marcel/Repositorys/a9_dataset_r01_s04/_points/r01_s04_sensor_data_s110_lidar_ouster_north_1646667318.539890142.pcd.pcd"]
    config_file = "/home/marcel/Repositorys/pylidartracker/data/street_config.json"
    config_file = "/home/marcel/Repositorys/pylidartracker/data/pcd_config.json"
    ctrl.pcap_filename = file
    ctrl.analyze_pcap_fn(file, None)
    ctrl.load_frames_fn(0, 3, None)
    ctrl.config_filename = config_file
    ctrl._model.init_from_config(config_file)
    ctrl._apply_preprocessing_fn(None)
    ctrl._apply_clustering_fn(None)
    ctrl._apply_tracking_fn(None)
    ctrl._view.clusteringDock.previewButton.toggle()
    ctrl.updateClusters()
    print('Hello World!')

    # packetStream = dpkt.pcap.Writer(open(pcap_file, 'wb'))