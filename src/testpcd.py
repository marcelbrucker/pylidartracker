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
    file = ["/home/marcel/Repositorys/a9_dataset_r01_s04/_points/r01_s05_sensor_data_s110_lidar_ouster_north_1646667395.048232462.pcd.pcd",
            "/home/marcel/Repositorys/a9_dataset_r01_s04/_points/r01_s05_sensor_data_s110_lidar_ouster_north_1646667395.142218273.pcd.pcd"]
    config_file = "/home/marcel/Downloads/pylidartracker_test_data/street_config.json"
    config_file = "/home/marcel/Downloads/pylidartracker_test_data/pcd_config.json"
    ctrl.analyze_pcap_fn(file, None)
    ctrl.load_frames_fn(0, 2, None)
    ctrl._model.init_from_config(config_file)
    ctrl._apply_preprocessing_fn(None)
    print('Hello World!')

    # packetStream = dpkt.pcap.Writer(open(pcap_file, 'wb'))