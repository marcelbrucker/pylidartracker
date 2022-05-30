import os
import re
from struct import unpack
import dpkt
from .dataentities import Packet,LaserFiring,Frame
import numpy as np
import open3d as o3d
from pypcd import pypcd


def cart2sph(x: np.ndarray, y: np.ndarray,
             z: np.ndarray):
    """ Convert cartesian to spherical coordinates """

    # Convert -0.0 in 0.0 because np.arctan2() distinguishes 0.0 and -0.0
    x = x + 0.0
    y = y + 0.0
    z = z + 0.0

    hypotxy = np.hypot(x, y)
    r = np.hypot(hypotxy, z)
    elevation = np.arctan2(z, hypotxy)
    azimuth = np.arctan2(y, x)
    return azimuth, elevation, r


def pcd_helper(pcd_files):
    pcdStream = []
    for pcd_file in pcd_files:
        ts = round(float(re.findall("\d+\.\d+", os.path.basename(pcd_file))[0]), 6)

        # Open3D does not read information like intensity in PCD data
        pc_intermediate = pypcd.PointCloud.from_path(pcd_file)
        pc_raw = o3d.geometry.PointCloud()
        points = np.hstack((pc_intermediate.pc_data['x'][:, None],
                            pc_intermediate.pc_data['y'][:, None],
                            pc_intermediate.pc_data['z'][:, None]))
        additional_info = np.hstack((pc_intermediate.pc_data['intensity'][:, None],
                                    pc_intermediate.pc_data['ring'][:, None],
                                    pc_intermediate.pc_data['range'][:, None]))
        pc_raw.points = o3d.utility.Vector3dVector(points)
        pc_raw.normals = o3d.utility.Vector3dVector(additional_info)
        pcdStream.append((ts, pc_raw))
        
    return pcdStream


class PcdFrameParser:
    def __init__(self, pcd_files):
        # check if PCAP file is really .pcap
        self.pcd_files = pcd_files
        self.packetStream = pcd_helper(self.pcd_files)
        self.frameCount = 0
        self.lastAzi = -1
        self.frame = Frame()

    def is_correct_port(self, buffer, port=2368):
        # get destination port from the UDP header
        # dport = unpack(">H",buffer[36:38])[0]
        # return  dport == port
        return True

    def generator(self):
        for ts, buf in self.packetStream:
            self.frame = Frame()
            points = np.asarray(buf.points)
            normals = np.asarray(buf.normals)
            azimuth, elevation, range_from_sensor = cart2sph(points[:, 0], points[:, 1], points[:, 2])
            self.frame.id = normals[:, 1]
            self.frame.elevation = np.rad2deg(elevation)
            azimuth = np.rad2deg(azimuth)
            # azimuth = -1.0 * azimuth  # point cloud is mirror-inverted
            azimuth[azimuth < 0] = azimuth[azimuth < 0] + 360
            self.frame.azimuth = azimuth
            self.frame.distance = range_from_sensor
            self.frame.intensity = normals[:, 0]
            yield (ts, self.frame)

    def peek_size(self):
        return len(self.packetStream)
