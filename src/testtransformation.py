import numpy as np
import open3d as o3d

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

def sph2cart(azimuth, elevation, r):
    # azimuth = np.deg2rad(azimuth)
    # elevation = np.deg2rad(elevation)
    x = (r * np.cos(elevation) * np.sin(azimuth)).flatten()
    y = (r * np.cos(elevation) * np.cos(azimuth)).flatten()
    z = (r * np.sin(elevation)).flatten()
    return x,y,z

def sph2cart_matlab(azimuth, elevation, r):
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)
    x = (r * np.cos(elevation) * np.cos(azimuth)).flatten()
    y = (r * np.cos(elevation) * np.sin(azimuth)).flatten()
    z = (r * np.sin(elevation)).flatten()
    return x,y,z

if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("/home/marcel/Repositorys/point_cloud2_night/_points/1653526381.493075325.pcd")
    points = np.asarray(pcd.points)
    x_orig, y_orig, z_orig = points[:, 0], points[:, 1], points[:, 2]
    azimuth, elevation, r = cart2sph(x_orig, y_orig, z_orig)
    azimuth = np.rad2deg(azimuth)
    elevation = np.rad2deg(elevation)
    # azimuth = -1.0 * azimuth # point cloud is mirror-inverted
    azimuth[azimuth < 0] = azimuth[azimuth < 0] + 360
    x_proc, y_proc, z_proc = sph2cart_matlab(azimuth, elevation, r)
    bg_compare = np.load("/home/marcel/bg_compare.npy")
    x_proc, y_proc, z_proc = sph2cart_matlab(bg_compare[:, 0], bg_compare[:, 1], bg_compare[:, 2])
    points = np.hstack((x_proc[:, None], y_proc[:, None], z_proc[:, None]))
    np.save("/home/marcel/bg.npy", points)
    print(f'x identical: {np.allclose(x_orig, x_proc)}, y identical: {np.allclose(y_orig, y_proc)}, z identical: {np.allclose(z_orig, z_proc)}')
