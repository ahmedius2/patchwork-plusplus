import os
import sys
import open3d as o3d
import numpy as np
# import pypatchworkpp

cur_dir = os.path.dirname(os.path.abspath(__file__))
#data_dir = os.path.join(cur_dir, '../../data/')
data_dir = '/root/Anytime-Lidar/data/nuscenes/v1.0-mini/sweeps/LIDAR_TOP/'

try:
    patchwork_module_path = os.path.join(cur_dir, "../../build/python_wrapper")
    sys.path.insert(0, patchwork_module_path)
    import pypatchworkpp
except ImportError:
    print("Cannot find pypatchworkpp!")
    exit(1)

def read_bin(bin_path):
    scan = np.fromfile(bin_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))

    return scan

def read_bin_nusc(file_name: str):
    """
    Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity, ring index).
    :param file_name: Path of the pointcloud file on disk.
    :return: LidarPointCloud instance (x, y, z, intensity).
    """
    assert file_name.endswith('.bin'), 'Unsupported filetype {}'.format(file_name)
    scan = np.fromfile(file_name, dtype=np.float32)
    points = scan.reshape((-1, 5))[:, :4]
    mask = points[:, 0] >= -54.0
    mask = np.logical_and(mask, points[:, 0] <= 54.)
    mask = np.logical_and(mask, points[:, 1] >= -54.)
    mask = np.logical_and(mask, points[:, 1] <= 54.)
    mask = np.logical_and(mask, points[:, 2] >= -5.)
    mask = np.logical_and(mask, points[:, 2] <= 3.)
    points = points[mask]
    return points

if __name__ == "__main__":

    # Patchwork++ initialization
    params = pypatchworkpp.Parameters()
    params.verbose = True

    PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)

    for file in sorted(os.listdir(data_dir)):

        # Load point cloud
        pointcloud = read_bin_nusc(data_dir+file)

        # Estimate Ground
        PatchworkPLUSPLUS.estimateGround(pointcloud)

        # Get Ground and Nonground
        ground      = PatchworkPLUSPLUS.getGround()
        nonground   = PatchworkPLUSPLUS.getNonground()
        print(ground.shape, nonground.shape)
        time_taken  = PatchworkPLUSPLUS.getTimeTaken()

        # Get centers and normals for patches
#        centers     = PatchworkPLUSPLUS.getCenters()
#        normals     = PatchworkPLUSPLUS.getNormals()

#        print("Origianl Points  #: ", pointcloud.shape[0])
#        print("Ground Points    #: ", ground.shape[0])
#        print("Nonground Points #: ", nonground.shape[0])
#        print("Time Taken : ", time_taken / 1000000, "(sec)")
#        print("Press ... \n")
#        print("\t H  : help")
#        print("\t N  : visualize the surface normals")
#        print("\tESC : close the Open3D window")
#        
#        vis = o3d.visualization.VisualizerWithKeyCallback()
#        vis.create_window(width = 600, height = 400)
#
#        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
#
#        ground_o3d = o3d.geometry.PointCloud()
#        ground_o3d.points = o3d.utility.Vector3dVector(ground)
#        ground_o3d.colors = o3d.utility.Vector3dVector(
#            np.array([[0.0, 1.0, 0.0] for _ in range(ground.shape[0])], dtype=float) # RGB
#        )
#
#        nonground_o3d = o3d.geometry.PointCloud()
#        nonground_o3d.points = o3d.utility.Vector3dVector(nonground)
#        nonground_o3d.colors = o3d.utility.Vector3dVector(
#            np.array([[1.0, 0.0, 0.0] for _ in range(nonground.shape[0])], dtype=float) #RGB
#        )
#
#        centers_o3d = o3d.geometry.PointCloud()
#        centers_o3d.points = o3d.utility.Vector3dVector(centers)
#        centers_o3d.normals = o3d.utility.Vector3dVector(normals)
#        centers_o3d.colors = o3d.utility.Vector3dVector(
#            np.array([[1.0, 1.0, 0.0] for _ in range(centers.shape[0])], dtype=float) #RGB
#        )
#
#        vis.add_geometry(mesh)
#        vis.add_geometry(ground_o3d)
#        vis.add_geometry(nonground_o3d)
#        vis.add_geometry(centers_o3d)
#
#        vis.run()
#        vis.destroy_window()
