#
#
#      0=================================0
#      |         Data Preparation       |
#      0=================================0
#      This prepares custom point cloud (.laz) to be fed into KPConv code for the part segmentation task.
#      The code normalizes the coordinate, intensity, and color values(16 bit). It shifts the lable values by +1.
#      The code also renames the file to match ShapeNetPart dataset name format.
#      The code rewrites prepared point clouds in 'train_ply', 'test_ply', and 'val_ply' folders in .ply format.
#      Modify the path to the parent folder containing three folders ('train_ply','test_ply','val+ply'): line 25
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Sara Yousefimashhoor - 07/2022
#
# ----------------------------------------------------------------------------------------------------------------------

from plyfile import PlyData, PlyElement
import numpy as np
import laspy
import os
import shutil

# Sets the target directory
os.chdir(r'/home/sara/Desktop/Internship/labelled_pole_dataset-20220117T100957Z-001/Laz/data')
wd = os.getcwd()
splits = ['train_ply', 'test_ply', 'val_ply']
n=0
for split in splits:
    new_wd = os.path.join(wd,split)
    os.chdir(new_wd)
    # Collects all the laz files names in this split
    laz_files = [x for x in os.listdir() if x.endswith('.laz')]
    # Reading Point clouds
    for file in laz_files:
        if file.endswith('.laz'):
            cloud = laspy.read(file)
            # Collects coordinate information
            xyz = (np.vstack((cloud.x, cloud.y, cloud.z)).T.astype('float32'))
            # Center the point cloud around origin (0,0,0)
            pmin = np.min(xyz[:, :3], axis=0)
            pmax = np.max(xyz[:, :3], axis=0)
            xyz[:, :3] -= (pmin + pmax) / 2

            # Collects RGB info
            rgb = (np.vstack((cloud.red, cloud.green, cloud.blue)).T.astype('float32'))
            # normalizing the color values (for 16bit RGB info)
            rgb[:,:] /= 2**16 - 1

            # Collects Intensity information
            intensity = (np.array((cloud.intensity).T.astype('float32')))
            # normalizing the intensity values (for 16bit reflectance info)
            intensity /= 2 ** 16 - 1

            # Reads the labels
            label = (np.array((cloud.label).astype('uint8')))

            # for preparing inference point clouds we set all labeles to 1
            #label = np.ones(xyz.shape[0])

            final_data = np.concatenate((xyz, rgb, intensity[:,None], label[:, None]), axis=1)
            # print(final_data)

            print('Saving BINARY PLY lidar data for ', file)
            #Setting the data type
            prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'f4'), ('green', 'f4'), ('blue', 'f4'),('intensity', 'f4'), ('label', 'u1')]
            vertex_all = np.empty(len(final_data), dtype=prop)
            vertex_all[prop[0][0]] = xyz[:, 0].astype(prop[0][1])
            vertex_all[prop[1][0]] = xyz[:, 1].astype(prop[1][1])
            vertex_all[prop[2][0]] = xyz[:, 2].astype(prop[2][1])
            vertex_all[prop[3][0]] = rgb[:, 0].astype(prop[3][1])
            vertex_all[prop[4][0]] = rgb[:, 1].astype(prop[4][1])
            vertex_all[prop[5][0]] = rgb[:, 2].astype(prop[5][1])
            vertex_all[prop[6][0]] = intensity[:].astype(prop[6][1])
            vertex_all[prop[7][0]] = label[:].astype(prop[7][1])
            print(vertex_all)
            # NOTE: CloudCompare has a bug that only BINARY format is compatible
            ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=False)
            filename_ply: str = 'Pole_' + str(n).zfill(4) + '.ply'
            ply.write(filename_ply)
            n += 1

# Deleting the .laz files
os.chdir(wd)
for split in splits:
    new_wd = os.path.join(wd,split)
    os.chdir(new_wd)
    laz_files2 = [x for x in os.listdir() if x.endswith('.laz')]
    for laz in laz_files2:
        os.remove(os.path.join(new_wd,laz))