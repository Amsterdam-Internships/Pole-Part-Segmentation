from plyfile import PlyData, PlyElement
import numpy as np
import os
import shutil
import laspy

# Sets the target directory
os.chdir(r'/home/sara/Desktop/Internship/Experiment Results/04- Default params - xyz rgb i/Prepared Data')
wd = os.getcwd()

# Checks if the folder are already written
splits = ['train_ply', 'test_ply', 'val_ply']
for split in splits:
    if not os.path.exists(split):
        os.makedirs(split)
    else:
        sp = os.listdir(os.path.join(wd, split))
        for file in sp:
            if os.path.isfile(file):
                os.remove(file)

# Collects all the labelled laz files
laz_path = r'/home/sara/Desktop/Internship/labelled_pole_dataset-20220117T100957Z-001/Laz/data'
laz_files = os.listdir(laz_path)
laz_files = [x for x in laz_files if x.endswith('.laz')]

# Split the laz files like the Set3_1
split_path = r'/home/sara/Desktop/Internship/Experiment Results/Data/Set3_1'
for s in splits:
    s_path = os.path.join(split_path, s)
    files = os.listdir(s_path)
    files = [x for x in files if x.endswith('.laz')]
    files = set(files)
    for laz in laz_files:
        if laz in files:
            shutil.copy(os.path.join(laz_path, laz), os.path.join(wd, s))

    os.chdir(os.path.join(wd, s))
    directory = os.listdir()
    for file in directory:
        if file.endswith('.laz'):
            cloud = laspy.read(file)

            # Collects coordinate information
            xyz = (np.vstack((cloud.x, cloud.y, cloud.z)).T.astype('float32'))
            # Center and rescale point for 1m radius
            pmin = np.min(xyz[:, :3], axis=0)
            pmax = np.max(xyz[:, :3], axis=0)
            xyz[:, :3] -= (pmin + pmax) / 2

            # Collects RGB info
            rgb = (np.vstack((cloud.red, cloud.green, cloud.blue)).T.astype('float32'))
            # normalizing the color values
            rgb[:,:] /= 2**16 - 1

            # Collects Intensity information
            intensity = (np.array((cloud.intensity).T.astype('float32')))
            intensity /= 2 ** 16 - 1

            # Reads the labels
            label = (np.array((cloud.label).astype('uint8')))
            for i in range(len(cloud.label)):
                cloud.label[i] += 1
            all = np.append(xyz, rgb, axis=1)
            alll = np.append(all, intensity[:, None], axis=1)
            final_data = np.append(alll, label[:, None], axis=1)
            # print(final_data)

            print('Saving BINARY PLY lidar data for ', file)
            # Setting the data type
            prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'f4'), ('green', 'f4'), ('blue', 'f4'),('intensity', 'f4'), ('label', 'u1')]
            #prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4'), ('label', 'u1')]
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
            filename_ply: str = file[:-3] + 'ply'
            ply.write(filename_ply)

# Moves the .ply file format to a new folder
os.chdir(wd)
splits = ['train_ply', 'test_ply', 'val_ply']
for split in splits:
    folder_name = 'modified_' + split
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        sp = os.listdir(folder_name)
        for file in sp:
            if os.path.isfile(file):
                os.remove(file)
    master_path = os.path.join(wd, split)
    master_files = os.listdir(master_path)
    target_path = os.path.join(master_path,folder_name)
    for file in master_files:
        if file.endswith('.ply'):
            shutil.move(os.path.join(master_path, file), folder_name)


