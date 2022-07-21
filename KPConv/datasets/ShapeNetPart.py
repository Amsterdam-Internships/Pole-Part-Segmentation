#
#
#      0==================================0
#      |    Kernel Point Convolutions     |
#      0==================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling ShapeNetPart dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#       Sara Yousefimashhoor - 18-01-2022
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Basic libs
import tensorflow as tf
import numpy as np
import time
import pickle
import json
import os
import shutil

# PLY reader
from plyfile import PlyData, PlyElement
from utils.ply import read_ply, write_ply

# OS functions
from os import listdir, makedirs
from os.path import exists, join

# Dataset parent class
from datasets.common import Dataset

import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#


def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.compute(points, sampleDl=sampleDl, verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.compute(points, features=features, sampleDl=sampleDl, verbose=verbose)
    elif (features is None):
        return cpp_subsampling.compute(points, classes=labels, sampleDl=sampleDl, verbose=verbose)
    else:
        return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=sampleDl, verbose=verbose)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Class Definition
#       \**********************/
#


class ShapeNetPartDataset(Dataset):
    """
    ShapeNetPart dataset for segmentation task. Can handle both unique object class or multi classes models.
    """

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, class_name, input_threads=8):
        """
        Initiation method. Give the name of the object class to segment (for example 'Airplane') or 'multi' to segment
        all objects with a single model.
        """
        Dataset.__init__(self, 'ShapeNetPart_' + class_name)

        ###########################
        # Object classes parameters
        ###########################

        # Dict from object labels to names
        self.label_to_names = {0: 'Airplane',
                               1: 'Bag',
                               2: 'Cap',
                               3: 'Car',
                               4: 'Chair',
                               5: 'Earphone',
                               6: 'Guitar',
                               7: 'Knife',
                               8: 'Lamp',
                               9: 'Laptop',
                               10: 'Motorbike',
                               11: 'Mug',
                               12: 'Pistol',
                               13: 'Rocket',
                               14: 'Skateboard',
                               15: 'Table',
                               16: 'Pole'}

        self.init_labels()

        # List of classes ignored during training (can be empty)
        self.ignored_labels = np.array([])

        # Number of parts for each object
        self.num_parts = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3, 12]

        # Type of dataset (one of the class names or 'multi')
        self.ShapeNetPartType = class_name

        if self.ShapeNetPartType == 'multi':

            # Number of models
            self.network_model = 'multi_segmentation'
            self.num_train = 14007
            self.num_test = 2874

        elif self.ShapeNetPartType in self.label_names:

            # Number of models computed when init_subsample_clouds is called
            self.network_model = 'segmentation'
            self.num_train = None
            self.num_test = None
            self.num_val = None

        else:
            raise ValueError('Unsupported ShapenetPart object class : \'{:s}\''.format(self.ShapeNetPartType))

        ##########################
        # Parameters for the files
        ##########################

        # Path of the folder containing ply files
        self.path = '/home/s2478366/internship_repo/final_internship/Data'

        # Number of threads
        self.num_threads = input_threads

        ###################
        # Prepare ply files
        ###################

        self.prepare_ShapeNetPart_ply()

        return

    def prepare_ShapeNetPart_ply(self):

        print('\nPreparing ply files')
        t0 = time.time()

#         List of class names and corresponding synset
#         category_and_synsetoffset = [['Airplane', '02691156'],
#                                      ['Bag', '02773838'],
#                                      ['Cap', '02954340'],
#                                      ['Car', '02958343'],
#                                      ['Chair', '03001627'],
#                                      ['Earphone', '03261776'],
#                                      ['Guitar', '03467517'],
#                                      ['Knife', '03624134'],
#                                      ['Lamp', '03636649'],
#                                      ['Laptop', '03642806'],
#                                      ['Motorbike', '03790512'],
#                                      ['Mug', '03797390'],
#                                      ['Pistol', '03948459'],
#                                      ['Rocket', '04099429'],
#                                      ['Skateboard', '04225987'],
#                                      ['Table', '04379243'],
#                                      ['Pole', '66666666']]
#         synsetoffset_to_category = {s: n for n, s in category_and_synsetoffset}

        # Collect splits
        # **************

#         # Train split
#         split_file = join(self.path, 'train_test_split', 'shuffled_train_file_list.json')
#         with open(split_file, 'r') as f:
#             train_files = json.load(f)
#         train_files = [name[11:] for name in train_files]

#         # Val split
#         split_file = join(self.path, 'train_test_split', 'shuffled_val_file_list.json')
#         with open(split_file, 'r') as f:
#             val_files = json.load(f)
#         val_files = [name[11:] for name in val_files]

#         # Test split
#         split_file = join(self.path, 'train_test_split', 'shuffled_test_file_list.json')
#         with open(split_file, 'r') as f:
#             test_files = json.load(f)
#         test_files = [name[11:] for name in test_files]

        # Rotate and Scale the plys
        # **************

#        split_files = ['train_ply', 'val_ply','test_ply']
        

#        for split in split_files:
#            new_path = self.path + '/' + split
#            os.chdir(new_path)
#            file_list = os.listdir()
#            file_list = [x for x in file_list if x.endswith('.ply')]
#            source= os.getcwd()
#            if not os.path.exists('modified_'+ split):
#                os.makedirs('modified_'+ split)
#            else:
#                target= os.getcwd()+'/modified_'+ split
#                list_temp = os.listdir(target)
#                for file in list_temp:
#                	if os.path.isfile(file):
#                		os.remove(file)
#            for file in file_list:
#                cloud = PlyData.read(file)
#                points = np.vstack((cloud['vertex']['x'], cloud['vertex']['y'], cloud['vertex']['z'])).T.astype(np.float32)
             
#                # Center and rescale point for 1m radius
#                pmin = np.min(points, axis=0)
#                pmax = np.max(points, axis=0)
#                points -= (pmin + pmax) / 2
#                scale = np.max(np.linalg.norm(points, axis=1))
#                points *= 1.0 / scale
                     
#                # Switch y and z dimensions
#                points = points[:, [0, 2, 1]]
#                labels= np.vstack(cloud['vertex']['label'])
#                final=np.concatenate((points,labels), axis = 1)
#                prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('label', 'u1')]
#                vertex_all = np.empty(len(final), dtype=prop)
#                for i_prop in range(0, 4):
#                    vertex_all[prop[i_prop][0]] = final[:, i_prop]
#                    # NOTE: CloudCompare has a bug that only BINARY format is compatible
#                    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=False)
#                    filename_ply = 'modified_'+ file
#                    ply.write(filename_ply)
#            for file in file_list:
#            	if file.startswith('modified_p'):
#                	shutil.move(os.path.join(source, file), target)
                    

#             # Create folder for this split
#             ply_path = join(self.path, '{:s}_ply'.format(split))
#             if not exists(ply_path):
#                 makedirs(ply_path)


#             N = len(files)
#             class_nums = {n: 0 for n, s in category_and_synsetoffset}
#             for i, file in enumerate(files):

#                 # Get class
#                 synset = file.split('/')[0]
#                 class_name = synsetoffset_to_category[synset]

#                 # Check if file already exists
#                 ply_name = join(ply_path, '{:s}_{:04d}.ply'.format(class_name, class_nums[class_name]))
#                 if exists(ply_name):
#                     class_nums[class_name] += 1
#                     continue

#                 # Get filename
#                 file_name = file.split('/')[1]

#                 # Load points and labels
#                     points = np.vstack((cloud['vertex']['x'], cloud['vertex']['y'], cloud['vertex']['z'])).astype(np.float32)
#                     labels = np.vstack(cloud['vertex']['label']).T.astype(np.int32)

#                 # Center and rescale point for 1m radius
#                     pmin = np.min(points, axis=0)
#                     pmax = np.max(points, axis=0)
#                     points -= (pmin + pmax) / 2
#                     scale = np.max(np.linalg.norm(points, axis=1))
#                     points *= 1.0 / scale

#                     # Switch y and z dimensions
#                     points = points[:, [0, 2, 1]]

#                     # Save in ply format
#                     write_ply(ply, (points, labels), ['x', 'y', 'z', 'label'])

                # Update class number
#                 class_nums[class_name] += 1

                # Display
#                 print('preparing {:s} ply: {:.1f}%'.format(split, 100 * i / N))
        

        
 

#        print('Done preparing the .ply files in {:.1f}s'.format(time.time() - t0))

    def load_subsampled_clouds(self, subsampling_parameter, color_info, intensity_info):
        """
        Presubsample point clouds and load into memory
        """

        if 0 < subsampling_parameter <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        # Initiate containers
        if color_info and intensity_info:
            self.input_feats = {'training': [], 'validation': [], 'test': []}
        elif color_info:
            self.input_colors = {'training': [], 'validation': [], 'test': []}
        elif intensity_info:
            self.input_ints = {'training': [], 'validation': [], 'test': []}
        self.input_points = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': [], 'test': []}
        self.input_point_labels = {'training': [], 'validation': [], 'test': []}

        ################
        # Training files
        ################

        # Restart timer
        t0 = time.time()

        # Load wanted points if possible
        print('\nLoading training points')
        filename = join(self.path, 'train_{:.3f}_record.pkl'.format(subsampling_parameter))

        #if exists(filename):
        #    with open(filename, 'rb') as file:
        #        if color_info and intensity_info:
        #            self.input_feats['training'], 
        #        elif color_info:
        #            self.input_colors['training'], 
        #        elif inensity_info:
        #            self.input_ints['training'], 
        #        self.input_labels['training'], 
        #        self.input_points['training'], 
        #        self.input_point_labels['training'] = pickle.load(file)

        # Else compute them from original points
        #else:

        # Collect training file names
        split_path = join(self.path, '{:s}_ply'.format('train'))
        names = [f[:-4] for f in listdir(split_path) if f[-4:] == '.ply']
        names = np.sort(names)

            # Collect point clouds
        for i, cloud_name in enumerate(names):
                data = read_ply(join(split_path, cloud_name + '.ply'))
                points = np.vstack((data['x'], data['y'], data['z'])).T
                point_labels = data['label']

                if color_info:
                    colors = np.vstack((data['red'], data['green'], data['blue'])).T

                if intensity_info:
                    intensity = np.array(data['intensity'][:, None]) 

                if color_info and intensity_info:
                    features = np.append(colors, intensity, axis=1)
                    if subsampling_parameter > 0:
                        sub_points, sub_feats, sub_labels = grid_subsampling(points, features=features, labels=point_labels,
                                                                              sampleDl=subsampling_parameter)
                        self.input_points['training'] += [sub_points]
                        self.input_feats['training'] += [sub_feats]
                        self.input_point_labels['training'] += [sub_labels]
                    else:
                        self.input_points['training'] += [points]
                        self.input_feats['training'] += [features]
                        self.input_point_labels['training'] += [point_labels]

                elif color_info:
                    if subsampling_parameter > 0:
                        sub_points, sub_colors, sub_labels = grid_subsampling(points, features=colors, labels=point_labels,
                                                                              sampleDl=subsampling_parameter)
                        self.input_points['training'] += [sub_points]
                        self.input_colors['training'] += [sub_colors]
                        self.input_point_labels['training'] += [sub_labels]
                    else:
                        self.input_points['training'] += [points]
                        self.input_colors['training'] += [colors]
                        self.input_point_labels['training'] += [point_labels]

                elif intensity_info:
                    if subsampling_parameter > 0:
                        sub_points, sub_ints, sub_labels = grid_subsampling(points, features=intensity, labels=point_labels,
                                                                           sampleDl=subsampling_parameter)
                        self.input_points['training'] += [sub_points]
                        self.input_ints['training'] += [sub_ints]
                        self.input_point_labels['training'] += [sub_labels]
                    else:
                        self.input_points['training'] += [points]
                        self.input_ints['training'] += [intensity]
                        self.input_point_labels['training'] += [point_labels]
             
                else:
                    if subsampling_parameter > 0:
                        sub_points, sub_labels = grid_subsampling(points, labels=point_labels,
                                                                  sampleDl=subsampling_parameter)
                        self.input_points['training'] += [sub_points]
                        self.input_point_labels['training'] += [sub_labels]
                    else:
                        self.input_points['training'] += [points]
                        self.input_point_labels['training'] += [point_labels]
 
            # Get labels
        label_names = ['_'.join(n.split('_')[:-1]) for n in names]
        self.input_labels['training'] = np.array([self.name_to_label[name] for name in label_names])

            # Save for later use
        with open(filename, 'wb') as file:
            if color_info and intensity_info:
                number = 8
                pickle.dump((self.input_labels['training'],
                             self.input_points['training'],
                             self.input_feats['training'],
                             self.input_point_labels['training']), file)
            elif color_info:
                 number = 7
                 pickle.dump((self.input_labels['training'],
                              self.input_points['training'],
                              self.input_colors['training'],
                              self.input_point_labels['training']), file)
            elif intensity_info:
                number = 5
                pickle.dump((self.input_labels['training'],
                             self.input_points['training'],
                             self.input_ints['training'],
                             self.input_point_labels['training']), file)
            else:
                number = 4
                pickle.dump((self.input_labels['training'],
                             self.input_points['training'],
                             self.input_point_labels['training']), file)


        lengths = [p.shape[0] for p in self.input_points['training']]
        sizes = [l * number * 3 for l in lengths]
        print('{:.1f} MB loaded in {:.1f}s'.format(np.sum(sizes) * 1e-6, time.time() - t0))

        ############
        # Validation files
        ############

        # Restart timer
        t0 = time.time()

        # Load wanted points if possible
        print('\nLoading validation points')
        filename = join(self.path, 'validation_{:.3f}_record.pkl'.format(subsampling_parameter))
        #if exists(filename):
        #    with open(filename, 'rb') as file:
        #        if color_info and intensity_info:
        #            self.input_feats['validation'],
        #        elif color_info:
        #            self.input_colors['validation'],
        #        elif intensity_info:
        #            self.input_ints['validation'], 
        #        self.input_labels['validation'], 
        #        self.input_points['validation'], 
        #        self.input_point_labels['validation'] = pickle.load(file)
        #else:

        # Collect Validation file names
        split_path = join(self.path, '{:s}_ply'.format('val'))
        names = [f[:-4] for f in listdir(split_path) if f[-4:] == '.ply']
        names = np.sort(names)

            # Collect point clouds
        for i, cloud_name in enumerate(names):
                data = read_ply(join(split_path, cloud_name + '.ply'))
                points = np.vstack((data['x'], data['y'], data['z'])).T
                point_labels = data['label']

                if color_info:
                    colors = np.vstack((data['red'], data['green'], data['blue'])).T

                if intensity_info:
                    intensity = np.array(data['intensity'][:, None]) 

                if color_info and intensity_info:
                    features = np.append(colors, intensity, axis=1)
                    if subsampling_parameter > 0:
                        sub_points, sub_feats, sub_labels = grid_subsampling(points, features=features, labels=point_labels,
                                                                              sampleDl=subsampling_parameter)
                        self.input_points['validation'] += [sub_points]
                        self.input_feats['validation'] += [sub_feats]
                        self.input_point_labels['validation'] += [sub_labels]
                    else:
                        self.input_points['validation'] += [points]
                        self.input_feats['validation'] += [features]
                        self.input_point_labels['validation'] += [point_labels]

                elif color_info:
                    if subsampling_parameter > 0:
                        sub_points, sub_colors, sub_labels = grid_subsampling(points, features=colors, labels=point_labels,
                                                                              sampleDl=subsampling_parameter)
                        self.input_points['validation'] += [sub_points]
                        self.input_colors['validation'] += [sub_colors]
                        self.input_point_labels['validation'] += [sub_labels]
                    else:
                        self.input_points['validation'] += [points]
                        self.input_colors['validation'] += [colors]
                        self.input_point_labels['validation'] += [point_labels]

                elif intensity_info:
                    if subsampling_parameter > 0:
                        sub_points, sub_ints, sub_labels = grid_subsampling(points, features=intensity, labels=point_labels,
                                                                           sampleDl=subsampling_parameter)
                        self.input_points['validation'] += [sub_points]
                        self.input_ints['validation'] += [sub_ints]
                        self.input_point_labels['validation'] += [sub_labels]
                    else:
                        self.input_points['validation'] += [points]
                        self.input_ints['validation'] += [intensity]
                        self.input_point_labels['validation'] += [point_labels]
                else:
                    if subsampling_parameter > 0:
                        sub_points, sub_labels = grid_subsampling(points, labels=point_labels,
                                                                  sampleDl=subsampling_parameter)
                        self.input_points['validation'] += [sub_points]
                        self.input_point_labels['validation'] += [sub_labels]
                    else:
                        self.input_points['validation'] += [points]
                        self.input_point_labels['validation'] += [point_labels]  
            # Get labels
        label_names = ['_'.join(n.split('_')[:-1]) for n in names]
        self.input_labels['validation'] = np.array([self.name_to_label[name] for name in label_names])

            # Save for later use
        with open(filename, 'wb') as file:
            if color_info and intensity_info:
                number = 8
                pickle.dump((self.input_labels['validation'],
                             self.input_points['validation'],
                             self.input_feats['validation'],
                             self.input_point_labels['validation']), file)
            elif color_info:
                number = 7
                pickle.dump((self.input_labels['validation'],
                             self.input_points['validation'],
                             self.input_colors['validation'],
                             self.input_point_labels['validation']), file)
            elif intensity_info:
                number = 5
                pickle.dump((self.input_labels['validation'],
                             self.input_points['validation'],
                             self.input_ints['validation'],
                             self.input_point_labels['validation']), file)
            else:
                number = 4
                pickle.dump((self.input_labels['validation'],
                             self.input_points['validation'],
                             self.input_point_labels['validation']), file)

        lengths = [p.shape[0] for p in self.input_points['validation']]
        sizes = [l * number * 3 for l in lengths]
        print('{:.1f} MB loaded in {:.1f}s'.format(np.sum(sizes) * 1e-6, time.time() - t0))

        ############
        # Test files
        ############

        # Restart timer
        t0 = time.time()

        # Load wanted points if possible
        print('\nLoading test points')
        filename = join(self.path, 'test_{:.3f}_record.pkl'.format(subsampling_parameter))
        #if exists(filename):
        #    with open(filename, 'rb') as file:
        #        if color_info and intensity_info:
        #            self.input_feats['test'], 
        #        elif color_info:
        #            self.input_colors['test'],
        #        elif intensity_info:
        #            self.input_ints['test'], 
        #        self.input_labels['test'], 
        #        self.input_points['test'],
        #        self.input_point_labels['test'] = pickle.load(file)

        # Else compute them from original points
        #else:

        # Collect test file names
        split_path = join(self.path, '{:s}_ply'.format('test'))
        names = [f[:-4] for f in listdir(split_path) if f[-4:] == '.ply']
        names = np.sort(names)
        print(names)

            # Collect point clouds
        for i, cloud_name in enumerate(names):
                data = read_ply(join(split_path, cloud_name + '.ply'))
                points = np.vstack((data['x'], data['y'], data['z'])).T
                point_labels = data['label']

                if color_info:
                    colors = np.vstack((data['red'], data['green'], data['blue'])).T

                if intensity_info:
                    intensity = np.array(data['intensity'][:, None]) 

                if color_info and intensity_info:
                    features = np.append(colors, intensity, axis=1)

                    if subsampling_parameter > 0:
                        sub_points, sub_feats, sub_labels = grid_subsampling(points, features=features, labels=point_labels,
                                                                              sampleDl=subsampling_parameter)
                        self.input_points['test'] += [sub_points]
                        self.input_feats['test'] += [sub_feats]
                        self.input_point_labels['test'] += [sub_labels]
                    else:
                        self.input_points['test'] += [points]
                        self.input_feats['test'] += [features]
                        self.input_point_labels['test'] += [point_labels]

                elif color_info:
                    if subsampling_parameter > 0:
                        sub_points, sub_colors, sub_labels = grid_subsampling(points, features=colors, labels=point_labels,
                                                                              sampleDl=subsampling_parameter)
                        self.input_points['test'] += [sub_points]
                        self.input_colors['test'] += [sub_colors]
                        self.input_point_labels['test'] += [sub_labels]
                    else:
                        self.input_points['test'] += [points]
                        self.input_colors['test'] += [colors]
                        self.input_point_labels['test'] += [point_labels]

                elif intensity_info:
                    if subsampling_parameter > 0:
                        sub_points, sub_ints, sub_labels = grid_subsampling(points, features=intensity, labels=point_labels,
                                                                           sampleDl=subsampling_parameter)
                        self.input_points['test'] += [sub_points]
                        self.input_ints['test'] += [sub_ints]
                        self.input_point_labels['test'] += [sub_labels]
                    else:
                        self.input_points['test'] += [points]
                        self.input_ints['test'] += [intensity]
                        self.input_point_labels['test'] += [point_labels]
  
                else:
                    if subsampling_parameter > 0:
                        sub_points, sub_labels = grid_subsampling(points, labels=point_labels,
                                                                  sampleDl=subsampling_parameter)
                        self.input_points['test'] += [sub_points]
                        self.input_point_labels['test'] += [sub_labels]
                    else:
                        self.input_points['test'] += [points]
                        self.input_point_labels['test'] += [point_labels]     
            # Get labels
        label_names = ['_'.join(n.split('_')[:-1]) for n in names]
        self.input_labels['test'] = np.array([self.name_to_label[name] for name in label_names])

            # Save for later use
        with open(filename, 'wb') as file:
            if color_info and intensity_info:
                number = 8
                pickle.dump((self.input_labels['test'],
                             self.input_points['test'],
                             self.input_feats['test'],
                             self.input_point_labels['test']), file)
            elif color_info:
                number = 7
                pickle.dump((self.input_labels['test'],
                             self.input_points['test'],
                             self.input_colors['test'],
                             self.input_point_labels['test']), file)
            elif intensity_info:
                number = 5
                pickle.dump((self.input_labels['test'],
                             self.input_points['test'],
                             self.input_ints['test'],
                             self.input_point_labels['test']), file)
            else:
                number = 4
                pickle.dump((self.input_labels['test'],
                             self.input_points['test'],
                             self.input_point_labels['test']), file)

        lengths = [p.shape[0] for p in self.input_points['test']]
        sizes = [l * number * 3 for l in lengths]
        print('{:.1f} MB loaded in {:.1f}s\n'.format(np.sum(sizes) * 1e-6, time.time() - t0))

        #######################################
        # Eliminate unconsidered object classes
        #######################################

        # Eliminate unconsidered classes
        if self.ShapeNetPartType in self.label_names:
            # Index of the wanted label
            wanted_label = self.name_to_label[self.ShapeNetPartType]

            # Manage training points
            boolean_mask = self.input_labels['training'] == wanted_label
            self.input_labels['training'] = self.input_labels['training'][boolean_mask]
            #print(str(boolean_mask))
            #print(len(boolean_mask))
            #print(self.input_labels['training'])
            #print(len(self.input_labels['training']))
            self.input_points['training'] = np.array(self.input_points['training'])[boolean_mask]
            if color_info and intensity_info:
                self.input_feats['training'] = np.array(self.input_feats['training'])[boolean_mask]
            elif color_info:
                self.input_colors['training'] = np.array(self.input_colors['training'])[boolean_mask]
            elif intensity_info:
                self.input_ints['training'] = np.array(self.input_ints['training'])[boolean_mask]
            self.input_point_labels['training'] = np.array(self.input_point_labels['training'])[boolean_mask]
            self.num_train = len(self.input_labels['training'])

            # Manage validation points
            boolean_mask = self.input_labels['validation'] == wanted_label
            self.input_labels['validation'] = self.input_labels['validation'][boolean_mask]
            self.input_points['validation'] = np.array(self.input_points['validation'])[boolean_mask]
            if color_info and intensity_info:
                self.input_feats['validation'] = np.array(self.input_feats['validation'])[boolean_mask]
            elif color_info:
                self.input_colors['validation'] = np.array(self.input_colors['validation'])[boolean_mask]
            elif intensity_info:
                self.input_ints['validation'] = np.array(self.input_ints['validation'])[boolean_mask]
            self.input_point_labels['validation'] = np.array(self.input_point_labels['validation'])[boolean_mask]
            self.num_val = len(self.input_labels['validation'])

            # Manage test points
            boolean_mask = self.input_labels['test'] == wanted_label
            self.input_labels['test'] = self.input_labels['test'][boolean_mask]
            self.input_points['test'] = np.array(self.input_points['test'])[boolean_mask]
            if color_info and intensity_info:
                self.input_feats['test'] = np.array(self.input_feats['test'])[boolean_mask]
            elif color_info:
                self.input_colors['test'] = np.array(self.input_colors['test'])[boolean_mask]
            elif intensity_info:
                self.input_ints['test'] = np.array(self.input_ints['test'])[boolean_mask]
            self.input_point_labels['test'] = np.array(self.input_point_labels['test'])[boolean_mask]
            self.num_test = len(self.input_labels['test'])

        # Change to 0-based labels
        #self.input_point_labels['training'] = [p_l - 1 for p_l in self.input_point_labels['training']]
        #self.input_point_labels['validation'] = [p_l - 1 for p_l in self.input_point_labels['validation']]
        #self.input_point_labels['test'] = [p_l - 1 for p_l in self.input_point_labels['test']]


        # Test = validation
        #self.input_labels['validation'] = self.input_labels['test']
        #self.input_points['validation'] = self.input_points['test']
        #self.input_point_labels['validation'] = self.input_point_labels['test']

        return

    # Utility methods
    # ------------------------------------------------------------------------------------------------------------------

    def get_batch_gen(self, split, config):

        ################
        # Def generators
        ################

        # Initiate potentials for regular generation
        if not hasattr(self, 'potentials'):
            self.potentials = {}

        # Reset potentials
        self.potentials[split] = np.random.rand(len(self.input_labels[split])) * 1e-3

        def variable_batch_gen_multi():

            # Initiate concatanation lists
            tp_list = []
            tl_list = []
            tpl_list = []
            ti_list = []
            batch_n = 0

            # Initiate parameters depending on the chosen split
            if split == 'training':
                gen_indices = np.random.permutation(self.num_train)

            elif split == 'validation':

                # Get indices with the minimum potential
                val_num = min(self.num_val, config.validation_size * config.batch_num)
                if val_num < self.potentials[split].shape[0]:
                    gen_indices = np.argpartition(self.potentials[split], val_num)[:val_num]
                else:
                    gen_indices = np.random.permutation(val_num)

                # Update potentials
                self.potentials[split][gen_indices] += 1.0

            elif split == 'test':

                # Get indices with the minimum potential
                val_num = min(self.num_test, config.validation_size * config.batch_num)
                if val_num < self.potentials[split].shape[0]:
                    gen_indices = np.argpartition(self.potentials[split], val_num)[:val_num]
                else:
                    gen_indices = np.random.permutation(val_num)

                # Update potentials
                self.potentials[split][gen_indices] += 1.0

            else:
                raise ValueError('Split argument in data generator should be "training", "validation" or "test"')

            # Generator loop
            for i, rand_i in enumerate(gen_indices):

                # Get points
                new_points = self.input_points[split][rand_i].astype(np.float32)
                n = new_points.shape[0]

                # In case batch is full, yield it and reset it
                if batch_n + n > self.batch_limit and batch_n>0:
                    yield (np.concatenate(tp_list, axis=0),
                           np.array(tl_list, dtype=np.int32),
                           np.concatenate(tpl_list, axis=0),
                           np.array(ti_list, dtype=np.int32),
                           np.array([tp.shape[0] for tp in tp_list]))
                    tp_list = []
                    tl_list = []
                    tpl_list = []
                    ti_list = []
                    batch_n = 0

                # Add data to current batch
                tp_list += [new_points]
                tl_list += [self.input_labels[split][rand_i]]
                tpl_list += [np.squeeze(self.input_point_labels[split][rand_i])]
                ti_list += [rand_i]

                # Update batch size
                batch_n += n

            yield (np.concatenate(tp_list, axis=0),
                   np.array(tl_list, dtype=np.int32),
                   np.concatenate(tpl_list, axis=0),
                   np.array(ti_list, dtype=np.int32),
                   np.array([tp.shape[0] for tp in tp_list]))

        def variable_batch_gen_segment():
            color_info = config.color_info
            intensity_info = config.intensity_info
            # Initiate concatanation lists
            if color_info or intensity_info:
                feat_list = []
            tp_list = []
            tpl_list = []
            ti_list = []
            batch_n = 0

            # Initiate parameters depending on the chosen split
            if split == 'training':
                gen_indices = np.random.permutation(self.num_train)
            elif split == 'validation':
                gen_indices = np.arange(self.num_val)
            elif split == 'test':
                gen_indices = np.arange(self.num_test)
            else:
                raise ValueError('Split argument in data generator should be "training", "validation" or "test"')

            # Generator loop
            for i, rand_i in enumerate(gen_indices):

                # Get points
                new_points = self.input_points[split][rand_i].astype(np.float32)
                if color_info and intensity_info:
                    new_features = self.input_feats[split][rand_i].astype(np.float32)
                elif color_info:
                    new_features= self.input_colors[split][rand_i].astype(np.float32)
                elif intensity_info:
                    new_features= self.input_ints[split][rand_i].astype(np.float32)
                n = new_points.shape[0]

                # In case batch is full, yield it and reset it
                if batch_n + n > self.batch_limit and batch_n>0:
                    if color_info or intensity_info:
                        yield (np.concatenate(tp_list, axis=0),
                               np.concatenate(feat_list, axis=0),
                               np.concatenate(tpl_list, axis=0),
                               np.array(ti_list, dtype=np.int32),
                               np.array([tp.shape[0] for tp in tp_list]))
                        tp_list = []
                        tpl_list = []
                        feat_list = []
                        ti_list = []
                        batch_n = 0
                    else:
                        yield (np.concatenate(tp_list, axis=0),
                               np.concatenate(tpl_list, axis=0),
                               np.array(ti_list, dtype=np.int32),
                               np.array([tp.shape[0] for tp in tp_list]))
                        tp_list = []
                        tpl_list = []
                        ti_list = []
                        batch_n = 0

                # Add data to current batch
                tp_list += [new_points]
                if color_info or intensity_info:
                    feat_list += [new_features]
                tpl_list += [np.squeeze(self.input_point_labels[split][rand_i])]
                ti_list += [rand_i]

                # Update batch size
                batch_n += n

            if color_info or intensity_info:
                yield (np.concatenate(tp_list, axis=0),
                       np.concatenate(feat_list, axis=0),
                       np.concatenate(tpl_list, axis=0),
                       np.array(ti_list, dtype=np.int32),
                       np.array([tp.shape[0] for tp in tp_list]))
            else:
                yield (np.concatenate(tp_list, axis=0),
                       np.concatenate(tpl_list, axis=0),
                       np.array(ti_list, dtype=np.int32),
                       np.array([tp.shape[0] for tp in tp_list]))

        ###################
        # Choose generators
        ###################

        if self.ShapeNetPartType == 'multi':
            # Generator types and shapes
            gen_types = (tf.float32, tf.int32, tf.int32, tf.int32, tf.int32)
            gen_shapes = ([None, 3], [None], [None], [None], [None])
            return variable_batch_gen_multi, gen_types, gen_shapes

        elif self.ShapeNetPartType in self.label_names:

            # Generator types and shapes
            if config.color_info and config.intensity_info:
                gen_types =(tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
                gen_shapes = ([None, 3], [None, 4], [None], [None], [None])
            elif config.color_info: 
                gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
                gen_shapes = ([None, 3], [None, 3], [None], [None], [None])
            elif config.intensity_info:
                gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
                gen_shapes = ([None, 3], [None, 1], [None], [None], [None])
            else:
                gen_types = (tf.float32, tf.int32, tf.int32, tf.int32)
                gen_shapes = ([None, 3], [None], [None], [None])
            return variable_batch_gen_segment, gen_types, gen_shapes
        else:
            raise ValueError('Unsupported ShapeNetPart dataset type')

    def get_tf_mapping(self, config):

        def tf_map_multi(stacked_points, stacked_colors, object_labels, point_labels, obj_inds, stack_lengths):
            """
            From the input point cloud, this function compute all the point clouds at each layer, the neighbors
            indices, the pooling indices and other useful variables.
            :param stacked_points: Tensor with size [None, 3] where None is the total number of points
            :param labels: Tensor with size [None] where None is the number of batch
            :param stack_lengths: Tensor with size [None] where None is the number of batch
            """

            # Get batch indice for each point
            batch_inds = self.tf_get_batch_inds(stack_lengths)

            # Augment input points
            stacked_points, scales, rots = self.tf_augment_input(stacked_points,
                                                                 batch_inds,
                                                                 config)

            # First add a column of 1 as feature for the network to be able to learn 3D shapes
            stacked_features = tf.ones((tf.shape(stacked_points)[0], 1), dtype=tf.float32)
            # Then use positions or not
            if config.in_features_dim == 1:
                pass
            elif config.in_features_dim == 4:
                stacked_features = tf.concat((stacked_features, stacked_colors), axis=1)
            elif config.in_features_dim == 7:
                stacked_features = tf.concat((stacked_features, stacked_colors, stacked_points), axis=1)
            else:
                raise ValueError('Only accepted input dimensions are 1, 4 and 7 (without and with RGB/XYZ)')

            # Get the whole input list
            input_list = self.tf_segmentation_inputs(config,
                                                     stacked_points,
                                                     stacked_features,
                                                     point_labels,
                                                     stack_lengths,
                                                     batch_inds,
                                                     object_labels=object_labels)

            # Add scale and rotation for testing
            input_list += [scales, rots, obj_inds]

            return input_list
        def tf_map_segment_xyz(stacked_points, point_labels, obj_inds, stack_lengths):
            """
            From the input point cloud, this function compute all the point clouds at each layer, the neighbors
            indices, the pooling indices and other useful variables.
            :param stacked_points: Tensor with size [None, 3] where None is the total number of points
            :param labels: Tensor with size [None] where None is the number of batch
            :param stack_lengths: Tensor with size [None] where None is the number of batch
            """

            # Get batch indice for each point
            batch_inds = self.tf_get_batch_inds(stack_lengths)

            # Augment input points
            stacked_points, scales, rots = self.tf_augment_input(stacked_points,
                                                                 batch_inds,
                                                                 config)

            # First add a column of 1 as feature for the network to be able to learn 3D shapes
            stacked_features = tf.ones((tf.shape(stacked_points)[0], 1), dtype=tf.float32)

            # Then use positions or not
            if config.in_features_dim == 1:
                pass
            elif config.in_features_dim == 4:
                stacked_features = tf.concat((stacked_features, stacked_points), axis=1)
            else:
                raise ValueError('Only accepted input dimensions are 1 or 4 (without and with XYZ)')

            # Get the whole input list
            input_list = self.tf_segmentation_inputs(config,
                                                     stacked_points,
                                                     stacked_features,
                                                     point_labels,
                                                     stack_lengths,
                                                     batch_inds)

            # Add scale and rotation for testing
            input_list += [scales, rots, obj_inds]

            return input_list

        def tf_map_segment(stacked_points, stacked_features, point_labels, obj_inds, stack_lengths):
            """
            From the input point cloud, this function compute all the point clouds at each layer, the neighbors
            indices, the pooling indices and other useful variables.
            :param stacked_points: Tensor with size [None, 3] where None is the total number of points
            :param labels: Tensor with size [None] where None is the number of batch
            :param stack_lengths: Tensor with size [None] where None is the number of batch
            """

            # Get batch indice for each point
            batch_inds = self.tf_get_batch_inds(stack_lengths)

            # Augment input points
            stacked_points, scales, rots = self.tf_augment_input(stacked_points,
                                                                 batch_inds,
                                                                 config)

            # First add a column of 1 as feature for the network to be able to learn 3D shapes
            stacked_feats = tf.ones((tf.shape(stacked_points)[0], 1), dtype=tf.float32)

            # Then use positions or not
            if config.in_features_dim == 1:
                pass
            elif config.in_features_dim == 4:
                stacked_features = tf.concat((stacked_feats, stacked_points), axis=1)
            elif config.in_features_dim == 5:
                stacked_ints = stacked_features
                stacked_features = tf.concat((stacked_feats, stacked_ints, stacked_points), axis=1)
            elif config.in_features_dim == 7:
                stacked_colors = stacked_features
                stacked_features = tf.concat((stacked_feats, stacked_colors, stacked_points), axis=1)
            elif config.in_features_dim == 8:
                stacked_colors= stacked_features[:,:3]
                stacked_ints = stacked_features[:, 3:]
                stacked_features = tf.concat((stacked_feats, stacked_colors, stacked_ints, stacked_points), axis=1)
            else:
                raise ValueError('Only accepted input dimensions are 5, 7 and 8')

            # Get the whole input list
            input_list = self.tf_segmentation_inputs(config,
                                                     stacked_points,
                                                     stacked_features,
                                                     point_labels,
                                                     stack_lengths,
                                                     batch_inds)

            # Add scale and rotation for testing
            input_list += [scales, rots, obj_inds]

            return input_list

        if self.ShapeNetPartType == 'multi':
            return tf_map_multi

        elif self.ShapeNetPartType in self.label_names:
            if config.intensity_info==False and config.color_info==False:
                return tf_map_segment_xyz
            else:
                return tf_map_segment
        else:
            raise ValueError('Unsupported ShapeNetPart dataset type')


    # Debug methods
    # ------------------------------------------------------------------------------------------------------------------

    def check_input_pipeline_timing(self, config):

        # Create a session for running Ops on the Graph.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Initialise iterator with train data
        self.sess.run(self.train_init_op)
        self.sess.run(self.val_init_op)
        # Run some epochs
        t0 = time.time()
        mean_dt = np.zeros(2)
        last_display = t0
        epoch = 0
        training_step = 0
        while epoch < 100:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = self.flat_inputs

                # Get next inputs
                np_flat_inputs = self.sess.run(ops)
                t += [time.time()]

                # Restructure flatten inputs
                points = np_flat_inputs[:config.num_layers]
                neighbors = np_flat_inputs[config.num_layers:2 * config.num_layers]
                t += [time.time()]

                # Average timing
                mean_dt = 0.99 * mean_dt + 0.01 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display
                if (t[-1] - last_display) > 1.0 / 10:
                    last_display = t[-1]
                    message = 'Step {:08d} : timings {:4.2f} {:4.2f} - {:d} x {:d}'
                    print(message.format(training_step,
                                         1000 * mean_dt[0],
                                         1000 * mean_dt[1],
                                         neighbors[0].shape[0],
                                         neighbors[0].shape[1]))

                training_step += 1

            except tf.errors.OutOfRangeError:
                print('End of train dataset')
                self.sess.run(self.train_init_op)
                epoch += 1

        return

    def check_input_pipeline_2(self, config):

        # Create a session for running Ops on the Graph.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Initialise iterator with train data
        self.sess.run(self.train_init_op)
        self.sess.run(self.val_init_op)

        # Run some epochs
        epoch = 0
        while epoch < 100:

            try:
                # Run one step of the model.
                t = [time.time()]

                # Get next inputs
                np_flat_inputs_1 = self.sess.run(self.flat_inputs)


            except tf.errors.OutOfRangeError:
                print('End of train dataset')
                self.sess.run(self.train_init_op)
                epoch += 1
        return

    def check_debug_input(self, config, path):

        # Get debug file
        file = join(path, 'all_debug_inputs.pkl')
        with open(file, 'rb') as f1:
            inputs = pickle.load(f1)

        # Print inputs
        nl = config.num_layers
        for layer in range(nl):
            print('Layer : {:d}'.format(layer))

            points = inputs[layer]
            neighbors = inputs[nl + layer]
            pools = inputs[2 * nl + layer]
            upsamples = inputs[3 * nl + layer]

            nan_percentage = 100 * np.sum(np.isnan(points)) / np.prod(points.shape)
            print('Points =>', points.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(neighbors)) / np.prod(neighbors.shape)
            print('neighbors =>', neighbors.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(pools)) / (np.prod(pools.shape) + 1e-6)
            print('pools =>', pools.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(upsamples)) / (np.prod(upsamples.shape) + 1e-6)
            print('upsamples =>', upsamples.shape, '{:.1f}% NaN'.format(nan_percentage))

        ind = 4 * nl
        features = inputs[ind]
        nan_percentage = 100 * np.sum(np.isnan(features)) / np.prod(features.shape)
        print('features =>', features.shape, '{:.1f}% NaN'.format(nan_percentage))
        ind += 1
        batch_weights = inputs[ind]
        ind += 1
        in_batches = inputs[ind]
        max_b = np.max(in_batches)
        print(in_batches.shape)
        in_b_sizes = np.sum(in_batches < max_b - 0.5, axis=-1)
        print('in_batch_sizes =>', in_b_sizes)
        ind += 1
        out_batches = inputs[ind]
        max_b = np.max(out_batches)
        print(out_batches.shape)
        out_b_sizes = np.sum(out_batches < max_b - 0.5, axis=-1)
        print('out_batch_sizes =>', out_b_sizes)
        ind += 1
        point_labels = inputs[ind]
        ind += 1
        if config.dataset.startswith('ShapeNetPart_multi'):
            object_labels = inputs[ind]
            nan_percentage = 100 * np.sum(np.isnan(object_labels)) / np.prod(object_labels.shape)
            print('object_labels =>', object_labels.shape, '{:.1f}% NaN'.format(nan_percentage))
            ind += 1
        augment_scales = inputs[ind]
        ind += 1
        augment_rotations = inputs[ind]
        ind += 1

        print('\npoolings and upsamples nums :\n')

        # Print inputs
        nl = config.num_layers
        for layer in range(nl):

            print('\nLayer : {:d}'.format(layer))

            neighbors = inputs[nl + layer]
            pools = inputs[2 * nl + layer]
            upsamples = inputs[3 * nl + layer]

            max_n = np.max(neighbors)
            nums = np.sum(neighbors < max_n - 0.5, axis=-1)
            print('min neighbors =>', np.min(nums))

            if np.prod(pools.shape) > 0:
                max_n = np.max(pools)
                nums = np.sum(pools < max_n - 0.5, axis=-1)
                print('min pools =>', np.min(nums))

            if np.prod(upsamples.shape) > 0:
                max_n = np.max(upsamples)
                nums = np.sum(upsamples < max_n - 0.5, axis=-1)
                print('min upsamples =>', np.min(nums))

        print('\nFinished\n\n')
        time.sleep(0.5)

        self.flat_inputs = [tf.Variable(in_np, trainable=False) for in_np in inputs]

