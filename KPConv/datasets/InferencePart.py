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
from datasets.common_inference import Dataset

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


class InferenceDataset(Dataset):
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

        # if self.ShapeNetPartType == 'multi':

            # Number of models
            # self.network_model = 'multi_segmentation'
            # self.num_train = 14007
            # self.num_test = 2874

        if self.ShapeNetPartType in self.label_names:

            # Number of models computed when init_subsample_clouds is called
            self.network_model = 'segmentation'
            # self.num_train = None
            # self.num_test = None
            # self.num_val = None
            self.num_inf = None

        else:
            raise ValueError('Unsupported ShapenetPart object class : \'{:s}\''.format(self.ShapeNetPartType))

        ##########################
        # Parameters for the files
        ##########################

        # Path of the folder containing ply files
        self.path = '/home/s2478366/internship_repo/KPConv_Experiment/Data/ShapeNetPart/shapenetcore_partanno_segmentation_benchmark_v0/Set2_1'

        # Number of threads
        self.num_threads = input_threads

        ###################
        # Prepare ply files
        ###################

        self.prepare_inference_ply()

        return

    def prepare_inference_ply(self):

        print('\nPreparing inference ply files')
        t0 = time.time()
        split = 'inf_ply'
        # Create folder for this split
        ply_path = join(self.path, '{:s}_ply'.format(split))
        if not exists(ply_path):
            makedirs(ply_path)

        new_path = self.path + '/' + split
        os.chdir(new_path)
        file_list = os.listdir()
        file_list = [x for x in file_list if x.endswith('.ply')]
        source = os.getcwd()
        if not os.path.exists('modified_'+ split):
            os.makedirs('modified_'+ split )
            target = source + '/modified_'+ split
        else:
            target = os.getcwd()+'/modified_'+ split
            list_temp = os.listdir(target)
            print("\nlist_temp\n", list_temp)
        for file in list_temp:
            print('file in list_temp', file)
            if os.path.isfile(file):
                os.remove(file)
        for file in file_list:
            if file.startswith('Pole'):
                cloud = PlyData.read(file)
                points = np.vstack((cloud['vertex']['x'], cloud['vertex']['y'], cloud['vertex']['z'])).T.astype(np.float32)
             
                # Center the points around zero
                pmin = np.min(points, axis=0)
                pmax = np.max(points, axis=0)
                points -= (pmin + pmax) / 2

                prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
                vertex_all = np.empty(len(points), dtype=prop)
                for i_prop in range(0, 3):
                    vertex_all[prop[i_prop][0]] = points[:, i_prop]
                    # NOTE: CloudCompare has a bug that only BINARY format is compatible
                    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=False)
                    filename_ply = 'modified_'+ file
                    ply.write(filename_ply)
                    shutil.move(os.path.join(source, file), target)

        print('Done preparing the inference .ply files in {:.1f}s'.format(time.time() - t0))

    def load_subsampled_clouds(self, subsampling_parameter):
        """
        Presubsample point clouds and load into memory
        """

        if 0 < subsampling_parameter <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        # Initiate containers
        self.input_points = {'inference': []}
        # self.input_labels = {'inference': []}
        # self.input_point_labels = {'inference': []}

        ################
        # Inference files
        ################

        # Restart timer
        t0 = time.time()

        # Load wanted points if possible
        print('\nLoading inference points')
        filename = join(self.path, 'inference_{:.3f}_record.pkl'.format(subsampling_parameter))

        if exists(filename):
            with open(filename, 'rb') as file:
                self.input_points['inference'] = pickle.load(file)
                print(self.input_points['inference'])

        # Else compute them from original points
        else:
            # Collect training file names
            split_path = join(self.path, '{:s}_ply'.format('inf'))
            names = [f[:-4] for f in listdir(split_path) if f[-4:] == '.ply']
            names = np.sort(names)

            # Collect point clouds
            for i, cloud_name in enumerate(names):
                print(cloud_name)
                data = read_ply(join(split_path, cloud_name + '.ply'))
                points = np.vstack((data['x'], data['y'], data['z'])).T
                #colors = np.vstack((data['red'], data['blue'], data['green'])).T
                if subsampling_parameter > 0:
                    sub_points = grid_subsampling(points, sampleDl=subsampling_parameter)
                    self.input_points['inference'] += [sub_points]
                else:
                    self.input_points['inference'] += [points]
            # Get labels
            # label_names = ['_'.join(n.split('_')[:-1]) for n in names]
            print(self.input_points['inference'])
            # Save for later use
            with open(filename, 'wb') as file:
                pickle.dump((self.input_points['inference']), file)

        lengths = [p.shape[0] for p in self.input_points['inference']]
        sizes = [l * 3 * 3 for l in lengths]
        print('{:.1f} MB loaded in {:.1f}s'.format(np.sum(sizes) * 1e-6, time.time() - t0))


        #######################################
        # Eliminate unconsidered object classes
        #######################################
        #
        # # Eliminate unconsidered classes
        if self.ShapeNetPartType in self.label_names:
        #     # Index of the wanted label
        #     wanted_label = self.name_to_label[self.ShapeNetPartType]
        #
        #     # Manage training points
        #     boolean_mask = self.input_labels['training'] == wanted_label
        #     self.input_labels['training'] = self.input_labels['training'][boolean_mask]
        #     self.input_points['training'] = np.array(self.input_points['training'])[boolean_mask]
        #     self.input_point_labels['training'] = np.array(self.input_point_labels['training'])[boolean_mask]
        #     self.num_train = len(self.input_labels['training'])
        #
        #     # Manage validation points
        #     boolean_mask = self.input_labels['validation'] == wanted_label
        #     self.input_labels['validation'] = self.input_labels['validation'][boolean_mask]
        #     self.input_points['validation'] = np.array(self.input_points['validation'])[boolean_mask]
        #     self.input_point_labels['validation'] = np.array(self.input_point_labels['validation'])[boolean_mask]
        #     self.num_val = len(self.input_labels['validation'])
        #
        #     # Manage test points
        #     boolean_mask = self.input_labels['test'] == wanted_label
        #     self.input_labels['test'] = self.input_labels['test'][boolean_mask]
        #     self.input_points['test'] = np.array(self.input_points['test'])[boolean_mask]
        #     self.input_point_labels['test'] = np.array(self.input_point_labels['test'])[boolean_mask]
            self.num_inf = len(self.input_points['inference'])
        #
        # # Change to 0-based labels
        # self.input_point_labels['training'] = [p_l - 1 for p_l in self.input_point_labels['training']]
        # self.input_point_labels['validation'] = [p_l - 1 for p_l in self.input_point_labels['validation']]
        # self.input_point_labels['test'] = [p_l - 1 for p_l in self.input_point_labels['test']]

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
        self.potentials[split] = np.random.rand(len(self.input_points[split])) * 1e-3

        def variable_batch_gen_segment():

            # Initiate concatanation lists
            tp_list = []
            tpl_list = []
            ti_list = []
            batch_n = 0

            # Initiate parameters depending on the chosen split
            if split == 'inference':
                gen_indices = np.random.permutation(self.num_inf)
            else:
                raise ValueError('Split argument in data generator should be "inference"')

            # Generator loop
            for i, rand_i in enumerate(gen_indices):

                # Get points
                new_points = self.input_points[split][rand_i].astype(np.float32)
                n = new_points.shape[0]

                # In case batch is full, yield it and reset it
                if batch_n + n > self.batch_limit:
                    yield (np.concatenate(tp_list, axis=0),
                           np.array(ti_list, dtype=np.int32),
                           np.array([tp.shape[0] for tp in tp_list]))
                    tp_list = []
                    # tpl_list = []
                    ti_list = []
                    batch_n = 0

                # Add data to current batch
                tp_list += [new_points]
                # tpl_list += [np.squeeze(self.input_point_labels[split][rand_i])]
                ti_list += [rand_i]

                # Update batch size
                batch_n += n

            yield (np.concatenate(tp_list, axis=0),
                   np.array(ti_list, dtype=np.int32),
                   np.array([tp.shape[0] for tp in tp_list]))

        ###################
        # Choose generators
        ###################

        if self.ShapeNetPartType in self.label_names:

            # Generator types and shapes
            gen_types = (tf.float32, tf.int32, tf.int32, tf.int32)
            gen_shapes = ([None, 3], [None], [None], [None])
            return variable_batch_gen_segment, gen_types, gen_shapes
        else:
            raise ValueError('Unsupported inference dataset type')

    def get_tf_mapping(self, config):

         def tf_map_segment(stacked_points,point_labels, obj_inds, stack_lengths):
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
            #stacked_points, scales, rots = self.tf_augment_input(stacked_points,batch_inds,config)

            # First add a column of 1 as feature for the network to be able to learn 3D shapes
            stacked_features = tf.ones((tf.shape(stacked_points)[0], 1), dtype=tf.float32)

            # Then use positions or not
            if config.in_features_dim == 1:
                pass
            elif config.in_features_dim == 4:
                stacked_features = tf.concat((stacked_features, stacked_points), axis=1)
            else:
                raise ValueError('Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)')

            # Get the whole input list
            input_list = self.tf_segmentation_inputs(config,
                                                     stacked_points,
                                                     stacked_features,
                                                     point_labels,
                                                     stack_lengths,
                                                     batch_inds)

            # Add scale and rotation for testing
            input_list += [obj_inds]

            return input_list

         if self.ShapeNetPartType in self.label_names:
            return tf_map_segment
         else:
            raise ValueError('Unsupported ShapeNetPart dataset type')


