#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the inference of the custom data
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Sara Yousefimashhoor - 01/03/2022
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import tensorflow as tf
import numpy as np
from os import makedirs, listdir
from os.path import exists, join
import time
from sklearn.neighbors import KDTree

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
# from utils.metrics import IoU_from_confusions
# from sklearn.metrics import confusion_matrix

from tensorflow.python.client import timeline
import json


# ----------------------------------------------------------------------------------------------------------------------
#
#           Tester Class
#       \******************/
#

class TimeLiner:

    def __init__(self):
        self._timeline_dict = None

    def update_timeline(self, chrome_trace):

        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)

        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict

        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)


class ModelInference:

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, model, restore_snap=None):

        # Tensorflow Saver definition
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='KernelPointNetwork')
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)

        # Create a session for running Ops on the Graph.
        on_CPU = False
        if on_CPU:
            cProto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            cProto = tf.ConfigProto()
            cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Name of the snapshot to restore to (None if you want to start from beginning)
        # restore_snap = join(self.saving_path, 'snapshots/snap-40000')
        if (restore_snap is not None):
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)

        # Add a softmax operation for predictions
        self.prob_logits = tf.nn.softmax(model.logits)


    def infer_segmentation(self, model, dataset):

        ##################
        # Pre-computations
        ##################

        print('Preparing test structures')
        t1 = time.time()

        # Collect original inference file names
        path = "/home/s2478366/internship_repo/KPConv_Experiment/Data/ShapeNetPart/shapenetcore_partanno_segmentation_benchmark_v0/Set2_1"
        original_path = join(path, 'inf_ply')
        object_name = 'Pole'
        infer_names = [f[:-4] for f in listdir(original_path) if f[-4:] == '.ply' and object_name in f]
        infer_names = np.sort(infer_names)

        # original_labels = []
        original_points = []
        projection_inds = []
        for i, cloud_name in enumerate(infer_names):

            # Read data in ply file
            data = read_ply(join(original_path, cloud_name + '.ply'))
            points = np.vstack((data['x'], data['y'], data['z'])).T
            # original_labels += [data['label'] - 1]
            original_points += [points]

            # Create tree structure and compute neighbors
            tree = KDTree(dataset.input_points['inference'][i])
            projection_inds += [np.squeeze(tree.query(points, return_distance=False))]

        t2 = time.time()
        print('Done in {:.1f} s\n'.format(t2 - t1))

        ##########
        # Initiate
        ##########

        # Test saving path
        if model.config.saving:
            infer_path = join('infer', model.saving_path.split('/')[-1])
            if not exists(infer_path):
                makedirs(infer_path)
        else:
            infer_path = None

        # Initialise iterator with test data
        self.sess.run(dataset.inf_init_op)

        # Initiate result containers
        average_predictions = [np.zeros((1, 1), dtype=np.float32) for _ in infer_names]

        #####################
        # Network predictions
        #####################

        mean_dt = np.zeros(2)
        last_display = time.time()
        # Run model on all test examples
        # ******************************

        # Initiate result containers
        all_predictions = []
        all_points = []
        #all_scales = []
        #all_rots = []

        while True:
            try:

                    # Run one step of the model
                t = [time.time()]
                ops = (self.prob_logits,
                       model.inputs['in_batches'],
                       model.inputs['points'])
                       #model.inputs['augment_scales'],
                       #model.inputs['augment_rotations'])
                preds, batches, points = self.sess.run(ops)
                t += [time.time()]

                    # Stack all predictions for each class separately
                max_ind = np.max(batches)
                for b_i, b in enumerate(batches):

                    # Eliminate shadow indices
                    b = b[b < max_ind - 0.5]

                    # Get prediction (only for the concerned parts)
                    predictions = preds[b]

                    # Stack all results
                    all_predictions += [predictions]
                    all_points += [points[0][b]]
                    #all_scales += [s[b_i]]
                    #all_rots += [R[b_i, :, :]]

                    # Average timing
                    t += [time.time()]
                    mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                    # Display
                    # if (t[-1] - last_display) > 1.0:
                    #     last_display = t[-1]
                    #     message = 'Vote {:d} : {:.1f}% (timings : {:4.2f} {:4.2f})'
                    #     print(message.format(v,
                    #                          100 * len(all_predictions) / len(original_labels),
                    #                          1000 * (mean_dt[0]),
                    #                          1000 * (mean_dt[1])))

            except tf.errors.OutOfRangeError:
                break

            # Project predictions on original point clouds
            # ********************************************
            #
            # print('\nGetting test confusions')
            # t1 = time.time()
            #
        proj_predictions = []
        for i, cloud_name in enumerate(infer_names):

                #Interpolate prediction from current positions to original points
            proj_predictions += [all_predictions[i][projection_inds[i]]]
            # Average prediction across votes
            # average_predictions[i] = average_predictions[i] + (proj_predictions[i] - average_predictions[i]) / (v + 1)

            # # Save the best/worst segmentations per class
            # # *******************************************
            #
            print('Saving inference point cloud')
            t1 = time.time()

            # Save the names in a file
            # Save the clouds
            obj_path = join(infer_path, object_name)
            for i, cloud_name in enumerate(infer_names):
                filename = join(obj_path, cloud_name)
                preds = np.argmax(proj_predictions[i], axis=1).astype(np.int32)
                write_ply(filename,
                          [original_points[i], preds],
                          ['x', 'y', 'z', 'pre'])

            t2 = time.time()
            print('Done in {:.1f} s\n'.format(t2 - t1))

            # Display results
            # ***************

            # print('Objs | Classes | Air  Bag  Cap  Car  Cha  Ear  Gui  Kni  Lam  Lap  Mot  Mug  Pis  Roc  Ska  Tab   Pole')
            # print('-----|---------|---------------------------------------------------------------------------------------')
            #
            # s = '---- | ---- | '
            # for obj in dataset.label_names:
            #     if obj == object_name:
            #         s += '{:5.2f} '.format(100 * np.mean(mIoUs))
            #     else:
            #         s += '---- '
            # print(s + '\n')

            # Initialise iterator with inference data
            self.sess.run(dataset.infer_init_op)

        return

