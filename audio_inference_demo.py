# coding: utf-8
# author: luuil@outlook.com

r"""Test on audio model."""

from __future__ import print_function
import sys
sys.path.append('./audio')

import tensorflow as tf
import numpy as np
from os.path import join as pjoin
from sklearn.metrics import accuracy_score

import audio_model
import audio_params
import audio_util as util
from audio_feature_extractor import VGGishExtractor
from audio_records import RecordsParser


CKPT_DIR = audio_params.AUDIO_CHECKPOINT_DIR
CKPT_NAME = audio_params.AUDIO_CHECKPOINT_NAME
META = pjoin(CKPT_DIR, audio_params.AUDIO_TRAIN_NAME, '{ckpt}.meta'.format(ckpt=CKPT_NAME))
CKPT = pjoin(CKPT_DIR, audio_params.AUDIO_TRAIN_NAME, CKPT_NAME)

VGGISH_CKPT = audio_params.VGGISH_CHECKPOINT
VGGISH_PCA = audio_params.VGGISH_PCA_PARAMS

SESS_CONFIG = tf.ConfigProto(allow_soft_placement=True)
SESS_CONFIG.gpu_options.allow_growth = True

def _restore_from_meta_and_ckpt(sess, meta, ckpt):
    """Restore graph from meta file and variables from ckpt file."""
    saver = tf.train.import_meta_graph(meta)
    saver.restore(sess, ckpt)


def _restore_from_defined_and_ckpt(sess, ckpt):
    """Restore graph from define and variables from ckpt file."""
    with sess.graph.as_default():
        audio_model.define_audio_slim(training=False)
        audio_model.load_audio_slim_checkpoint(sess, ckpt)

def inference_wav(wav_file):
    """Test audio model on a wav file."""
    label = util.urban_labels([wav_file])[0]
    graph = tf.Graph()
    with tf.Session(graph=graph, config=SESS_CONFIG) as sess:
        with VGGishExtractor(VGGISH_CKPT,
                         VGGISH_PCA,
                         audio_params.VGGISH_INPUT_TENSOR_NAME,
                         audio_params.VGGISH_OUTPUT_TENSOR_NAME) as ve:
            vggish_features = ve.wavfile_to_features(wav_file)
        assert vggish_features is not None
        labels = [label] * vggish_features.shape[0]
        
        # restore graph
        # _restore_from_meta_and_ckpt(sess, META, CKPT)
        _restore_from_defined_and_ckpt(sess, CKPT)

        # get input and output tensor
        # graph = tf.get_default_graph()
        inputs = graph.get_tensor_by_name(audio_params.AUDIO_INPUT_TENSOR_NAME)
        outputs = graph.get_tensor_by_name(audio_params.AUDIO_OUTPUT_TENSOR_NAME)
        
        predictions = sess.run(outputs, feed_dict={inputs: vggish_features})
        idxes = np.argmax(predictions, 1)
        probs = np.max(predictions, 1)
        print(predictions)
        print(idxes)
        print(labels)
        print(probs)
        acc = accuracy_score(labels, idxes)
        print('acc:', acc)


def inference_on_test():
    """Test audio model on test dataset."""
    graph = tf.Graph()
    with tf.Session(graph=graph, config=SESS_CONFIG) as sess:
        rp = RecordsParser([audio_params.TF_RECORDS_TEST], 
            audio_params.NUM_CLASSES, feature_shape=None)
        test_iterator, test_batch = rp.iterator(is_onehot=True, batch_size=1)


        # restore graph: 2 ways to restore, both will working
        # _restore_from_meta_and_ckpt(sess, META, CKPT)
        _restore_from_defined_and_ckpt(sess, CKPT)

        # get input and output tensor
        # graph = tf.get_default_graph()
        inputs = graph.get_tensor_by_name(audio_params.AUDIO_INPUT_TENSOR_NAME)
        outputs = graph.get_tensor_by_name(audio_params.AUDIO_OUTPUT_TENSOR_NAME)

        sess.run(test_iterator.initializer)
        predicted = []
        groundtruth = []
        while True:
            try:
                # feature: [batch_size, num_features]
                # label: [batch_size, num_classes]
                te_features, te_labels = sess.run(test_batch)
            except tf.errors.OutOfRangeError:
                break
            predictions = sess.run(outputs, feed_dict={inputs: te_features})
            predicted.extend(np.argmax(predictions, 1))
            groundtruth.extend(np.argmax(te_labels, 1))
            # print(te_features.shape, te_labels, te_labels.shape)

        right = accuracy_score(groundtruth, predicted, normalize=False) # True: return prob
        print('all: {}, right: {}, wrong: {}, acc: {}'.format(
            len(predicted), right, len(predicted) - right, right/(len(predicted))))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    inference_wav('./data/wav/13230-0-0-1.wav')
    inference_on_test()

