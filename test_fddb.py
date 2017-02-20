import numpy as np
import argparse
import sys
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','3')
sys.path.insert(0, "/home/zhangboyu/tensorflow/_python_build")
from core.model import P_Net, R_Net, O_Net
from core.imdb import IMDB
from config import config
from core.loader import TestLoader
from core.detector import Detector
from core.fcn_detector import FcnDetector
from core.MtcnnDetector import MtcnnDetector


def test_net(root_path, dataset_path, prefix, epoch,
             batch_size, test_mode="onet",
             thresh=[0.6, 0.6, 0.7], min_face_size=24,
             stride=2, slide_window=False, shuffle=False, vis=False):

    detectors = [None, None, None]

    model_path=['%s-%s'%(x,y) for x,y in zip(prefix,epoch)]
    # load pnet model
    if slide_window:
        PNet = Detector(P_Net, 12, batch_size[0],model_path[0])
    else:
        PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet

    # load rnet model
    if test_mode in ["rnet", "onet"]:
        RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[1] = RNet

    # load onet model
    if test_mode == "onet":
        ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
        detectors[2] = ONet

    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh, slide_window=slide_window)
    for i in range(1,11):
        image_set = "fold-" + str(i).zfill(2)
        imdb = IMDB("fddb", image_set, root_path, dataset_path, 'test')
        gt_imdb = imdb.gt_imdb()
        test_data = TestLoader(gt_imdb)
        all_boxes = mtcnn_detector.detect_face(imdb, test_data, vis=vis)
        imdb.write_results(all_boxes)



def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_path', dest='root_path', help='output data folder',
                        default='./data', type=str)
    parser.add_argument('--dataset_path', dest='dataset_path', help='dataset folder',
                        default='./data/fddb', type=str)
    parser.add_argument('--test_mode', dest='test_mode', help='test net type, can be pnet, rnet or onet',
                        default='onet', type=str)
    parser.add_argument('--prefix', dest='prefix', help='prefix of model name', nargs="+",
                        default=['./data/wider_model/pnet', './data/wider_model/rnet', './data/wider_model/onet'], type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load', nargs="+",
                        default=[16, 16, 16], type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='list of batch size used in prediction', nargs="+",
                        default=[2048, 256, 16], type=int)
    parser.add_argument('--thresh', dest='thresh', help='list of thresh for pnet, rnet, onet', nargs="+",
                        default=[0.6, 0.15, 0.05], type=float)
    parser.add_argument('--min_face', dest='min_face', help='minimum face size for detection',
                        default=20, type=int)
    parser.add_argument('--stride', dest='stride', help='stride of sliding window',
                        default=2, type=int)
    parser.add_argument('--sw', dest='slide_window', help='use sliding window in pnet', action='store_true')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device to train with',
                        default=0, type=int)
    parser.add_argument('--shuffle', dest='shuffle', help='shuffle data on visualization', action='store_true')
    parser.add_argument('--vis', dest='vis', help='turn on visualization', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print 'Called with argument:'
    print args
    test_net(args.root_path, args.dataset_path, args.prefix,
             args.epoch, args.batch_size, args.test_mode,
             args.thresh, args.min_face, args.stride,
             args.slide_window, args.shuffle, args.vis)
