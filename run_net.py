import sys
import glob
import numpy as np
import os
from pspnet import *
from cityscapes_labels import name2label
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i', '--input', help="get input images from directory")
parser.add_argument('-o', '--output', help="save output to directory")
args = parser.parse_args()



fnames = glob.glob(os.path.expanduser(os.path.join(args.input, "*/*")))
print("found {} files".format(len(fnames)))

EVALUATION_SCALES = [1.0]
pspnet = PSPNet101(nb_classes=19, input_shape=(713, 713),
                           weights="pspnet101_cityscapes")

road_id = name2label["road"].trainId

for i in range(len(fnames)):
    fname = fnames[i]
    img = misc.imread(fname)
    
    class_scores = predict_multi_scale(img, pspnet, EVALUATION_SCALES, False, False)
    road_image = class_scores[:, :, road_id]
    class_image = np.argmax(class_scores, axis=2)
    pm = np.max(class_scores, axis=2)
    colored_class_image = utils.color_class_image(class_image, "pspnet101_cityscapes")
    # colored_class_image is [0.0-1.0] img is [0-255]
    alpha_blended = 0.5 * colored_class_image + 0.5 * img
    
    cam_name = os.path.basename(os.path.dirname(fname))
    directory = os.path.expanduser(os.path.join(args.output, cam_name))
    output_path = os.path.join(directory, os.path.basename(fname))
    print(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filename, ext = splitext(output_path)
    misc.imsave(filename + "_seg" + ext, colored_class_image)
    misc.imsave(filename + "_probs" + ext, pm)
    misc.imsave(filename + "_road_probs" + ext, road_image)
    np.save(filename + "_road_probs.npy", road_image)
    misc.imsave(filename + "_seg_blended" + ext, alpha_blended)
