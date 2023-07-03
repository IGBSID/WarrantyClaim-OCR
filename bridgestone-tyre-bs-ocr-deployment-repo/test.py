import cv2
import argparse
import os
import configparser
from inference_pipeline.pipeline import Pipeline
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configFilePath", help = "path to config file",required=True)
    parser.add_argument("-i", "--imagePath", help = "path to config file",required=True)
    args = parser.parse_args()
    P= Pipeline(args.configFilePath)
    visualization_image, tag_number = P.run(args.imagePath)
    print(visualization_image.shape)
    # cv2.imshow("image_viz", visualization_image)
    # cv2.waitKey(0)
    print(tag_number)