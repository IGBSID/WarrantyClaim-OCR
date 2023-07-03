from training_pipeline.utils.validation import inference_of_whole_pipeline
from TF_docker_setup import inference_pipeline
import argparse
import os
from pycocotools.coco import COCO

parser = argparse.ArgumentParser()

if __name__ == "__main__":
    parser.add_argument("-c", "--path2config", help = "path to config file", required=True)
    parser.add_argument("-i", "--path2images", help = "path to images", required=True)
    parser.add_argument("-a", "--path2annotations", help = "path to image annotations", required=True)
    parser.add_argument("-o", "--path2outdir", help = "Dir to save output images", required=True)

    # Read arguments from command line
    args = parser.parse_args()

    path2config = os.path.abspath(args.path2config)
    path2images = os.path.abspath(args.path2images)
    path2annotations = os.path.abspath(args.path2annotations)
    path2outdir = os.path.abspath(args.path2outdir)

    m = inference_pipeline.Model(path2config)
    t = inference_pipeline.TagNumberExtractionPipeline(m)
        
    inference_of_whole_pipeline(path2images=path2images,
                            annotations=path2annotations,
                            path2outdir=path2outdir,
                            Extractor=t,
                            box_th=0.7)
    
 