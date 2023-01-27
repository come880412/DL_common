"""
Data visualization tool using wandb.
python scripts/utils/data_vis.py <project_name> <dataset_name> <data_path>
"""

import argparse
import wandb
from tqdm import tqdm
import numpy as np
from PIL import Image
import glob
import os

DATA_CLASSES = {i:c for i,c in enumerate(['Urban','Agriculture','Rangeland','Forest','Water','Barren','Unknown'])}
COLORMAP = {(0,255,255):0, (255,255,0):1, (255,0,255):2, (0,255,0):3, (0,0,255):4, (255,255,255):5, (0,0,0):6, (255,0,0):2}
def get_classes_per_image(mask_data, class_labels):
    unique = list(np.unique(mask_data))
    result_dict = {}
    for _class in class_labels.keys():
        result_dict[class_labels[_class]] = int(_class in unique)
    return result_dict

def rgb_to_label(mask_data):
    target = [[COLORMAP[tuple(mask_data[i,j, :])] for j in range(mask_data.shape[0])] for i in range(mask_data.shape[1])]
    return np.array(target)

def _create_table(image_files, mask_files, class_labels):
    "Create a table with the dataset"
    labels = [str(class_labels[_lab]) for _lab in list(class_labels)]
    table = wandb.Table(columns=["File_Name", "Images", "Split"] + labels)
    
    for image_file, mask_file in tqdm(zip(image_files, mask_files), total=len(image_files)):
        image = Image.open(image_file)
        mask_data = np.array(Image.open(mask_file))
        mask_data = rgb_to_label(mask_data)
        class_in_image = get_classes_per_image(mask_data, class_labels)
        table.add_data(
            str(image_file),
            wandb.Image(
                    image,
                    masks={
                        "predictions": {
                            "mask_data": mask_data,
                            "class_labels": class_labels,
                        }
                    }
            ),
            "None", # we don't have a dataset split yet
            *[class_in_image[_lab] for _lab in labels]
        )
    
    return table


def main(args):
    run = wandb.init(project=args.project_name, job_type="upload")
    raw_data_at = wandb.Artifact(args.dataset_name, type="raw_data")

    raw_data_at.add_dir(args.data_path, name='data')

    image_files = glob.glob(os.path.join(args.data_path, "images", "*.jpg"))
    mask_files = glob.glob(os.path.join(args.data_path, "labels", "*.png"))
    image_files.sort()
    mask_files.sort()

    table = _create_table(image_files, mask_files, DATA_CLASSES)
    raw_data_at.add(table, "eda_table")

    run.log_artifact(raw_data_at)
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', default="Test_data_vis", help='Your project name')
    parser.add_argument('--dataset_name', default="Remote_sensing", help='Your dataset name')
    parser.add_argument('--data_path', default="./data/Semantic_Seg", help='Path to the data folder')
    args = parser.parse_args()

    main(args)

