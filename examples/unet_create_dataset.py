
import logging
from pathlib import Path
import sys
from glob import glob


import nibabel as nib
import numpy as np


import monai
from monai.data import create_test_image_3d


def main(dataset_dir):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # create a temporary directory and 40 random image, mask pairs
    image_paths = []
    segmentation_paths = []
    
    print(f"generating synthetic data to {dataset_dir} (this may take a while)")
    for i in range(80):
        im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1)

        n = nib.Nifti1Image(im, np.eye(4))
        image_path = dataset_dir / f"im{i:d}.nii.gz" 
        nib.save(n, image_path)
        image_paths.append(image_path)

        n = nib.Nifti1Image(seg, np.eye(4))
        segmentation_path = dataset_dir / f"seg{i:d}.nii.gz"
        nib.save(n, segmentation_path)
        segmentation_paths.append(segmentation_path)
    
        
    
if __name__ == '__main__':
    dataset_dir = Path('dataset')
    dataset_dir.mkdir(exist_ok=True, parents=True)
    main(dataset_dir)