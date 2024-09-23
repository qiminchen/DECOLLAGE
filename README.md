# DECOLLAGE
Pytorch Implementation of [[ECCV2024] DECOLLAGE: 3D Detailization by Controllable, Localized, and Learned Geometry Enhancement](https://arxiv.org/abs/2409.06129), [Qimin Chen](https://qiminchen.github.io/), [Zhiqin Chen](https://czq142857.github.io/), [Vladimir G. Kim](http://www.vovakim.com/), [Noam Aigerman](https://noamaig.github.io/), [Hao Zhang](http://www.cs.sfu.ca/~haoz/), [Siddhartha Chaudhuri](https://www.cse.iitb.ac.in/~sidch/).

### [Paper](https://arxiv.org/abs/2409.06129)  |  [Project page](https://qiminchen.github.io/decollage/)

<img src='teaser.svg' />

## Citation
If you find our work useful in your research, please consider citing (to be updated):

	  @misc{chen2024decollage3ddetailizationcontrollable,
        title={DECOLLAGE: 3D Detailization by Controllable, Localized, and Learned Geometry Enhancement}, 
        author={Qimin Chen and Zhiqin Chen and Vladimir G. Kim and Noam Aigerman and Hao Zhang and Siddhartha Chaudhuri},
        year={2024},
        eprint={2409.06129},
        archivePrefix={arXiv},
      }

## Dependencies
Requirements:
- Python 3.7 with numpy, pillow, h5py, scipy, sklearn and Cython
- [PyTorch 1.9](https://pytorch.org/get-started/locally/) (other versions may also work)
- [PyMCubes](https://github.com/pmneila/PyMCubes) (for marching cubes)
- [OpenCV-Python](https://opencv-python-tutroals.readthedocs.io/en/latest/) (for reading and writing images)

Build Cython module:
```
python setup.py build_ext --inplace
```

## Datasets and pre-trained weights
We provide the ready-to-use datasets here. Note that we only use 16 chairs, 16 tables, and 5 plants from ShapeNet; and 5 buildings, 3 cakes, and 3 crystals from 3D Warehouse for training. The training coarse voxels are obtained via data augmentation.

- [DECOLLAGE data (will be updated shortly)]()

We also provide the pre-trained network weights.

- [DECOLLAGE checkpoint (will be updated shortly)]()

## Training
For chair and table style mixing:
```
python main.py --data_style style_seg_chair_table_32 --data_content dummy --data_dir ./data/03001627_04379243/ --alpha 0.5 --beta 10.0 --input_size 16 --output_size 256 --train --gpu 0 --epoch 20 --sample_dir ./your_sample_dir/
```
For plants, buildings, cakes and crystals style mixing:
```
python main.py --data_style style_seg_plant_building_cake_crystal_16 --data_content dummy --data_dir ./data/03593526_00000000_00000001/ --alpha 0.5 --beta 10.0 --input_size 16 --output_size 256 --train --gpu 0 --epoch 20 --sample_dir ./your_sample_dir/
```

## Testing
For chair and table style mixing:
```
python main.py --data_style style_seg_chair_table_32 --data_content dummy --data_dir ./data/03001627_04379243/ --input_size 16 --output_size 256 --test --gpu 0 --checkpoint_model ./path/to/checkpoint.pt/
```
For plants, buildings, cakes and crystals style mixing:
```
python main.py --data_style style_seg_plant_building_cake_crystal_16 --data_content dummy --data_dir ./data/03593526_00000000_00000001/ --input_size 16 --output_size 256 --test --gpu 0 --checkpoint_model ./path/to/checkpoint.pt/
```

## GUI
1. Build Cython module:
```
cd gui
python setup.py build_ext --inplace
```
2. Make sure you put the checkpoint.pth in the `checkpoint` folder, checkpoint can be found [here (will be updated shortly)]()
3. Change the `cpk_path` in the `gui_demo.py`
4. Run the GUI
```
python gui_demo.py --category 00000000
```
5. Some basic modeling operations of GUI
```
add voxel - ctrl + left click
delete voxel - shift + left click
rotate - left click + drag
zoom in/out - scroll wheel
```
GUI currently only supports editing voxel from scratch, more input formats will be supported in the future.

### Demo video
Demo video can be found in the `/gui` folder
