The dataset will be added to this repository upon publication. 
For now, example files from the dataset can be found under the ``examples`` folder.

# BenDFM dataset

The BenDFM dataset contains geometrically varied sheet metal bending designs derived from parametric CAD modelling in PythonOCC. The dataset provides a valuable source for Design For Manufacturing (DFM) using deep learning, and contains a rich set of labels and representations (3D vs unfolded sequence information) that allow it to be used for other related tasks.
Benchmarks show that current state-of-the-art techniques are not able to solve the problems present in the dataset, encouraging research on better descriptors of global structural information in CAD designs.
This repo contains the BenDFM and BenDFM-U datasets, as well as detailed dataset characteristics and instructions on how to interpret its files.

# BenDFM dataset characteristics

The BenDFM dataset consists of 14,000 sheet metal bending designs in STEP format. Each design contains 2 to 8 bends, and the dataset has 2000 designs for each number of bends. 
Each design is identified by its ``identifier`` of format ``design_x_bends_y``, denoting the ID ``xx`` and number of bends ``y``. For each design, the dataset contains 5 files:
- ``design_x_bends_y.stp``: the 3D CAD file in STEP format.
- ``design_x_bends_y_unfolded.stp``: the CAD file of the unfolded representation in STEP format.
- ``design_x_bends_y.json``: a JSON containing the parameters of the base plate and each of the bends.
- ``design_x_bends_y_labels.json``: a JSON containing the relevant labels used in the BenDFM paper along with some other ones.
- ``design_x_bends_y_hashes.json``: a JSON containing the coordinate hashes to link each bend to a face / edge.



# Publications

Please cite the paper below if you use the BenDFM dataset in your research.
````
@article{ballegeerXXXXbendfm,
  title={XXX},
  author={Ballegeer, Matteo and Benoit, Dries F},
  journal={XXX},
  year={XXX}
}
````
