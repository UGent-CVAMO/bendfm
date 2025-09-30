# BenDFM: A Synthetic Dataset for Manufacturability Assessment in Sheet Metal Bending

<img src="docs/images/bendfm_preview.png" alt="BenDFM Dataset Preview" style="display:block; margin:auto; width:100%; max-width:900px;"/>

<p align="center">
  <span style="font-size:1.5em;"><u><strong>Full dataset will be released upon publication.</strong></u></span>
</p>

This repository hosts **BenDFM**, a large-scale synthetic dataset of sheet metal parts designed for data-driven Design for Manufacturing (DFM) research. 

The dataset is introduced in the paper: *BenDFM: A data-driven framework and synthetic dataset for manufacturability assessment in sheet metal bending* and provides:

- Geometrically diverse 3D sheet metal bending parts and their unfolded representations.
- Rich part-level metadata.
- Comprehensive manufacturability labels structured according to a novel taxonomy, spanning both geometric and configurational feasibility and complexity.

BenDFM enables a wide range of tasks, including binary feasibility classification, regression of manufacturing complexity, and analysis of process constraints.

The dataset is split into two subsets: **BenDFM** and **BenDFM-U**, each balanced to support specific manufacturability tasks.

---

## Dataset Overview

- **Total Parts**: 20,000 unique 3D bent part geometries in STEP format.
- **Subset 1 — BenDFM**: 14,000 parts with 2–8 bends, tooling collisions balanced at 50%, all parts guaranteed free of unfolding overlaps.
- **Subset 2 — BenDFM-U**: 6,000 parts with 7–10 bends, balanced 50/50 for unfolding overlaps.

---

## Dataset generation parameters

- **Base Sheet Dimensions**: 150–300 mm (length & width, uniform sampling)
- **Sheet Thickness**: 2–6 mm
- **Bend Angles**: {45°, 60°, 90°, 120°, 135°}, biased toward 90°
- **Flange Heights**: 75 mm to maximum sheet dimension
- **Bend Radii**: 1.0–1.5 × sheet thickness
- **Realism Features**: Automatic bend reliefs, flange geometry variants, symmetry bias

---

## File Structure

Each part (identified by `identifier` 0–19999) contains:

- `identifier.stp`: 3D CAD model of the bent part.
- `identifier_unfolded.stp`: 2D unfolded CAD model.
- `identifier_sequence.json`: Base plate parameters and ordered bend sequence.
- `identifier_labels.json`: Detailed manufacturability and descriptive labels.

You can find sample files in the [`example_files`](example_files/) folder, which contains the first 5 designs from the training set, with a detailed description of the file types and their contents given in [`file_types`](docs/file_types.md).
**The full dataset will be released upon publication.**

## Dataset characteristics

For an overview of the most important dataset characteristics, we refer to [`dataset_characteristics`](docs/dataset_characteristics.md)
## Publications

Please cite the following publication if you use BenDFM:

```bibtex
@article{ballegeer2025bendfm,
  title={BenDFM: A data-driven framework and synthetic dataset for manufacturability assessment in sheet metal bending},
  author={Ballegeer, Matteo and Benoit, Dries F},
  journal={xxx},
  year={xxx}
}
```

## Contact
For questions or collaboration, please contact the authors via email: [matteo.ballegeer@ugent.be](mailto:matteo.ballegeer@ugent.be)

See [`LICENSE`](LICENSE) for terms of use.