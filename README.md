# BenDFM: A Synthetic Dataset for Manufacturability Assessment in Sheet Metal Bending

<img src="docs/images/bendfm_preview.png" alt="BenDFM Dataset Preview" style="display:block; margin:auto; width:100%; max-width:900px;"/>

<p align="center">
  <span style="font-size:1.5em;"><u><strong>Full dataset release will follow publication.</strong></u></span>
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

## Dataset Characteristics & Generation

Key parameters used to generate parts:

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

The `examples` folder contains the first 5 designs from the training set.

---

### Label Table (`identifier_labels.json`)

| Label Key                | Type    | Description                                                        |
|--------------------------|---------|--------------------------------------------------------------------|
| total_bends              | int     | Number of bends in the part                                        |
| y_tool_collision         | int/bool| 1 if any bend collides with punch/die, else 0                      |
| y_unfolding_collision    | int/bool| 1 if unfolded part self-intersects, else 0                         |
| num_punch_collisions     | int     | Number of punch collisions across all bends                        |
| num_die_collisions       | int     | Number of die collisions across all bends                          |
| sheet_flips              | int     | Number of times the sheet is flipped during manufacturing          |
| total_punch_rotations    | float   | Total punch rotation angle (degrees)                               |
| total_bend_distances     | float   | Total travel distance (mm) for all bends                           |
| part_volume_3d           | float   | Part volume (cm³)                                                  |
| bbox_volume_3d           | float   | 3D bounding box volume (cm³)                                       |
| part_mass_kg             | float   | Mass of the part (kg)                                              |
| bbox_area_unfolded       | float   | Area of unfolded bounding box (cm²)                                |
| sheet_thickness          | float   | Sheet thickness (mm)                                               |
| num_rounded_flanges      | int     | Number of rounded flanges                                          |
| num_slanted_flanges      | int     | Number of slanted flanges                                          |
| min_bend_height          | float   | Minimum bend height (mm)                                           |
| max_bend_height          | float   | Maximum bend height (mm)                                           |
| min_bend_radius          | float   | Minimum bend radius (mm)                                           |
| max_bend_radius          | float   | Maximum bend radius (mm)                                           |
| min_bend_angle           | float   | Minimum bend angle (degrees)                                       |
| max_bend_angle           | float   | Maximum bend angle (degrees)                                       |
| num_bend_reliefs         | int     | Number of bend reliefs                                             |
| num_symmetric_pairs      | int     | Number of symmetric bend pairs                                     |

---

### Sequence Table (`identifier_sequence.json`)

| Field Name         | Type    | Description                                                        |
|--------------------|---------|--------------------------------------------------------------------|
| base_length        | float   | Base plate length (mm)                                             |
| base_width         | float   | Base plate width (mm)                                              |
| part_thickness     | float   | Part thickness (mm)                                                |
| face_name          | str     | Face where bend is applied                                         |
| edge_name          | int/str | Identifier of edge being bent                                       |
| full_width         | float   | Full face width (mm)                                               |
| width_ratio        | float   | Edge width / full width                                             |
| edge_width         | float   | Edge width being bent (mm)                                         |
| bend_angle         | float   | Bend angle (degrees)                                               |
| bend_radius        | float   | Bend radius (mm)                                                   |
| bend_height        | float   | Bend height (mm)                                                   |
| orientation        | str     | 'up' or 'down'                                                    |
| flange_type        | str     | 'rectangular', 'rounded', or 'slanted'                             |
| rect_ratio         | float   | Rectangular flange ratio                                           |
| side               | str     | Side of flange (if applicable)                                     |
| symmetric_to_last  | bool    | True if symmetric to previous bend                                  |
| distance_last_bend | float   | Distance to previous bend (mm)                                     |
| punch_rotation     | float   | Punch rotation (degrees)                                           |
| flip               | bool    | True if sheet flipped for this bend                                 |
| collision_self     | bool    | True if self-collision occurs                                       |
| collision_punch    | bool    | True if punch collision occurs                                      |
| collision_die      | bool    | True if die collision occurs                                        |

---

## Label Categories

**Feasibility Labels:**

- **y_tooling_collision**: Binary flag for any bend colliding with punch/die.
- **y_unfolding_collision**: Binary flag for self-intersections in unfolded geometry.
- **Per-Bend Collisions**: Detailed flags for each bend.

**Complexity Labels:**

- **Configuration-Dependent**: Number of flips, total punch rotations, total bend travel distance.
- **Configuration-Independent**: Part mass, 3D and unfolded bounding box volumes, number of symmetric bend pairs, etc.

---

## Publications

Please cite if you use BenDFM:

```bibtex
@article{ballegeer2025bendfm,
  title={BenDFM: A data-driven framework and synthetic dataset for assessing manufacturability in sheet metal bending},
  author={Ballegeer, Matteo and Benoit, Dries F},
  journal={xxx},
  year={2025}
}
```

## Contact
For questions or collaboration, please contact the authors via the following email: `matteo.ballegeer@ugent.be`

## License
See `LICENSE` for terms of use.