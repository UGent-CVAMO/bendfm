## File Structure

Each part (identified by `identifier` 0–19999) contains:

- `identifier.stp`: 3D CAD model of the bent part.
- `identifier_unfolded.stp`: 2D unfolded CAD model.
- `identifier_sequence.json`: Base plate parameters and ordered bend sequence.
- `identifier_labels.json`: Detailed manufacturability and descriptive labels.

---
### Label descriptions (`identifier_labels.json`)

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

### Bend-per-bend sequence descriptions (`identifier_sequence.json`)

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
