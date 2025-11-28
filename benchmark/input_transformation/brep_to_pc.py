from tqdm import tqdm
import numpy as np
import open3d as o3d
from pathlib import Path
import random

from OCC.Core.BRepGProp import BRepGProp_Face
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepTools import breptools_UVBounds
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop

from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Pnt
from OCC.Core.AIS import AIS_Point
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.Geom import Geom_CartesianPoint

def visualize_pointcloud(shape, points, show_shape=True):
    """Visualize the original shape and sampled points."""
    display, start_display, _, _ = init_display()
    
    # Display the original shape in semi-transparent mode
    if show_shape:
        display.DisplayShape(shape, transparency=0.5)
    
    # Display points
    for x, y, z in points:
        point = gp_Pnt(x, y, z)
        # Wrap gp_Pnt in Geom_CartesianPoint
        geom_point = Geom_CartesianPoint(point)
        ais_point = AIS_Point(geom_point)
        # Set point color to red
        ais_point.SetColor(Quantity_Color(1, 0, 0, Quantity_TOC_RGB))
        display.Context.Display(ais_point, True)
    
    display.FitAll()
    start_display()


def sample_points_on_face(face, num_points):
    """Sample points uniformly on a face based on surface area."""

    # Get UV bounds
    umin, umax, vmin, vmax = breptools_UVBounds(face)
    
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    area = float(props.Mass()) 
    
    points = []
    surf = BRepAdaptor_Surface(face)
    
    # Sample points using rejection sampling
    attempts = 0
    max_attempts = num_points * 10  # Avoid infinite loops
    
    while len(points) < num_points and attempts < max_attempts:
        # Generate random UV coordinates
        u = umin + (umax - umin) * random.random()
        v = vmin + (vmax - vmin) * random.random()
        
        # Generate point
        pnt = surf.Value(u, v)
        points.append([pnt.X(), pnt.Y(), pnt.Z()])
        attempts += 1
    
    return points, area

def step_to_pointcloud(step_path: Path, total_points=1000):
    """Convert STEP file to point cloud with fixed total number of points."""
    reader = STEPControl_Reader()
    if not step_path.exists():
        raise FileNotFoundError(f"STEP file does not exist: {step_path}")
    status = reader.ReadFile(str(step_path))
    if not status:
        raise RuntimeError(f"Failed to read STEP file: {step_path}")
    reader.TransferRoot()
    shape = reader.OneShape()

    # First pass: calculate total area and collect faces
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    faces = []
    areas = []
    total_area = 0
    
    while explorer.More():
        face = explorer.Current()
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        area = float(props.Mass()) 
        
        faces.append(face)
        areas.append(area)
        total_area += area
        explorer.Next()

    # Calculate points per face based on area ratio
    points_per_face = [int(area / total_area * total_points) for area in areas]
    # Ensure we use exactly total_points by adjusting the largest face
    remaining_points = total_points - sum(points_per_face)
    max_face_idx = areas.index(max(areas))
    points_per_face[max_face_idx] += remaining_points

    # Second pass: sample points
    all_points = []
    for face, num_points in zip(faces, points_per_face):
        if num_points > 0:
            face_points, _ = sample_points_on_face(face, num_points)
            all_points.extend(face_points)

    return shape, np.array(all_points)

def save_pointcloud_to_npy(points: np.ndarray, output_path: Path):
    np.save(str(output_path), points)
    # print(f"Saved point cloud to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert STEP files to point cloud .npy files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing STEP files.")
    parser.add_argument("--num_points", type=int, default=1024, help="Number of points to sample per part.")
    parser.add_argument("--only_new", action="store_true", help="Skip files that already have .npy output.")
    args = parser.parse_args()

    step_dir = Path(args.input_dir)
    output_dir = step_dir / "pcs"
    output_dir.mkdir(parents=True, exist_ok=True)

    step_files = list(step_dir.glob("*.stp"))
    filtered_step_files = [f for f in step_files if "unfolded" not in f.stem]
    print(f"Found {len(step_files)} STEP files in {step_dir.resolve()}")
    print(f"Processing {len(filtered_step_files)} STEP files (excluding 'unfolded')")

    for step_file in tqdm(filtered_step_files, desc="Processing STEP files"):
        output_file = output_dir / (step_file.stem + ".npy")
        if args.only_new and output_file.exists():
            continue
        shape, points = step_to_pointcloud(step_file, total_points=args.num_points)
        save_pointcloud_to_npy(points, output_file)