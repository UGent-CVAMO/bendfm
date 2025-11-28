"""Convert STEP files to UV-grid DGL graphs for UV-Net."""

import json
import random
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import dgl
import networkx as nx
from tqdm import tqdm

from OCC.Core.BRep import BRep_Tool
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop, brepgprop_LinearProperties
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Line, GeomAbs_Circle
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.gp import gp_Vec, gp_Pnt

from occwl.graph import face_adjacency
from occwl.compound import Compound
from occwl.uvgrid import ugrid, uvgrid

# Set numpy and torch print options
np.set_printoptions(precision=3)
torch.set_printoptions(precision=3, sci_mode=False)
torch.set_float32_matmul_precision('medium')


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds(42)


# ── Face features ───────────────────────────────────────────────────────

def plane_feature(face):
    return 1.0 if BRepAdaptor_Surface(face).GetType() == GeomAbs_Plane else 0.0

def cylinder_feature(face):
    return 1.0 if BRepAdaptor_Surface(face).GetType() == GeomAbs_Cylinder else 0.0

def area_feature(face):
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    return float(props.Mass())

def normal_vector_feature(face):
    surf = BRepAdaptor_Surface(face)
    u_mid = (surf.FirstUParameter() + surf.LastUParameter()) / 2.0
    v_mid = (surf.FirstVParameter() + surf.LastVParameter()) / 2.0
    prop = GeomLProp_SLProps(surf.Surface().Surface(), u_mid, v_mid, 1, 1e-6)
    if not prop.IsNormalDefined():
        return (0.0, 0.0, 0.0)
    normal = prop.Normal()
    return normal.X(), normal.Y(), normal.Z()

def cylinder_angular_extent(face):
    surf = BRepAdaptor_Surface(face)
    if surf.GetType() != GeomAbs_Cylinder:
        return 0.0
    extent = surf.LastUParameter() - surf.FirstUParameter()
    return round(extent * 180 / np.pi, 2)


# ── Edge features ───────────────────────────────────────────────────────

def circular_edge_feature(edge):
    return 1.0 if BRepAdaptor_Curve(edge).GetType() == GeomAbs_Circle else 0.0

def closed_edge_feature(edge):
    return 1.0 if BRep_Tool().IsClosed(edge) else 0.0

def straight_edge_feature(edge):
    return 1.0 if BRepAdaptor_Curve(edge).GetType() == GeomAbs_Line else 0.0

def edge_length(edge):
    props = GProp_GProps()
    brepgprop_LinearProperties(edge, props)
    return float(props.Mass())

def edge_direction_vector(edge):
    curve = BRepAdaptor_Curve(edge)
    mid_param = 0.5 * (curve.FirstParameter() + curve.LastParameter())
    vec = gp_Vec()
    curve.D1(mid_param, gp_Pnt(), vec)
    if vec.Magnitude() == 0:
        return (0.0, 0.0, 0.0)
    vec.Normalize()
    return vec.X(), vec.Y(), vec.Z()


# ── Graph construction ──────────────────────────────────────────────────

def build_graph(solid, curv_num_u_samples, surf_num_u_samples, surf_num_v_samples):
    """Build a DGL graph from a B-Rep solid with UV-grid and geometric features."""
    graph = face_adjacency(solid)
    solid = solid.topods_shape()

    graph_face_points = []
    graph_face_feat = []

    for face_idx in graph.nodes:
        face = graph.nodes[face_idx]["face"]

        points = uvgrid(face, method="point", num_u=surf_num_u_samples, num_v=surf_num_v_samples)
        normals = uvgrid(face, method="normal", num_u=surf_num_u_samples, num_v=surf_num_v_samples)
        visibility = uvgrid(face, method="visibility_status", num_u=surf_num_u_samples, num_v=surf_num_v_samples)
        mask = np.logical_or(visibility == 0, visibility == 2)
        face_feat = np.concatenate((points, normals, mask), axis=-1)

        graph_face_points.append(face_feat)
        plane = plane_feature(face.topods_shape())
        cylinder = cylinder_feature(face.topods_shape())
        angle = cylinder_angular_extent(face.topods_shape())
        area = area_feature(face.topods_shape())
        graph_face_feat.append(np.array([plane, cylinder, area, angle]))

    graph_edge_points = []
    graph_edge_feat = []

    for edge_idx in graph.edges:
        edge = graph.edges[edge_idx]["edge"]

        points = ugrid(edge, method="point", num_u=curv_num_u_samples)
        tangents = ugrid(edge, method="tangent", num_u=curv_num_u_samples)
        edge_feat = np.concatenate((points, tangents), axis=-1)

        edge_shape = edge.topods_shape()
        circular = circular_edge_feature(edge_shape)
        closed = closed_edge_feature(edge_shape)
        straight = straight_edge_feature(edge_shape)
        length = edge_length(edge_shape)
        edge_dir = edge_direction_vector(edge_shape)
        graph_edge_points.append(edge_feat)
        graph_edge_feat.append(np.array([circular, closed, straight, length, *edge_dir]))

    src = [e[0] for e in graph.edges]
    dst = [e[1] for e in graph.edges]
    dgl_graph = dgl.graph((src, dst), num_nodes=len(graph.nodes))
    dgl_graph.ndata["x"] = torch.from_numpy(np.asarray(graph_face_points))
    dgl_graph.ndata["face_feat"] = torch.from_numpy(np.asarray(graph_face_feat))
    dgl_graph.edata["x"] = torch.from_numpy(np.asarray(graph_edge_points))
    dgl_graph.edata["edge_feat"] = torch.from_numpy(np.asarray(graph_edge_feat))

    return dgl_graph


# ── File processing ─────────────────────────────────────────────────────

def process_one_file(fn, curv_num_u_samples, surf_num_u_samples, surf_num_v_samples, output_dir):
    """Convert a single STEP file to a DGL graph and save it."""
    fn_stem = fn.stem
    output_path = pathlib.Path(output_dir)

    solid = list(Compound.load_from_step(fn).solids())[0]
    graph = build_graph(solid, curv_num_u_samples, surf_num_u_samples, surf_num_v_samples)
    dgl.data.utils.save_graphs(str(output_path / (fn_stem + ".bin")), [graph])
    return graph


def process_files(input_dir, output_dir, curv_num_u_samples, surf_num_u_samples, surf_num_v_samples, num_processes=4, skip_existing=True):
    """Convert all STEP files in a directory to DGL graphs in parallel."""
    input_path = pathlib.Path(input_dir)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    step_files = [f for f in input_path.glob("*.st*p")]
    if skip_existing:
        step_files = [fn for fn in step_files if not (output_path / (fn.stem + ".bin")).exists()]

    if not step_files:
        print("No new files to process.")
        return [], []

    results, error_files = [], []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {
            executor.submit(process_one_file, f, curv_num_u_samples, surf_num_u_samples, surf_num_v_samples, output_path): f
            for f in step_files
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            f = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing {f.name}: {e}")
                error_files.append(f.name)

    if error_files:
        print(f"Files with errors: {error_files}")
    return results, error_files


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert STEP files to UV-grid DGL graphs.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing STEP files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save .bin graph files.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of UV samples (curve U, surface U & V).")
    parser.add_argument("--num_processes", type=int, default=4, help="Number of parallel workers.")
    args = parser.parse_args()

    results, error_files = process_files(
        args.input_dir, args.output_dir,
        curv_num_u_samples=args.num_samples,
        surf_num_u_samples=args.num_samples,
        surf_num_v_samples=args.num_samples,
        num_processes=args.num_processes,
        skip_existing=True,
    )
    print(f"Processed {len(results)} files. Errors: {len(error_files)}")
