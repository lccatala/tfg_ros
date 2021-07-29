import open3d as o3d
import trimesh
import numpy as np
import argparse
import pprint
from typing import Optional
from typing import Sequence

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='input filename')
    args = parser.parse_args(argv)
    pprint.pprint(vars(args))
    pcd = o3d.io.read_point_cloud(args.filename)
    pcd.estimate_normals()

    print('estimating distances')
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    alpha = 1.5
    radius = alpha * avg_dist
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))

    print('creating triangular mesh')
    tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), vertex_normals=np.asarray(mesh.vertex_normals))

    print(trimesh.convex.is_convex(tri_mesh))
    print('saving')
    tri_mesh.export(file_obj='meshes/output' + args.filename + '.obj')
if __name__ == '__main__':
    exit(main())

