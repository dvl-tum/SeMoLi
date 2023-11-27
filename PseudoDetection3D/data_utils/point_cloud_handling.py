import numpy as np

def compute_interior_points_mask(points_xyz, cuboid_vertices):
    r"""Compute the interior points mask for the cuboid.
    Reference: https://math.stackexchange.com/questions/1472049/check-if-a-point-is-inside-a-rectangular-shaped-area-3d
            5------4
            |\\    |\\
            | \\   | \\
            6--\\--7  \\
            \\  \\  \\ \\
        l    \\  1-------0    h
         e    \\ ||   \\ ||   e
          n    \\||    \\||   i
           g    \\2------3    g
            t      width.     h
             h.               t.
    Args:
        points_xyz: (N,3) Array representing a point cloud in Cartesian coordinates (x,y,z).
        cuboid_vertices: (8,3) Array representing 3D cuboid vertices, ordered as shown above. 
   
    Returns:
        (N,) An array of boolean flags indicating whether the points are interior to the cuboid.
    """
    # Get three corners of the cuboid vertices.
    vertices = np.stack((cuboid_vertices[6], cuboid_vertices[3], cuboid_vertices[1]))  # (3,3)

    # Choose reference vertex.
    # vertices and choice of ref_vertex are coupled.
    ref_vertex = cuboid_vertices[2]  # (3,)

    # Compute orthogonal edges of the cuboid.
    uvw = ref_vertex - vertices  # (3,3)

    # Compute signed values which are proportional to the distance from the vector.
    sim_uvw_points = points_xyz @ uvw.transpose()  # (N,3)
    sim_uvw_ref = uvw @ ref_vertex  # (3,)

    # Only care about the diagonal.
    sim_uvw_vertices = np.diag(uvw @ vertices.transpose())  # (3,)

    # Check 6 conditions (2 for each of the 3 orthogonal directions).
    # Refer to the linked reference for additional information.
    constraint_a = np.logical_and(sim_uvw_ref <= sim_uvw_points, sim_uvw_points <= sim_uvw_vertices)
    constraint_b = np.logical_and(sim_uvw_ref >= sim_uvw_points, sim_uvw_points >= sim_uvw_vertices)
    is_interior = np.logical_or(constraint_a, constraint_b).all(axis=1)
    return is_interior