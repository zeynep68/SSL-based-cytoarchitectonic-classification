# pip3 install vtk
import h5py
import meshio
import pymeshlab as pm
import siibra
from meshio import write_mesh


def get_space(name='colin27', format='neuroglancer/precompmesh/surface'):
    space = siibra.spaces[name].get_template().fetch(format=format)
    return space['verts'], space['faces']


def reduce_mesh(vertices, faces, scale=10, save_mesh=True, save_path='reduced_mesh.h5'):
    """
        target_length: Sets the target length for the remeshed mesh edges.
    """
    targetlen = pm.AbsoluteValue(scale)
    print(f'Shape vertices before: {vertices.shape}')
    print(f'Shape faces before: {faces.shape}')
    print(30 * '-')
    mesh = pm.Mesh(vertex_matrix=vertices, face_matrix=faces)
    mesh_set = pm.MeshSet()
    mesh_set.add_mesh(mesh)
    mesh_set.meshing_isotropic_explicit_remeshing(targetlen=targetlen)
    output_mesh = mesh_set.mesh(0)

    print(output_mesh.vertex_matrix().shape)
    print(output_mesh.face_matrix().shape)

    if save_mesh:
        meshio.write_mesh(fname="mesh_reduced.ply",
                          vertices=output_mesh.vertex_matrix(),
                          faces=output_mesh.face_matrix())
    if save_path is not None:
        h5_file = h5py.File(save_path, 'w')
        h5_file.create_dataset('vertices', data=output_mesh.vertex_matrix())
        h5_file.create_dataset('faces', data=output_mesh.face_matrix())


if __name__ == '__main__':
    vertices, faces = get_space()
    # write_mesh(fname="mesh.ply", vertices=vertices, faces=faces)
    reduce_mesh(vertices, faces)
