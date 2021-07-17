import numpy as np
import torch
import torch
import pytorch3d
import trimesh
from pytorch3d.ops import cubify
from dataset import PartNetDataset
from torch.utils.data import DataLoader

def reflective_symmetry(samples, visualize=False, name="rs.png"):
    ### code taken from: https://github.com/mikedh/trimesh/issues/947
    # get a bunch of unit vectors on a hemisphere
    hemisphere = trimesh.util.spherical_to_vector((
        trimesh.util.grid_linspace([[np.pi/2.01, np.pi/2.01], [np.pi/1.99, np.pi/1.99]], 100))) # narrow the space for 
                                                                                                # vertical reflective symmetries

    # find the sum number of points on each side of the plane
    # if there are an *equal* number of points on both sides it is
    # likely a plane of reflection although this is pretty crude
    u = np.abs([np.sign(np.dot(samples, v)).sum() for v in hemisphere])

    # show the best candidate vector
    if visualize:
        viz = trimesh.Scene(
            [trimesh.load_path([[0, 0, 0], hemisphere[u.argmin()]]),
            trimesh.PointCloud(samples)]).save_image()
        with open(name, "wb") as im:
            im.write(viz)
    return hemisphere[u.argmin()]

def find_part_reflective_vector(model_name, part_id, num_visualize=0):

    dataset = PartNetDataset("./dataset/train_test_split/shuffled_train_file_list.json",
                                    "./processed/train/", model_name, part_id)
    dataloader = DataLoader(dataset, batch_size=1)
    reflective_vector_list = []
    for i,data in enumerate(dataloader):
        voxels = data.view(-1, 64, 64, 64)
        p3d_mesh = cubify(voxels, 1e-5)
        vertices = p3d_mesh.verts_list()[0].numpy()
        faces = p3d_mesh.faces_list()[0].numpy()
        vertex_normals = p3d_mesh.verts_normals_list()[0].numpy()
        t_mesh = trimesh.Trimesh(vertices, faces, vertex_normals=vertex_normals)
        try:
            samples = t_mesh.sample(5000)
            visualize = False
            if num_visualize > 0 and i <= num_visualize:
                visualize = True
            reflective_vector_list.append(reflective_symmetry(samples, visualize=visualize, name="./symm_images/{}_{}.png".format(part_id,i)))
        except Exception as e:
            reflective_vector_list.append(np.array([0,0,0]))
    return reflective_vector_list

if __name__ == "__main__":
    for part_id in range(1,5):
        revlective_vectors = find_part_reflective_vector("03001627", part_id, num_visualize=10)

        np.save("./symmetry{}.npy".format(part_id),np.array(revlective_vectors))



