from camera import *
from utils import *

def make_pcd(cam):

    h, w, _ = cam.objimg.shape

    depth = cam.objdepth
    
    mask = (depth < 1.5)
    u, v = np.meshgrid(np.arange(w),np.arange(h))
    u = u[mask]
    v = v[mask]
    z = depth[mask]

    x = (u-cam.cx) * z / cam.fx 
    y = (v-cam.cy) * z / cam.fy 

    points = np.dstack((x,y,z)).squeeze()
    colors = cam.objimg[mask]
    colors = (colors-np.min(colors))/(np.max(colors)-np.min(colors))

    # idx = np.random.choice(points.shape[0], int(points.shape[0]/10), replace=False)
    # points_sampled = points[idx]
    # colors_sampled = colors[idx]

    voxel_index = (points/ 0.01).astype(np.int32)
    _, unique_indices = np.unique(voxel_index, axis=0, return_index=True)
    points_sampled = points[unique_indices]
    colors_sampled = colors[unique_indices]

    # points_viz = cam.cam2world(points_sampled)
    # visualize_point_cloud([points_viz],[colors_sampled],cams=[cam])

    # merge pcd in ref. cam coordinate
    points_ref = merge_pcd(points_sampled,cam)

    # # # downsample point cloud # # # 
    # voxel_index = (points_ref / 0.008).astype(np.int32)
    # _, unique_indices = np.unique(voxel_index, axis=0, return_index=True)
    # points_ref = points_ref[unique_indices]
    # colors_sampled = colors_sampled[unique_indices]

    merge = np.concatenate((points_ref,colors_sampled),axis=1)

    cam.pcd = merge

def merge_pcd(points,cam):
    tar2ref = cam.getTar2Ref()
    tar2ref_R = tar2ref[:3,:3]
    tar2ref_t = tar2ref[:3,3]
    tar2ref_t = tar2ref_t.reshape(-1,1)

    # tar2ref_R = np.eye(3)
    # tar2ref_t = np.array([0,0,0]).reshape(-1,1)

    points_tar2ref = (tar2ref_R @ points.T + tar2ref_t).T

    return points_tar2ref