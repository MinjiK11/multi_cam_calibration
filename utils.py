import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import cv2
import os

import plotly.express as px

pio.renderers.default = "browser"

def visualize_point_cloud(points_list, colors_list=None, size=1, cams=None):
    """
    Visualize 3D point cloud with Plotly. (with camera pose)
    
    Args:
        points (np.ndarray): (N, 3) array of 3D points (x, y, z). (list)
        colors (np.ndarray): (N, 3) or (N,) RGB colors in [0,1] or grayscale. (list)
        size (int): Marker size.
        cams: list of Cameras()
    """

    scatters = []
    for idx, points in enumerate(points_list):

        points[:,1] = -points[:,1]  # flip Y
        points[:,2] = -points[:,2]  # flip Z (optional, depends on your convention)

        xmin = -0.04
        xmax = 0.69
        zmin = -0.1
        zmax = 0.45
        ymin = -0.8
        ymax = 0.01

        mask = (
            (points[:,0] >= xmin) & (points[:,0] <= xmax) &
            (points[:,1] >= ymin) & (points[:,1] <= ymax) &
            (points[:,2] >= zmin) & (points[:,2] <= zmax)
        )

        points = points[mask]
        colors = colors_list[idx]

        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        if colors is None:
            marker_color = 'blue'
        else:
            if not isinstance(colors,str) and colors.ndim == 2 and colors.shape[1] == 3:  # RGB
                colors = colors[mask]
                colors = (colors * 255).astype(np.uint8)  # scale if needed
                marker_color = ['rgb({},{},{})'.format(b, g, r) for r, g, b in colors]
            else:  # grayscale or scalar
                marker_color = colors

        scatter = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=size, color=marker_color)
        )

        scatters+=[scatter]
    
    if cams!=None:
        # camera pose
        traces = plot_camera_poses(cams)
        fig = go.Figure(data=scatters + traces)
    else:
        fig = go.Figure(data=scatters)

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
        )
    )

    fig.show()

def visualize_point_cloud_only(cam, points, colors=None, size=2):
    """
    Visualize 3D point cloud with Plotly. (with camera pose)
    
    Args:
        points (np.ndarray): (N, 3) array of 3D points (x, y, z).
        colors (np.ndarray): (N, 3) or (N,) RGB colors in [0,1] or grayscale.
        size (int): Marker size.
    """
    points[:,1] = -points[:,1]  # flip Y
    points[:,2] = -points[:,2]  # flip Z (optional, depends on your convention)

    # workspace to be visualized
    x_min, x_max = -0.05, 0.9
    y_min, y_max = -0.8, 0.06

    mask = (
        (points[:,0] >= x_min) & (points[:,0] <= x_max) &
        (points[:,1] >= y_min) & (points[:,1] <= y_max)
    )

    points = points[mask]
    

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    if colors is None:
        marker_color = 'blue'
    else:
        colors = colors[mask]
        if colors.ndim == 2 and colors.shape[1] == 3:  # RGB
            colors = (colors * 255).astype(np.uint8)  # scale if needed
            marker_color = ['rgb({},{},{})'.format(b, g, r) for r, g, b in colors]
        else:  # grayscale or scalar
            marker_color = colors

    scatter = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=size, color=marker_color)
    )

    cam = [cam]
    traces = plot_camera_poses(cam)

    fig = go.Figure(data=[scatter]+traces)

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
        )
    )

    fig.show()

# # # # camera pose visualization # # # #
def plot_camera_pose(cam, scale=0.1, name="cam"):
    """Return plotly traces for a camera coordinate frame"""

    R = cam.getExtrinsic()[:3,:3]
    t = cam.getExtrinsic()[:3,3]

    # camera position w.r.t. world coordinate
    R_c2w = R.T
    t_c2w = -R.T @ t

    R_flip = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
    ])

    t_c2w = R_flip @ t_c2w
    R_c2w = R_flip @ R_c2w

    origin = t_c2w.reshape(3)

    # Axes in world coordinates
    x_axis = origin + R_c2w[:,0] * scale
    y_axis = origin + R_c2w[:,1] * scale
    z_axis = origin + R_c2w[:,2] * scale

    traces = []

    # X-axis (red)
    traces.append(go.Scatter3d(x=[origin[0], x_axis[0]],
                               y=[origin[1], x_axis[1]],
                               z=[origin[2], x_axis[2]],
                               mode='lines', line=dict(color='red', width=5), name=f"{name}-x"))
    # Y-axis (green)
    traces.append(go.Scatter3d(x=[origin[0], y_axis[0]],
                               y=[origin[1], y_axis[1]],
                               z=[origin[2], y_axis[2]],
                               mode='lines', line=dict(color='green', width=5), name=f"{name}-y"))
    # Z-axis (blue)
    traces.append(go.Scatter3d(x=[origin[0], z_axis[0]],
                               y=[origin[1], z_axis[1]],
                               z=[origin[2], z_axis[2]],
                               mode='lines', line=dict(color='blue', width=5), name=f"{name}-z"))

    # Label
    traces.append(go.Scatter3d(x=[origin[0]], y=[origin[1]], z=[origin[2]],
                               mode='text', text=[name], textposition="top center"))

    return traces

def plot_camera_poses(cams):
    traces = []

    for cam in cams:
        cname = 'cam'+str(cam.cam_idx)
        traces+=plot_camera_pose(cam,name=cname)

    return traces

def viz_world_axis(cam):
    ## visualize world coordinate ###
    ext = cam.getExtrinsic()
    axis_len = 0.05

    points_world = np.array([
    [0, 0, 0],          # origin
    [axis_len, 0, 0],   # X
    [0, axis_len, 0],   # Y
    [0, 0, axis_len]    # Z
    ])

    R_w2c = ext[:3,:3]
    t_w2c = ext[:3,3]

    points_cam = (R_w2c @ points_world.T + t_w2c.reshape(3,1)).T

    imgpts = []
    for pt in points_cam:
        imgx, imgy = cam[0].cam2img(pt[0]/pt[-1], pt[1]/pt[-1])
        imgpts.append(np.array([imgx, imgy]))

    origin = tuple(imgpts[0].ravel().astype(int))
    x_axis = tuple(imgpts[1].ravel().astype(int))
    y_axis = tuple(imgpts[2].ravel().astype(int))
    z_axis = tuple(imgpts[3].ravel().astype(int))
    
    img = cv2.imread(os.path.join('./color',cam.cam_idx+'.png'))

    cv2.line(img, origin, x_axis, (0,0,255), 3)  # X: red
    cv2.line(img, origin, y_axis, (0,255,0), 3)  # Y: green
    cv2.line(img, origin, z_axis, (255,0,0), 3)  # Z: blue

    cv2.imwrite('world_axis.png', img)