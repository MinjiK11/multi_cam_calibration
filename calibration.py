import yaml
import os
import cv2
import numpy as np
import glob
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import open3d as o3d
import argparse

from merge_pcd import *
from camera import *
from utils import *

def parse_args():
    ap = argparse.ArgumentParser(description="Eye-to-hand capture with ChArUco.")
    ap.add_argument("-c", "--config", type=str, required=True, help="Path to config.yaml")
    return ap.parse_args()

def load_config(path: str) -> dict:
    """
    Load YAML config and apply defaults for optional sections.

    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    return cfg

class CalibBoard():

    def __init__(self, config):

        # define variable value using config file
        self.number_x_square = config["number_x_square"]
        self.number_y_square = config["number_y_square"]
        self.length_square = config["length_square"]
        self.length_marker = config["length_marker"]
        
        self.number_camera = config["number_camera"]
        self.cam_intrinsic_path = config["cam_intrinsic_path"]
        self.fix_intrinsic = config["fix_intrinsic"]

        self.data_path = config["data_path"]
        self.depth_path = config["depth_path"]
        self.obj_img_path = config["obj_img_path"]
        self.obj_depth_path = config["obj_depth_path"]
        
        self.ransac_threshold = config["ransac_threshold"]
        self.number_iterations = config["number_iterations"]

        self.save_path = config["save_path"]
        self.save_detection = config["save_detection"]
        self.save_reprojection = config["save_reprojection"]

        self.cams = []
        self.ref_cam = config["ref_cam"]

        # initialize camera information
        for idx in range(self.number_camera):
            self.cams.append(Camera(idx))
            fname = str(idx)+'.png'
            dname = str(idx)+'.npy'
            self.cams[idx].img = cv2.imread(os.path.join(self.data_path,fname))
            self.cams[idx].depth = np.load(os.path.join(self.depth_path,dname),allow_pickle=True)
            self.cams[idx].objimg = cv2.imread(os.path.join(self.obj_img_path,fname))
            self.cams[idx].objdepth = np.load(os.path.join(self.obj_depth_path,dname),allow_pickle=True)

        # set intrinsic parameter
        if self.fix_intrinsic:
            if not os.path.exists(self.cam_intrinsic_path):
                raise FileNotFoundError(f"File for camera intrinsic '{self.cam_intrinsic_path}' not found!")

            try:
                with open(self.cam_intrinsic_path, "r") as f:
                    intrinsics = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(f"Error parsing YAML file: {e}")
                intrinsics = None

            for idx in range(self.number_camera):
                self.cams[idx].setIntrinsic(intrinsics[idx])

        # check if the save dir exist and create it if it does not
        os.makedirs(self.save_path, exist_ok=True)
        print(f"Save directory '{self.save_path}' is ready.")

    def detect_charuco_corners(self, cam):
        """
        Detect corner of Charuco board

        cam: Camera()

        """
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

        charuco_board = cv2.aruco.CharucoBoard_create(
            self.number_x_square, self.number_y_square, self.length_square, self.length_marker, aruco_dict
        )

        image = cam.img

        if image is None:
            print(f"Error: Unable to load image")
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Aruco marker detection
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
        
        if ids is not None:
            # Corner detection via interpolation
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, charuco_board)
            
            if len(charuco_corners) > 0:
                for c in charuco_corners:
                    cv2.cornerSubPix(
                        gray,                    # grayscale image
                        c,                       # corner coordinates
                        winSize=(5,5),           # search window size
                        zeroZone=(-1,-1),        # region to exclude
                        criteria=criteria
                    )

            cam.corners = charuco_corners.squeeze() # detected 2D image corner point
            cam.objp = charuco_board.chessboardCorners[charuco_ids.flatten()] # define object point (3D world)

            if self.save_detection:
                # visualize detected corner as red dot
                if charuco_corners is not None:
                    for corner in charuco_corners:
                        x, y = int(corner[0][0]), int(corner[0][1])  
                        cv2.circle(image, (x, y), 3, (0, 0, 255), -1) 

                # save visualization result
                save_dir = os.path.join(self.save_path,'detection')
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_dir,str(cam.cam_idx)+'_corner.png'), image)
                print(f"Visualization of detected corners saved to {save_dir}")
            else:
                print("No ArUco markers detected in the image.")
    
    # # # extrinsic calibration for single camera # # # 
    def estimatePose(self, cam):
        """
        Estimate camera extrinsic parameters (w2c R,t)

        """
        mtx = cam.getIntrinsic()

        # calculate R, t using PnP & RANSAC
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(cam.objp, cam.corners, mtx, cam.dist)

        # # # reprojection error before R, t refinement # # #
        w2c = np.concatenate([rvecs, tvecs])
        w2c = w2c.squeeze()
        reprojection_error = self.calculate_reprojection_error(w2c,cam.corners,cam.objp,cam)
        rmse = np.sqrt(np.mean(reprojection_error**2))
        print("rmse before refinement: ", rmse)

        # refine R, t
        rvec_refine, tvec_refine = self.refine_param(cam,rvecs,tvecs)

        rvec_refine = np.expand_dims(rvec_refine,axis=1)
        tvec_refine = np.expand_dims(tvec_refine,axis=1)

        cam.setExtrinsic(rvec_refine, tvec_refine)
        print("Extrinsic matrix:", cam.getExtrinsic())

        # project 3D points to image plane
        if self.save_detection:
            image = cam.img.copy()
            for pt in cam.objp:
                imgx, imgy = cam.world2img(pt)
                cv2.circle(image, (int(imgx), int(imgy)), 5, (255, 0, 0), 2, cv2.LINE_AA)

            save_dir = os.path.join(self.save_path,'detection')
            cv2.imwrite(os.path.join(save_dir,str(cam.cam_idx)+'_reprojection.png'), image)
            print('Visualization of reprojection result is saved!')

    def calculate_reprojection_error(self,param,imgpts,objpts,cam):
        '''
        calculate reprojection error (world point to img plane)

        param: 6D vector [rvec, tvec] (w2c)
        imgpts: detected corner point (img plane)
        objpts: 3D world point
        cam: Camera() which its R, t to be refined

        reprojection_error: array containing residual error

        '''
        reprojection_error=[]

        rvec = param[:3]
        tvec = param[3:]
        
        rmat, _ = cv2.Rodrigues(rvec)
        skew = 0 # TODO

        for i, objpt in enumerate(objpts):
            # project world point to img plane

            skew = 0 # TODO

            # world to cam
            w2c = rmat @ objpt + tvec
            camx = w2c[0]/w2c[2]
            camy = w2c[1]/w2c[2]

            # cam to img
            imgx, imgy = cam.cam2img(camx, camy)

            # calculate reprojection error
            residual=[imgpts[i][0]-imgx, imgpts[i][1]-imgy]

            reprojection_error+=residual

        # rmse
        residuals = np.array(reprojection_error)
        rmse = np.sqrt(np.mean(residuals**2))
        print("rmse: ", rmse)

        return np.array(reprojection_error).flatten()
    
    def refine_param(self,cam,rvecs=None,tvecs=None,isPair=False):
        '''
        refine extrinsic parameter via nonlinear optimization

        cam: camera which their parameters are to be optimized
        rvecs: calculated rotation vector (to be optimzed)
        tvec: calculated translation vector (to be optimized)

        rvec_refine, tvec_refine: refined rvec, tvec
        '''
        param = np.concatenate([rvecs, tvecs])
        param = param.squeeze()

        objpts = cam.objp
        imgpts = cam.corners

        new_param = least_squares(self.calculate_reprojection_error,param,method='lm',max_nfev=20000,ftol=1e-4,xtol=1e-4,gtol=1e-4,args=(imgpts,objpts,cam),verbose=1)

        rvec_refine = new_param['x'][:3]
        tvec_refine = new_param['x'][3:]

        return rvec_refine,tvec_refine


    # # # multi-camera calibration # # #
    def computeCameraPairPose(self,ref,tar):
        '''
        compute relative pose between reference cam and target cam (tar2ref)

        ref: reference Camera()
        tar: target Camera()
        '''
        ref_R = ref.getExtrinsic()[:3,:3]
        ref_t = ref.tvec
        tar_R = tar.getExtrinsic()[:3,:3]
        tar_t = tar.tvec

        t2r_R = ref_R @ tar_R.T
        t2r_t = ref_t - t2r_R @ tar_t

        t2r_rvec, _ = cv2.Rodrigues(t2r_R)

        param = np.concatenate([t2r_rvec, t2r_t])
        param = param.squeeze()

        new_param = least_squares(self.reprojectionErrorPairPose,param,method='lm',ftol=1e-04,xtol=1e-04,gtol=1e-04,args=(tar.objp,ref,tar),verbose=1)

        rvec_refine = new_param['x'][:3] # refined tar2ref rvec
        tvec_refine = new_param['x'][3:] # refined tar2ref tvec

        tar.setTar2Ref(rvec_refine,tvec_refine)

        if self.save_detection:
            image = ref.img.copy()
            for pt in tar.t2rpt:
                imgx, imgy = ref.cam2img(pt[0],pt[1])
                cv2.circle(image, (int(imgx), int(imgy)), 5, (0, 255, 0), 2, cv2.LINE_AA)

            save_dir = os.path.join(self.save_path,'detection')
            cv2.imwrite(os.path.join(save_dir,str(tar.cam_idx)+'_pairPose.png'), image)
            print('Visualization of reprojection result is saved!')

    def reprojectionErrorPairPose(self,param,tar_objpts,ref,tar):
        reprojection_error=[]

        rvec = param[:3] # tar2ref rvec
        tvec = param[3:] # tar2ref tvec

        rmat, _ = cv2.Rodrigues(rvec) 
        skew = 0 # TODO

        t2rpt = []

        for i, objpt in enumerate(tar_objpts):
            refCam_proj = self.tar2ref(rmat, tvec, objpt, tar) # project cam point in target camera to ref camera

            # calculate reprojection error
            refCam = ref.world2cam(objpt)
            residual=[refCam[0]-refCam_proj[0],refCam[1]-refCam_proj[1],refCam[2]-refCam_proj[2]]

            reprojection_error+=residual

            # for visualization
            refCam_x_proj = refCam_proj[0]/refCam_proj[2]
            refCam_y_proj = refCam_proj[1]/refCam_proj[2]
            t2rpt.append(np.array([refCam_x_proj,refCam_y_proj]))

        if self.save_detection:
            tar.t2rpt = np.array(t2rpt)

        # rmse
        residuals = np.array(reprojection_error)
        rmse = np.sqrt(np.mean(residuals**2))
        print("rmse (relative pose): ", rmse)

        return np.array(reprojection_error).flatten()
  
    def tar2ref(self, rmat, tvec, objpt, tar):
        '''
        convert point in target camera coordinate to reference camera coordinate

        rmat: tar2ref rotation matrix (to be optimized)
        tvec: tar2ref translation vector (to be optimized)
        objpt: 3D world point
        tar: target Camera()
        '''
        # convert world point to target camera coordinate
        tar_R = tar.getExtrinsic()[:3,:3]
        tar_t = tar.getExtrinsic()[:3,3]

        tarCam = tar_R @ objpt + tar_t

        # convert to point in ref camera coordinate
        refCam = rmat @ tarCam + tvec

        return refCam


    def calibrateSingle(self, cam):
        self.detect_charuco_corners(cam)
        self.estimatePose(cam)

    def calibratePair(self,ref,tar):
        self.computeCameraPairPose(ref,tar)


    def save_pose(self,cams):
        poses = dict()
        pair_poses = dict()

        for cam in cams:
            T_w2c = cam.getExtrinsic()
            poses[str(cam.cam_idx)] = T_w2c

        for cam in cams:
            T_pair = cam.getTar2Ref()
            pair_poses[str(cam.cam_idx)] = T_pair

        np.savez(os.path.join(self.save_path,'w2c_ext.npz'), **poses)
        np.savez(os.path.join(self.save_path,'t2r_ext.npz'), **pair_poses)

if __name__=='__main__':

    args = parse_args()
    cfg = load_config(args.config)

    calib = CalibBoard(cfg)

    # # single camera calibration # #
    for idx in range(calib.number_camera):
        calib.calibrateSingle(calib.cams[idx])

    # # multi camera calibration # #
    target_cams = [x for i, x in enumerate(calib.cams) if i!=calib.ref_cam]

    ref_cam = calib.cams[calib.ref_cam]

    for tar_cam in target_cams:
        calib.calibratePair(ref_cam,tar_cam)

    # save world2refCam, target2ref transformation matrix
    calib.save_pose(calib.cams)

    # # visualization # #
    # merge pcd from each camera (in ref. cam coordinate)
    pcds = np.zeros((1,6))
    for cam in calib.cams:
        make_pcd(cam)

        pcds = np.vstack((pcds,cam.pcd))

    pcds = np.array(pcds)
    points = pcds[:,:3]
    colors = pcds[:,3:]
    
    # downsample points
    voxel_index = (points / 0.003).astype(np.int32)
    _, unique_indices = np.unique(voxel_index, axis=0, return_index=True)
    points = points[unique_indices]
    colors = colors[unique_indices]

    # transform point in ref. cam coordiante to world coordinate
    pcd_world = ref_cam.cam2world(points)

    xmin = -0.05
    xmax = 0.65
    zmin = -0.45
    zmax = 0.1
    ymin = -0.05
    ymax = 0.8

    mask = (
        (pcd_world[:,0] >= xmin) & (pcd_world[:,0] <= xmax) &
        (pcd_world[:,1] >= ymin) & (pcd_world[:,1] <= ymax) &
        (pcd_world[:,2] >= zmin) & (pcd_world[:,2] <= zmax)
    )

    pcd_world = pcd_world[mask]
    colors = colors[mask]

    # save pcd in world coordinate
    # save_pcd = np.hstack((pcd_world,colors))
    # np.save('./result/pcd_world.npy',save_pcd)

    calib.cams[0].pcd = calib.cams[0].cam2world(calib.cams[0].pcd[:,:3])
    calib.cams[1].pcd = calib.cams[0].cam2world(calib.cams[1].pcd[:,:3])
    calib.cams[2].pcd = calib.cams[0].cam2world(calib.cams[2].pcd[:,:3])
    visualize_point_cloud([calib.cams[0].pcd,calib.cams[1].pcd,calib.cams[2].pcd],['red','black','green'],cams=calib.cams)
    visualize_point_cloud([pcd_world],[colors],cams=calib.cams)