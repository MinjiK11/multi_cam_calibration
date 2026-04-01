import numpy as np
import cv2
import open3d as o3d

class Camera():
    def __init__(self, cam_idx):
        self.cam_idx = cam_idx
        self.skew = 0 # TODO

        self.tar2ref_rvec = np.array([0.0,0.0,0.0])
        self.tar2ref_tvec = np.array([0.0,0.0,0.0])

    def setIntrinsic(self, intrinsic):
        '''
        set intrinsic parameter from .yaml file (case for fixed intrinsic)
        '''
        self.fx = intrinsic["fx"]
        self.cx = intrinsic["cx"]
        self.fy = intrinsic["fy"]
        self.cy = intrinsic["cy"]

        self.dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) # TODO

    def getIntrinsic(self):
        '''
        construct and return 3x3 intrinsic matrix
        '''
        self.mtx = np.array([(self.fx,0,self.cx),(0,self.fy,self.cy),(0,0,1)])

        return self.mtx
    
    def setExtrinsic(self,rvec,tvec):
        '''
        set extrinsic parameter
        '''
        self.rvec = rvec
        self.tvec = tvec

    def getExtrinsic(self):
        '''
        construct and return 3x4 w2c transformation matrix

        '''
        R, _ = cv2.Rodrigues(self.rvec)  # Convert rotation vector to rotation matrix
        self.w2c = np.hstack((R, self.tvec))

        return self.w2c

    def setTar2Ref(self,rvec,tvec):
        self.tar2ref_rvec = rvec
        self.tar2ref_tvec = tvec

    def getTar2Ref(self):
        '''
        construct and return 3x4 tar2ref transformation matrix
        '''
        R, _ = cv2.Rodrigues(self.tar2ref_rvec)  # Convert rotation vector to rotation matrix
        self.tar2ref = np.hstack((R, self.tar2ref_tvec.reshape(-1,1)))

        return self.tar2ref
    
    def cam2img(self, camx, camy):
        r2 = camx**2 + camy**2
        r4 = r2**2

        r_coeff = 1 + self.dist[0] * r2 + self.dist[1] * r4

        imgx = camx * r_coeff * self.fx + self.skew * camy * r_coeff + self.cx
        imgy = camy * r_coeff * self.fy + self.cy

        return imgx, imgy

    # def img2cam(self, imgpt)
    #     x = (imgx-cam.cx) * z / cam.fx 
    #     y = (imgy-cam.cy) * z / cam.fy 

    #     imgpt = np.array([imgx,imgy],dtype=np.float32).reshape(-1,2)
    #     imgpt = np.expand_dims(imgpt,axis=1)

    #     breakpoint()
    #     campt = cv2.undistortPoints(imgpt, self.getIntrinsic(), self.dist)
    #     campt = campt.reshape(-1,2)

    #     return campt[0], campt[1]
    
    def world2cam(self,objp):
        '''
        convert 3D world point to camera point

        objp: 3D world point

        campt: point in camera coordinate
        '''
        rmat = self.getExtrinsic()[:3,:3]
        tvec = self.getExtrinsic()[:3,3]

        w2c = rmat @ objp + tvec

        return w2c

    def cam2world(self,campt):
        '''
        campt: points in camera coordinate (when z=1) Nx3
        '''
        rmat = self.getExtrinsic()[:3,:3]
        tvec = self.getExtrinsic()[:3,3].reshape(-1,1)

        c2w = (rmat.T @ campt.T - rmat.T @ tvec).T

        return c2w
    
    def world2img(self,objpt):
        campt = self.world2cam(objpt)
        camx = campt[0]/campt[-1]
        camy = campt[1]/campt[-1]

        imgx, imgy = self.cam2img(camx, camy)

        return imgx, imgy

