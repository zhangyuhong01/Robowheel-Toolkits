import os
import platform
import shutil
import json
import h5py
import tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.transform import Rotation as R
import cv2
import numpy as np
import torch


# 使用项目本地的URDF路径
URDF_LIBRARY_PATH = os.path.dirname(os.path.abspath(__file__))

if platform.system() == "Linux":
    _TMP_DIR = "/tmp"
elif platform.system() == "Windows":
    _TMP_DIR = "./tmp"
    if os.path.isdir("./.tmp"):
        shutil.rmtree("./.tmp")
    os.makedirs("./.tmp", exist_ok=True)
else:
    assert False, platform.system()
    

def posquat2tf(pos_quat):
    tf = np.zeros((pos_quat.shape[0], 4, 4))
    tf[:, :3, 3] = pos_quat[:, :3]
    quat = np.concatenate([pos_quat[:, 4:], pos_quat[:, 3:4]], axis=1)
    tf[:, :3, :3] = R.from_quat(quat).as_matrix()
    tf[:, 3, 3] = 1.0
    return tf

def decode_cam_binary_data_to_dir(cam_binary_data, target_dir: str):
    """Decode encoded camera binary (h265) into PNG files in target_dir using ffmpeg.

    Uses a direct subprocess call (no shell) and validates ffmpeg availability. Raises
    RuntimeError with stdout/stderr if ffmpeg fails, and a clear message if ffmpeg is
    not installed.
    """
    command_str = None
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    # Create a temporary file and close it before invoking ffmpeg (Windows requires this)
    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(dir=_TMP_DIR, delete=False, suffix=".h265") as f:
            f.write(bytes(cam_binary_data))
            f.flush()
            tmp_file = f.name

        # Ensure ffmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found. Please install ffmpeg and ensure it's in your PATH.")
        except subprocess.CalledProcessError:
            # ffmpeg present but returned non-zero for --version; still continue to attempt conversion
            pass

        # Build safe command without shell quoting issues
        out_pattern = os.path.join(target_dir, '%d.png')
        cmd = ['ffmpeg', '-y', '-i', tmp_file, out_pattern]
        command_str = " ".join(cmd)

        result = subprocess.run(cmd, shell=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (returncode={result.returncode})\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
    finally:
        # Clean up the temporary file if it was created
        try:
            if tmp_file and os.path.exists(tmp_file):
                os.remove(tmp_file)
        except Exception:
            pass


def get_images_from_dir(image_dir: str, target_t_ls: list = None):
    if target_t_ls is None:
        target_t_ls = range(len(os.listdir(image_dir)))
    
    images = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        image_files = [os.path.join(image_dir, f"{idx+1}.png") for idx in target_t_ls]
        results = executor.map(cv2.imread, image_files)
        for t_idx, bgr_image in enumerate(results):  
            images.append(bgr_image)
    images = np.stack(images, axis=0)[:, :, :, ::-1]  # RGB
    return images

   
class DataLoaderV3:
    
    def __init__(self, post_h5_file: str, which_hand: str = "right", target_camera: str = None):
        self.which_hand = which_hand
        self._root = h5py.File(post_h5_file, 'r')
        
        # 触觉点 FK 用外骨骼链，必须用 joints_glove（29 维）；proj_point_to_obj 手部 mesh 用 joints_mano（48 维），此处不读
        joints_path_glove = f"/dataset/observation/{which_hand}hand/joints_glove/data"
        joints_path_legacy = f"/dataset/observation/{which_hand}hand/joints/data"
        
        if joints_path_glove in self._root:
            self.joints_path = joints_path_glove
        elif joints_path_legacy in self._root:
            self.joints_path = joints_path_legacy
        else:
            raise AssertionError(f"{which_hand} data is not available (checked both joints_glove and joints)")
        
        self.num_frames = self._root[self.joints_path].shape[0]
        
        # handpose：合并文件用 handpose_glove（与 tactile/joints_glove 同源）；兼容旧文件用 handpose
        hand_grp = f"/dataset/observation/{which_hand}hand"
        if f"{hand_grp}/handpose_glove" in self._root:
            self.handpose_path = f"{hand_grp}/handpose_glove"
        elif f"{hand_grp}/handpose" in self._root:
            self.handpose_path = f"{hand_grp}/handpose"
        else:
            self.handpose_path = None
        if self.handpose_path is not None:
            self.source_camera: str = self._root[self.handpose_path].attrs["source_camera"]
            self.target_camera: str = target_camera if target_camera is not None else self.source_camera
        else:
            self.source_camera = None
            self.target_camera = None
        
        # print(f"dataloader's source_camera={self.source_camera}, target_camera={self.target_camera}")
        if f"/dataset/observation/image" in self._root:
            assert self.source_camera is not None and self.target_camera is not None, f"handpose (or handpose_glove) not in root for {which_hand}hand"
            assert f"/dataset/observation/image/{self.target_camera}" in self._root, f"target camera {self.target_camera} not in h5 root"
            self.intrinsic_matrix = self._root[f"/dataset/observation/image/{self.target_camera}/color/intrinsics"][:]
            self.image_width = self._root[f"/dataset/observation/image/{self.target_camera}/color/intrinsics"].attrs["width"]
            self.image_height = self._root[f"/dataset/observation/image/{self.target_camera}/color/intrinsics"].attrs["height"]
            src_extrinsics = self._root[f"/dataset/observation/image/{self.source_camera}/color/extrinsics"][:]
            tgt_extrinsics = self._root[f"/dataset/observation/image/{self.target_camera}/color/extrinsics"][:]
            if self.source_camera == self.target_camera:
                tgt_tf_rel_src = np.eye(4)
            elif self.source_camera.startswith("RGB_") and self.target_camera.startswith("RGB_"):
                tgt_tf_rel_src = np.linalg.inv(tgt_extrinsics) @ src_extrinsics
            elif self.source_camera.startswith("RGBD_") and target_camera.startswith("RGBD_"):
                tgt_tf_rel_src = np.linalg.inv(tgt_extrinsics) @ src_extrinsics
            elif self.source_camera.startswith("RGBD_") and self.target_camera.startswith("RGB_"):
                rgbd0_extrinsics = self._root[f"/dataset/observation/image/RGBD_0/color/extrinsics"][:]
                rgbd0_tf_rel_src = np.linalg.inv(rgbd0_extrinsics) @ src_extrinsics
                
                rgb1_extrinsics = self._root[f"/dataset/observation/image/RGB_Camera1/color/extrinsics"][:]
                tgt_tf_rel_rgb1 = np.linalg.inv(tgt_extrinsics) @ rgb1_extrinsics
                
                rgbd_rgb_extrinsics = json.loads(self._root["/dataset/observation/image/rgbd_rgb_extrinsic"][:].tolist()[0].decode('utf-8'))
                rgb1_tf_rel_rgbd0 = np.array(rgbd_rgb_extrinsics["RGBD_0_to_RGB_Camera1"])
                
                tgt_tf_rel_src = tgt_tf_rel_rgb1 @ rgb1_tf_rel_rgbd0 @ rgbd0_tf_rel_src
            elif self.source_camera.startswith("RGB_") and self.target_camera.startswith("RGBD_"):
                rgbd0_extrinsics = self._root[f"/dataset/observation/image/RGBD_0/color/extrinsics"][:]
                rgbd0_tf_rel_tgt = np.linalg.inv(rgbd0_extrinsics) @ tgt_extrinsics
                
                rgb1_extrinsics = self._root[f"/dataset/observation/image/RGB_Camera1/color/extrinsics"][:]
                src_tf_rel_rgb1 = np.linalg.inv(src_extrinsics) @ rgb1_extrinsics
                
                rgbd_rgb_extrinsics = json.loads(self._root["/dataset/observation/image/rgbd_rgb_extrinsic"][:].tolist()[0].decode('utf-8'))
                rgb1_tf_rel_rgbd0 = np.array(rgbd_rgb_extrinsics["RGBD_0_to_RGB_Camera1"])
                
                src_tf_rel_tgt = src_tf_rel_rgb1 @ rgb1_tf_rel_rgbd0 @ rgbd0_tf_rel_tgt
                tgt_tf_rel_src = np.linalg.inv(src_tf_rel_tgt)
            else:
                assert False
            os.makedirs(_TMP_DIR, exist_ok=True)
            self._temp_dir = tempfile.mkdtemp(dir=_TMP_DIR)
            decode_cam_binary_data_to_dir(cam_binary_data=self._root[f"/dataset/observation/image/{self.target_camera}/color/data"][:], target_dir=self._temp_dir)
            print(f"_temp_dir: {self._temp_dir}")
            assert len(os.listdir(self._temp_dir)) == self.num_frames, f"number of {self.target_camera}'s frames is not qual to joint's"
        else:
            self.intrinsic_matrix = None
            self.image_width = None
            self.image_height = None
            tgt_tf_rel_src = None
            self._temp_dir = None
        self.tgt_tf_rel_src = tgt_tf_rel_src

        self.hand_description_name = self._root[f"/dataset/observation/{which_hand}hand"].attrs.get("description", f"exoskeleton_hand_{which_hand}_1_0_description")
        self.urdf_file = os.path.join(URDF_LIBRARY_PATH, self.hand_description_name, "urdf", f"{self.hand_description_name}.urdf")
        if os.path.isfile(self.urdf_file):
            import pytorch_kinematics as pk
            try:
                self.chain = pk.build_chain_from_urdf(open(self.urdf_file).read().encode('utf-8'))
            except Exception as e:
                print(f"fail to build chain with urdf_file: {self.urdf_file}, error: {e}")
        else:
            self.chain = None
    
    def get_image(self, t: int):
        assert t < self.num_frames
        return get_images_from_dir(image_dir=self._temp_dir, target_t_ls=np.array([t]))[0]
    
    def get_joints(self, frame_indices):
        frame_indices = np.asarray(frame_indices)
        joints = self._root[self.joints_path][frame_indices]
        return joints
    
    def get_links_tf_rel_wrist(self, frame_indices) -> np.ndarray:
        assert self.chain is not None
        frame_indices = np.asarray(frame_indices)
        joints = self.get_joints(frame_indices=frame_indices)
        all_tfs = self.chain.forward_kinematics(torch.from_numpy(joints))
        links_tf_rel_wrist = {name: all_tfs[name].get_matrix().numpy() for name in all_tfs}  # [T, 4, 4]
        return links_tf_rel_wrist
    
    def get_objs_tf_rel_world(self, frame_indices):
        frame_indices = np.asarray(frame_indices)
        obj_poses = {}
        # 如果没有相机数据，使用单位矩阵（不做坐标变换）
        tgt_tf_rel_src = self.tgt_tf_rel_src if self.tgt_tf_rel_src is not None else np.eye(4)
        for grp_name in self._root[f"/dataset/observation"].keys():
            if grp_name.startswith("obj"):
                obj_id = self._root[f"/dataset/observation/{grp_name}"].attrs["obj_id"]
                obj_poses[obj_id] = tgt_tf_rel_src[None] @ posquat2tf(self._root[f"/dataset/observation/{grp_name}/data"][frame_indices, :7])
        return obj_poses
    
    def get_wrist_tf_rel_world(self, frame_indices):
        if self.handpose_path is not None:
            frame_indices = np.asarray(frame_indices)
            wrist_pose = posquat2tf(self._root[f"{self.handpose_path}/data"][frame_indices])  # [T, 4, 4]
            assert wrist_pose.shape == (len(frame_indices), 4, 4)
            # 如果没有相机数据，使用单位矩阵（不做坐标变换）
            tgt_tf_rel_src = self.tgt_tf_rel_src if self.tgt_tf_rel_src is not None else np.eye(4)
            return tgt_tf_rel_src[None] @ wrist_pose
        else:
            return None
    
    def get_tactile_data_dict(self, frame_indices):
        frame_indices = np.asarray(frame_indices)
        try:
            tactile_array = self._root[f"/dataset/observation/{self.which_hand}hand/tactile/data"][:]
            sensor_names = self._root[f"/dataset/observation/{self.which_hand}hand/tactile/data"].attrs["sensor_names"]
            sensor_lengths = self._root[f"/dataset/observation/{self.which_hand}hand/tactile/data"].attrs["sensor_lengths"]
        except Exception as e:
            print(f"fail to parse {self.which_hand} tactile, error: {e}")
            return 
        
        tactile_data_dict = {}
        cnt = 0
        for new_sensor_name, length in dict(zip(sensor_names, sensor_lengths)).items():
            tactile_data_dict[new_sensor_name] = tactile_array[:, cnt: cnt + int(length)]
            cnt = cnt + int(length)
        return tactile_data_dict
    
    def get_links_press_rel_sensor_base(self, frame_indices):
        if not self.hand_description_name.startswith("exoskeleton_hand"):
            return
        frame_indices = np.asarray(frame_indices)
        tactile_data_dict = self.get_tactile_data_dict(frame_indices=frame_indices)
        
        anchor_path = os.path.join(URDF_LIBRARY_PATH, self.hand_description_name, "anchor_points")
        links_press_tf_rel_link_base = {}
        links_press_force_rel_link_base = {}
        links_joint_force_rel_link_base = {}
        links_dist_force_rel_link_base = {}
        for sensor_name in tactile_data_dict.keys():
            tactile = tactile_data_dict[sensor_name][frame_indices, :]  # [T, D]
            sensor_npy_file = os.path.join(anchor_path, f"{sensor_name}.npy")
            if not os.path.isfile(sensor_npy_file):
                print(f"cannot find anchor, skip sensor {sensor_npy_file}")
                continue
            
            points_tf_rel_sensor_base = np.load(sensor_npy_file)  # [M, 4, 4]
            assert tactile.shape[-1] == 30 + 3 * points_tf_rel_sensor_base.shape[0], f"{tactile.shape}, {points_tf_rel_sensor_base.shape}"
            
            points_pos_rel_sensor_base = points_tf_rel_sensor_base[:, :3, 3]  # [M, 3]
            
            joint_force = tactile[:, :3]
            press_x = tactile[:, 3:27:4]  # [T, 6]
            press_y = tactile[:, 4:27:4]  # [T, 6]
            press_z = tactile[:, 5:27:4]  # [T, 6]
            press_nf = tactile[:, 6:27:4]  # [T, 6]
            
            if np.any(press_x > 1.0) or np.any(press_y > 1.0) or np.any(press_z > 1.0):
                press_x *= 0.001
                press_y *= 0.001
                press_z *= 0.001
            
            # print(press_nf[:, 0].tolist())
            threshold = 0.3
            press_x[press_nf < threshold] = 0.0
            press_y[press_nf < threshold] = 0.0
            press_z[press_nf < threshold] = 0.0
            press_pos = np.stack([press_x, press_y, press_z], axis=-1)  # [T, 6, 3]
            
            dis_fx = tactile[:, 30::3]  # [T, M]
            dis_fy = tactile[:, 31::3]  # [T, M]
            dis_fz = tactile[:, 32::3]  # [T, M]
            
            dis_force = np.stack([dis_fx, dis_fy, dis_fz], axis=-1)  # [T, M, 3]
        
            dis = np.square(press_pos[:, :, None] - points_pos_rel_sensor_base[None, None]).sum(-1)
            # ([T, 6, 1, 3] - [1, 1, M, 3]).pow(2).mean() -> [T, 6, M]
            # print(press_pos.tolist())
            nearest_point_index = np.argmin(dis, axis=2)  # [T, 6]
            
            press_tf_rel_sensor_base = np.take_along_axis(points_tf_rel_sensor_base[None, None, :, :, :], nearest_point_index[..., None, None, None], axis=2).squeeze(2)  # shape [T, 6, 4, 4]
            # [1, 1, M, 4, 4] [T, 6, 1, 1, 1] -> [T, 6, 1, 4, 4] -> [T, 6, 4, 4]
            
            press_force = np.take_along_axis(dis_force[:, None, :, :], nearest_point_index[..., None, None], axis=2).squeeze(2)  # shape [T, 6, 3]
            # [T, 1, M, 3] [T, 6, 1, 1] -> [T, 6, 1, 3] -> [T, 6, 1, 3] -> [T, 6, 3]
            # print(press_force[:, 0].tolist())
            # print("=====================")
            links_press_tf_rel_link_base[sensor_name] = press_tf_rel_sensor_base
            links_press_force_rel_link_base[sensor_name] = press_force
            links_joint_force_rel_link_base[sensor_name] = joint_force
            links_dist_force_rel_link_base[sensor_name] = dis_force
            # print(np.concatenate([press_tf_rel_sensor_base[:, 0, :3, 3], press_force[:, 0, :], nearest_point_index[:, 0, None]], axis=-1))
        # print(press_pos_f, max_f_indices, press_pos_f.shape)
        
        return links_press_tf_rel_link_base, links_press_force_rel_link_base, links_joint_force_rel_link_base, links_dist_force_rel_link_base
    
    def get_links_tf_rel_world(self, frame_indices) -> np.ndarray:
        wrist_tf_rel_world = self.get_wrist_tf_rel_world(frame_indices=frame_indices)  # [T, 4, 4]
        links_tf_rel_wrist = self.get_links_tf_rel_wrist(frame_indices=frame_indices)  # {[T, 4, 4]}
        links_tf_rel_world = {}
        for link_name, link_tf_rel_wrist in links_tf_rel_wrist.items():
            assert link_tf_rel_wrist.shape == (len(frame_indices), 4, 4)
            assert np.all(link_tf_rel_wrist[:, 3, 3] == 1)
            link_tf_rel_world = wrist_tf_rel_world @ link_tf_rel_wrist  # [T, 4, 4]
            links_tf_rel_world[link_name] = link_tf_rel_world
        return links_tf_rel_world
    
    def get_links_press_press_rel_world(self, frame_indices) -> np.ndarray:
        # 触觉点世界坐标计算链：触觉点世界 = handpose_glove @ FK(joints_glove) @ point_in_link
        # 因此 handpose_glove 与 joints_glove/tactile 必须来自同一源（合并文件时用 handpose_glove）
        ret = self.get_links_press_rel_sensor_base(frame_indices=frame_indices)  # {[T, k, 4, 4]}, {[T, k, 3]}
        if ret is None:
            return None
        links_press_tf_rel_link_base, links_press_force_rel_link_base, links_joint_force_rel_link_base, links_dist_force_rel_link_base = ret
        assert set(links_press_tf_rel_link_base.keys()) == set(links_press_force_rel_link_base.keys())
        
        links_tf_rel_world = self.get_links_tf_rel_world(frame_indices=frame_indices)  # {[T, 4, 4]}
        links_press_tf_rel_world = {}
        for sensor_name in links_press_tf_rel_link_base.keys():
            link_press_tf_rel_link_base = links_press_tf_rel_link_base[sensor_name]
            assert link_press_tf_rel_link_base.shape == (len(frame_indices), link_press_tf_rel_link_base.shape[1], 4, 4)
            assert links_press_force_rel_link_base[sensor_name].shape == (len(frame_indices), link_press_tf_rel_link_base.shape[1], 3)
            link_press_tf_rel_world = links_tf_rel_world[sensor_name][:, None] @ link_press_tf_rel_link_base
            # [T, 1, 4, 4] @ [T, k, 4, 4]
            links_press_tf_rel_world[sensor_name] = link_press_tf_rel_world
            
        return links_press_tf_rel_world, links_press_force_rel_link_base, links_joint_force_rel_link_base, links_dist_force_rel_link_base
    
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def close(self):
        if self._temp_dir is not None:
            shutil.rmtree(self._temp_dir)


if __name__ == "__main__":
    # 使用 merge_hdf5.py 合并后的文件（与 proj_point_to_obj 默认路径一致）
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    post_h5_file = os.path.join(_script_dir, "data", "hdf5", "100034", "episode_577_102812_89_120078_merged.hdf5")
    
    loader = DataLoaderV3(post_h5_file=post_h5_file, which_hand="left")
    frame_indices = list(range(loader.num_frames))
    
    image = loader.get_image(t=0)
    joints = loader.get_joints(frame_indices=frame_indices)
    links_tf = loader.get_links_tf_rel_world(frame_indices=frame_indices)  # {[T, 4, 4]}
    src_links_press_tf_rel_world, src_links_force_rel_link_base, links_joint_force_rel_link_base, links_dist_force_rel_link_base = loader.get_links_press_press_rel_world(frame_indices=frame_indices)  
    # [T, K, 6, 4, 4], [T, K, 6, 3], [T, K, M, 3]
    wrist_tf = loader.get_wrist_tf_rel_world(frame_indices=frame_indices)  # [T, 4, 4]
    objs_tf = loader.get_objs_tf_rel_world(frame_indices=frame_indices)  # [T, 4, 4]
    loader.close()
    
    print(links_dist_force_rel_link_base["M42L"][-1])
    
    
   

    
        
        
