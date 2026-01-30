import os
import sys
import h5py
import torch
import numpy as np
import re
from typing import Optional
sys.dont_write_bytecode = True
# Isaac Lab Imports
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils

from loguru import logger
from src.robowheel_replay.configs.registry import get_config_class
from src.robowheel_replay.configs.schema import H5Source, EntityType

class HORASceneBuilder:
    """
    Orchestrator class that loads HDF5 (HORA format) and builds Isaac Lab Scene.
    """

    def __init__(self, dataset_identifier: str, hdf5_path: Optional[str] = None):
        """
        Args:
            dataset_identifier: Config Registry ID (e.g. 'hora/dual_arm_handover')
            hdf5_path: Absolute path to HDF5 file
        """
        self.logger = logger.bind(module="SceneBuilder")
        
        # 1. Load Config
        try:
            config_cls = get_config_class(dataset_identifier)
            self.config = config_cls()
        except Exception as e:
            self.logger.critical(f"Config not found for {dataset_identifier}: {e}")
            raise

        # 2. Resolve HDF5 Path
        target_path = hdf5_path
        
        if not target_path:
            if self.config.local_hdf5_path:
                target_path = self.config.local_hdf5_path
                self.logger.info(f"Using default HDF5 from config: <cyan>{target_path}</cyan>")
            else:
                self.logger.critical("No HDF5 path provided and failed to load default from Config.")
                raise ValueError("HDF5 Source Missing")

        # 3. Verify Final Path
        if not os.path.exists(target_path):
            self.logger.critical(f"Target HDF5 file not found: {target_path}")
            raise FileNotFoundError(target_path)
            
        self.hdf5_path = target_path

        # 4. Pre-check HDF5 structure
        with h5py.File(self.hdf5_path, 'r') as f:
            if "data" not in f:
                raise ValueError("HDF5 structure invalid: Root group 'data' missing.")
            self.total_demos = len(f["data"].keys())
            self.logger.opt(colors=True).info(f"Initialized <bold>{dataset_identifier}</bold>. Dataset: <green>{self.total_demos}</green> demos.")

        self.current_episode_idx = -1
        self.current_episode_data = {}

    def build_scene_cfg(self) -> InteractiveSceneCfg:
        """
        Dynamically build Isaac Lab Scene Config based on EntityType.
        """
        use_scene = False
        scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
        scene_cfg.dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        )

        for name, entity_def in self.config.entities.items():
            if entity_def.entity_type == EntityType.STATIC_SCENE:
                use_scene = True
                prim_cfg = AssetBaseCfg(
                    prim_path=f"/World/envs/env_.*/{name}",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=entity_def.usd_path,
                        scale=(1.0, 1.0, 1.0), 
                        visible=True
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(
                        pos=entity_def.init_pos,
                        rot=entity_def.init_rot
                    )
                )
                setattr(scene_cfg, name, prim_cfg)
            
            # === Type 1: Articulation ===
            elif entity_def.entity_type == EntityType.ARTICULATION:
                prim_cfg = ArticulationCfg(
                    prim_path=f"/World/envs/env_.*/{name}",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=entity_def.usd_path,
                        activate_contact_sensors=False,
                        articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=entity_def.is_fixed_base),
                    ),
                    init_state=ArticulationCfg.InitialStateCfg(
                        pos=entity_def.init_pos,
                        rot=entity_def.init_rot,
                        joint_pos={".*": 0.0}, 
                    ),
                    actuators={
                        "all_joints": ImplicitActuatorCfg(
                            joint_names_expr=[".*"],
                            stiffness=None,  
                            damping=None  
                        ),
                    },
                )
                setattr(scene_cfg, name, prim_cfg)
                
            # === Type 2: Rigid Object ===
            elif entity_def.entity_type == EntityType.RIGID_OBJECT:
                if entity_def.is_fixed_base:
                    rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True)
                else:
                    rigid_props = sim_utils.RigidBodyPropertiesCfg()
                prim_cfg = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/{name}",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=entity_def.usd_path,
                        rigid_props=rigid_props,
                        scale=(1.0, 1.0, 1.0)
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=entity_def.init_pos,
                        rot=entity_def.init_rot
                    )
                )
                setattr(scene_cfg, name, prim_cfg)
            
            # === Error Handling ===
            else:
                self.logger.warning(f"Skipping entity '{name}': Unknown type {entity_def.entity_type}.")
        
        if not use_scene:
            scene_cfg.ground = AssetBaseCfg(
                prim_path="/World/GroundPlane",
                spawn=sim_utils.GroundPlaneCfg()
            )

        return scene_cfg

    def _load_sources(self, sources: list, get_data_fn: callable) -> Optional[torch.Tensor]:
        """
        Helper: Read and concatenate all sources in the list.
        """
        if not sources:
            return None
            
        tensors_to_cat = []
        
        for src in sources:
            # 1. Get raw data via callback
            raw_data = get_data_fn(src.path)
            
            if raw_data is None:
                raise KeyError(f"Required HDF5 key not found: '{src.path}'")
            
            data = raw_data.clone()
            
            # 2. Slice (Indices)
            if src.indices is not None:
                try:
                    data = data[:, src.indices]
                except IndexError as e:
                    self.logger.error(f"Index error slicing '{src.path}'. Data shape: {data.shape}, Indices: {src.indices}")
                    raise e
            
            # 3. Dimension fix (Ensure T, D)
            if data.dim() == 1:
                data = data.unsqueeze(1)
            
            tensors_to_cat.append(data)
            
        # 5. Concatenate
        if not tensors_to_cat:
            return None
            
        try:
            return torch.cat(tensors_to_cat, dim=1)
        except RuntimeError as e:
            self.logger.error(f"Concatenation failed. Shapes: {[t.shape for t in tensors_to_cat]}")
            raise e

    def load_episode(self, episode_idx: int):
        """
        Load data from HDF5 and build full trajectory tensors for all entities.
        """
        # Concise start log
        self.logger.info(f"Loading Episode {episode_idx}...")
        
        demo_key = f"demo_{episode_idx}"
        
        with h5py.File(self.hdf5_path, 'r') as f:
            if demo_key not in f["data"]:
                raise ValueError(f"Episode {demo_key} not found.")
            
            demo_grp = f["data"][demo_key]

            self._scan_and_register_objects(demo_grp)
            
            # === 1. Data Retrieval Closure (Cached) ===
            raw_data_cache = {} 
            
            def get_tensor_with_cache(key: str) -> Optional[torch.Tensor]:
                if key not in raw_data_cache:
                    if key in demo_grp:
                        raw_data_cache[key] = torch.tensor(demo_grp[key][:], dtype=torch.float32)
                    elif key in f["data"]: 
                        raw_data_cache[key] = torch.tensor(f["data"][key][:], dtype=torch.float32)
                    else:
                        return None
                return raw_data_cache[key]

            # === 2. Build Trajectories ===
            self.entity_trajectories = {}
            total_frames = 0

            # Iterate silently unless there are warnings
            for name, entity in self.config.entities.items():
                if entity.entity_type == EntityType.STATIC_SCENE:
                    continue
                
                # Check for missing sources definition
                if not entity.joint_sources:
                    # Keep this warning, it indicates a config issue
                    self.logger.warning(f"Entity '{name}' has NO joint_sources defined.")

                # A. Process Joints
                joint_traj = self._load_sources(entity.joint_sources, get_tensor_with_cache)
                if joint_traj is None and entity.joint_sources:
                     self.logger.warning(f"Failed to load joint_traj for {name}")

                # B. Process Root Pose
                root_traj = self._load_sources(entity.root_sources, get_tensor_with_cache)

                # C. Record Total Frames
                if total_frames == 0:
                    if joint_traj is not None:
                        total_frames = joint_traj.shape[0]
                    elif root_traj is not None:
                        total_frames = root_traj.shape[0]

                # D. Store
                self.entity_trajectories[name] = {
                    "joint_pos": joint_traj,
                    "root_pose": root_traj
                }

            self.current_episode_idx = episode_idx
            self.current_total_frames = total_frames
            
            del raw_data_cache
            
            # Single success log at the end
            self.logger.success(f"Episode {episode_idx} prepared. Frames: {total_frames}")

    def _scan_and_register_objects(self, demo_grp):
        """
        Scan HDF5 keys for dynamic objects matching 'object_{name}_pos' and register them.
        """
        if "obs" not in demo_grp:
            return

        obs_grp = demo_grp["obs"]
        all_keys = list(obs_grp.keys())
        
        pattern = re.compile(r"object_(.+)_pos")
        found_objects = set()
        
        for key in all_keys:
            match = pattern.match(key)
            if match:
                found_objects.add(match.group(1))
        
        if found_objects:
            self.logger.info(f"Auto-discovered objects: {found_objects}")
            for obj_name in found_objects:
                self.config.add_object(obj_name)

    def get_frame_state(self, frame_idx: int) -> dict:
        """
        Fast lookup for frame state.
        """
        if self.current_episode_idx == -1: return {}
        
        if frame_idx >= self.current_total_frames:
            frame_idx = self.current_total_frames - 1
            
        result = {}
        for name, traj in self.entity_trajectories.items():
            state = {}
            
            # Joints
            state["joint_pos"] = traj["joint_pos"][frame_idx] if traj["joint_pos"] is not None else None
                
            # Root Pose
            if traj["root_pose"] is not None:
                state["root_pose"] = traj["root_pose"][frame_idx]
            else:
                state["root_pose"] = None
                
            result[name] = state
            
        return result
