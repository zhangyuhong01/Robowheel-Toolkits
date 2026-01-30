from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from src.robowheel_replay.configs.assets import AssetLibrary 
import os
from loguru import logger

from src.robowheel_replay.utils.hf_client import HFLoader

class EntityType(Enum):
    ARTICULATION = "articulation"  # Robots, grippers, etc.
    RIGID_OBJECT = "rigid_object"  # Props, dynamic objects
    STATIC_SCENE = "static_scene"  # Static background

@dataclass
class H5Source:
    """
    Descriptor for a specific tensor slice in HDF5.
    """
    path: str                   # Internal HDF5 path (e.g. "obs/joint_pos")
    indices: Optional[List[int]] = None # Specific dimension indices to slice

@dataclass
class SceneEntity:
    """Represents any entity in the scene (robot, object)."""
    uid: str                 # Unique ID
    usd_path: str            # Path to USD asset
    entity_type: EntityType
    
    # === Initial State (For reset or missing data) ===
    init_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    init_rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0) # w,x,y,z
    
    # === Dynamic Drive Sources ===
    # For Articulations (Joint positions)
    joint_sources: List[H5Source] = field(default_factory=list)
    
    # For Root Pose (Floating base robots or objects)
    root_sources: List[H5Source] = field(default_factory=list)
    
    # Physics Property
    is_fixed_base: bool = True 

@dataclass
class DatasetToSceneConfig:
    dataset_id: str
    data_root: str
    hf_repo_id: str
    hf_assets_dir: str
    hf_default_file: Optional[str] = None

    # Registry of all scene entities
    entities: Dict[str, SceneEntity] = field(default_factory=dict)

    # Internal asset library helper
    _asset_lib: Optional[AssetLibrary] = field(default=None, init=False, repr=False)
    local_hdf5_path: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """
        Assets & Data Strategy: "Local Anchor with Remote Fallback"
        """

        local_assets_path = os.path.join(self.data_root, self.hf_assets_dir)
        

        if not os.path.exists(local_assets_path):
            logger.warning(f"Assets missing at {local_assets_path}. Syncing from HF...")
            if self.hf_repo_id:
                try:
                    HFLoader.sync_assets(
                        repo_id=self.hf_repo_id,
                        remote_dir=self.hf_assets_dir, 
                        local_dir=self.data_root
                    )
                except Exception as e:
                    logger.error(f"Failed to sync assets: {e}")
        
        if os.path.exists(local_assets_path):
            self._asset_lib = AssetLibrary(local_assets_path)
        else:
            logger.critical(f"Asset path unreachable: {local_assets_path}")

        if self.hf_default_file:
            expected_h5_path = os.path.join(self.data_root, self.hf_default_file)
            
            if os.path.exists(expected_h5_path):
                self.local_hdf5_path = expected_h5_path
                # logger.info(f"Found local default HDF5: {self.local_hdf5_path}")
            else:
                logger.warning(f"Default HDF5 missing at {expected_h5_path}. Downloading...")
                try:
                    self.local_hdf5_path = HFLoader.get_file(
                        repo_id=self.hf_repo_id,
                        filename=self.hf_default_file,
                        local_dir=self.data_root
                    )
                except Exception as e:
                    logger.error(f"Failed to download default HDF5: {e}")
    def add_object(self, obj_name: str):
        """
        Dynamically add a RigidObject to the entity registry by name.
        """
        # Skip if already exists
        if obj_name in self.entities:
            return

        if not self._asset_lib:
            raise RuntimeError("Cannot add object: AssetLibrary not initialized (assets_root missing).")

        usd_path = self._asset_lib.get_usd_path(obj_name)
        
        # Infer HDF5 Keys
        pos_key = f"obs/object_{obj_name}_pos"
        quat_key = f"obs/object_{obj_name}_quat"
        
        self.entities[obj_name] = SceneEntity(
            uid=obj_name,
            entity_type=EntityType.RIGID_OBJECT,
            usd_path=usd_path,
            init_pos=(0.0, 0.0, -10.0), # Default to hidden if no data
            
            # Auto-generate Data Sources
            root_sources=[
                H5Source(path=pos_key),  # Pos (T, 3)
                H5Source(path=quat_key)  # Quat (T, 4)
            ],
            is_fixed_base=False
        )