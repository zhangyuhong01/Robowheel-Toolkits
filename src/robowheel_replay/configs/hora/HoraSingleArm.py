from dataclasses import dataclass, field
from src.robowheel_replay.configs.schema import (
    DatasetToSceneConfig, SceneEntity, H5Source, EntityType
)
from src.robowheel_replay.configs.registry import register_dataset_config

@register_dataset_config("hora/HoraSingleArm")
@dataclass
class HoraSingleArmConfig(DatasetToSceneConfig):
    dataset_id: str = "hora/HoraSingleArm"
    data_root: str = "data"

    hf_repo_id: str = "HORA-DB/HORA"        # Default Repo
    hf_default_file: str = "SingleArm/place_metal_can_on_mat/place_metal_can_on_mat_chunk_0.hdf5"
    hf_assets_dir: str = "hora_assets"
    # Local dev override (Optional)
    
    entities: dict = field(default_factory=lambda: {
        
        "ur5_robot0": SceneEntity(
            uid="ur5_robot0",
            entity_type=EntityType.ARTICULATION,
            usd_path="data/hora_assets/robots/ur5/usd/ur5_2f_gripper_rotate37.usd",
            joint_sources=[
                # Arm joints
                H5Source(path="obs/robot0_joint_states"),
                # Gripper joint
                H5Source(path="obs/robot0_gripper_states", indices=[0,1,5,2,4,3])
            ],
            root_sources=[
                # Base drive: Sliced from robot_states (Pos:3 + Quat:4)
                H5Source(
                    path="robot_states",
                    indices=[0, 1, 2, 3, 4, 5, 6] 
                )
            ]
        ),
    })
