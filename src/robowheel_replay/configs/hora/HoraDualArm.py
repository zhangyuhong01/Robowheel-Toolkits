from dataclasses import dataclass, field
from src.robowheel_replay.configs.schema import (
    DatasetToSceneConfig, SceneEntity, H5Source, EntityType
)
from src.robowheel_replay.configs.registry import register_dataset_config

@register_dataset_config("hora/HoraDualArm")
@dataclass
class HoraDualArmConfig(DatasetToSceneConfig):
    dataset_id: str = "hora/HoraDualArm"
    data_root: str = "data"

    hf_repo_id: str = "HORA-DB/HORA"        # Default Repo
    hf_default_file: str = "DualArm/handover_metal_can_on_mat/handover_metal_can_on_mat_chunk_59.hdf5"
    hf_assets_dir: str = "hora_assets"
    # Local dev override (Optional)
    
    entities: dict = field(default_factory=lambda: {
        
        "piper_robot0": SceneEntity(
            uid="piper_robot0",
            entity_type=EntityType.ARTICULATION,
            usd_path="data/hora_assets/robots/piper/usd/piper_tcp.usd",
            init_pos=(0.0, 0.29, 0.0), 
            joint_sources=[
                # Arm joints
                H5Source(path="obs/robot0_joint_states"),
                # Gripper joint
                H5Source(path="obs/robot0_gripper_states")
            ],
            root_sources=[
                # Base drive: Sliced from robot_states (Pos:3 + Quat:4)
                H5Source(
                    path="robot_states",
                    indices=[0, 1, 2, 3, 4, 5, 6] 
                )
            ]
        ),

        "piper_robot1": SceneEntity(
            uid="piper_robot1",
            entity_type=EntityType.ARTICULATION,
            usd_path="data/hora_assets/robots/piper/usd/piper_tcp.usd",
            init_pos=(0.0, -0.29, 0.0),
            joint_sources=[
                # Arm joints
                H5Source(path="obs/robot1_joint_states"),
                # Gripper joint
                H5Source(path="obs/robot1_gripper_states")
            ],
            root_sources=[
                # Base drive: Sliced from robot_states (Pos:3 + Quat:4)
                H5Source(
                    path="robot_states",
                    indices=[7, 8, 9, 10, 11, 12, 13] 
                )
            ]
        ),
    })
