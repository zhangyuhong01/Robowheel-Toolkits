import torch
from typing import Dict, Any, Optional

# Isaac Lab Imports
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from loguru import logger

class IsaacReplayEnv:
    """
    Wrapper for Isaac Lab simulation environment handling HDF5 playback.
    Manages SimulationContext, InteractiveScene, and state writing.
    """
    def __init__(self, scene_cfg: InteractiveSceneCfg, device: str = "cuda:0"):
        """
        Args:
            scene_cfg: Scene configuration
            device: Computation device
        """
        self.device = device
        
        # 1. Configure Simulation Context
        self.sim_cfg = SimulationCfg(dt=0.01, device=device)
        self.sim = SimulationContext(self.sim_cfg)
        
        # Default camera view
        self.sim.set_camera_view([1.5, 0.0, 1.5], [0.0, 0.0, 0.0])

        # 2. Create Scene
        logger.info("Creating InteractiveScene...")
        self.scene = InteractiveScene(cfg=scene_cfg)

        # 3. Reset
        logger.info("Resetting simulation...")
        self.sim.reset()
        
        self._cache_registries()

    def _cache_registries(self):
        """Log current entity registries for inspection."""
        logger.info("--- Inspecting Scene Registries ---")
        logger.info(f"Scene Articulations: {list(self.scene.articulations.keys())}")
        logger.info(f"Scene Rigid Objects: {list(self.scene.rigid_objects.keys())}")

    def reset(self):
        self.sim.reset()

    def set_state(self, state_dict: Dict[str, Any]):
        """
        Write frame state to simulation
        
        Args:
            state_dict: Dictionary from builder.get_frame_state(i)
                        Format: {"robot_name": {"joint_pos": ..., "root_pose": ...}, ...}
        """
        for name, entity_data in state_dict.items():
            # 1. Find Entity
            sim_entity = self._get_entity(name)
            if sim_entity is None:
                continue

            # 2. Apply Joint Positions
            if entity_data.get("joint_pos") is not None:
                j_pos = entity_data["joint_pos"]
                if torch.isnan(j_pos).any():
                    continue
                j_pos = j_pos.to(self.device).unsqueeze(0)
                j_vel = torch.zeros_like(j_pos)
                sim_entity.write_joint_state_to_sim(position=j_pos, velocity=j_vel)

            # 3. Apply Root Pose
            if entity_data.get("root_pose") is not None:
                r_pose = entity_data["root_pose"]
                if torch.isnan(r_pose).any():
                    continue

                # Adjust dimensions: (7,) -> (1, 7)
                r_pose = r_pose.to(self.device).unsqueeze(0)
                r_vel = torch.zeros((r_pose.shape[0], 6), device=self.device)
                
                sim_entity.write_root_pose_to_sim(root_pose=r_pose)
                sim_entity.write_root_velocity_to_sim(root_velocity=r_vel)

    def step(self):
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(dt=self.sim.get_physics_dt())

    def _get_entity(self, name: str):
        if name in self.scene.articulations:
            return self.scene.articulations[name]
        elif name in self.rigid_objects:
            return self.scene.rigid_objects[name]
        return None
    
    @property
    def rigid_objects(self):
        return self.scene.rigid_objects

    @property
    def articulations(self):
        return self.scene.articulations