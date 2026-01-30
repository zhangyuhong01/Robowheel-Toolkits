import time
import argparse
import sys
import os
from loguru import logger
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # Use mirror for faster downloads

# --- Logger Configuration ---
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>", level="INFO", colorize=True)

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from isaaclab.app import AppLauncher

# --- App & Arguments Setup ---
parser = argparse.ArgumentParser()
parser.add_argument("--hdf5_path", type=str, default=None, help="Optional: Override default HDF5 file")
parser.add_argument("--dataset_id", type=str, default="hora/HoraDualArm")
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--fps", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from src.robowheel_replay.core.scene_builder import HORASceneBuilder
from src.robowheel_replay.sim.isaac_env import IsaacReplayEnv

@logger.catch
def main():
    # --- Initialization ---
    logger.info("Initializing scene builder and environment...")
    
    builder = HORASceneBuilder(dataset_identifier=args.dataset_id, hdf5_path=args.hdf5_path)
    builder.load_episode(args.episode)

    scene_cfg = builder.build_scene_cfg()
    env = IsaacReplayEnv(scene_cfg=scene_cfg)
    import omni.kit.commands
    omni.kit.commands.execute("ChangeSetting", path="rtx/translucency/enabled", value=True)
    omni.kit.commands.execute("ChangeSetting", path="rtx/raytracing/fractionalCutoutOpacity", value=True)

    logger.info(f"Starting replay. Total Frames: {builder.current_total_frames} | Target FPS: {args.fps}")
    
    frame_idx = 0
    target_dt = 1.0 / args.fps
    last_log_time = time.time()
    log_interval = args.fps  # Log roughly once per second

    # --- Main Loop ---
    while simulation_app.is_running():
        loop_start = time.time()

        # 1. Physics & State Update
        state = builder.get_frame_state(frame_idx)
        env.set_state(state)
        env.step()

        # 2. FPS Limiter (Sleep if running too fast)
        compute_dt = time.time() - loop_start
        if compute_dt < target_dt:
            time.sleep(target_dt - compute_dt)

        # 3. Logging (Prevents flooding, logs approx once/sec)
        if frame_idx % log_interval == 0 and frame_idx > 0:
            current_time = time.time()
            # Calculate actual FPS based on time elapsed since last log
            actual_fps = log_interval / (current_time - last_log_time)
            logger.info(f"Progress: {frame_idx}/{builder.current_total_frames} | FPS: {actual_fps:.1f}")
            last_log_time = current_time

        # 4. Loop Logic
        frame_idx += 1
        if frame_idx >= builder.current_total_frames:
            logger.info("Episode finished. Resetting...")
            frame_idx = 0

    simulation_app.close()

if __name__ == "__main__":
    main()