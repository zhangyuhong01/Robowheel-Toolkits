import h5py
import shutil
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
import os
from PIL import Image
import numpy as np


TASK_PROMPT="Pick up the fruit and put it on the mat."

def main(hdf5_dir: str, *, push_to_hub: bool = False):
    hdf5_dir = Path(hdf5_dir)
    #TODO:REPO_NAME & HF_LEROBOT_HOME 
    REPO_NAME = "NAME of YOUR REPO"
    HF_LEROBOT_HOME = "NAME of YOUR HF LEROBOT HOME DIRECTORY"  

    output_path = Path(HF_LEROBOT_HOME) / REPO_NAME

    if output_path.exists():
        shutil.rmtree(output_path)
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="UR5",
        fps=30,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )


    for hdf5_file in hdf5_dir.glob("*.hdf5"):
        with h5py.File(hdf5_file, "r") as f:


            #TODO:check keys
            images = f["data"]['demo_0']['obs']['agentview_rgb'][:]# numpy array
            wrist_images = f["data"]['demo_0']['obs']['eye_in_hand_rgb'][:]
            states =f["data"]['demo_0']['obs']['robot0_joint_pos'][:]
            gripper_states = f["data"]['demo_0']['obs']['robot0_gripper_qpos'][:]
            actions = f["data"]['demo_0']['actions'][:,:6]
            gripper_action = f["data"]['demo_0']['actions'][:,6:]

            gripper_states_bin = (~np.all(gripper_states <= 0.001, axis=1)).astype(np.float32)
            gripper_action_bin = (~np.all(gripper_action <= 0.001, axis=1)).astype(np.float32)
            states = np.concatenate([states, gripper_states_bin[:, None]], axis=1)
            actions = np.concatenate([actions, gripper_action_bin[:, None]], axis=1)
            task = TASK_PROMPT  # or f["task"] #TODO

            for i in range(images.shape[0]):
                img = Image.fromarray(images[i])
                wrist_img = Image.fromarray(wrist_images[i])

                img_resized = img.resize((256, 256), Image.BILINEAR)
                wrist_resized = wrist_img.resize((256, 256), Image.BILINEAR)
                img_resized_np = np.array(img_resized)
                wrist_resized_np = np.array(wrist_resized)

                dataset.add_frame({
                    "image": img_resized_np,
                    "wrist_image": wrist_resized_np,
                    "state": states[i],
                    "actions": actions[i],
                    "task": task if isinstance(task, str) else task.decode(),
                })
            
            dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            repo_id=REPO_NAME, 
            #TODO: customize your dataset card
            tags=["TAG_one", "UR5", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
            repo_type="dataset",
        )

if __name__ == "__main__":
    tyro.cli(main)

