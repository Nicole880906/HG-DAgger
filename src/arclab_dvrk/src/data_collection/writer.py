import os
import cv2
import json
import datetime
import numpy as np
import time
# from .rerun_visualizer import RerunLogger
from queue import Queue, Empty
from threading import Thread

class EpisodeWriter():
    def __init__(self, task_dir, frequency=30, image_size=[640, 480], rerun_log = True):
        """
        image_size: [width, height]
        """
        print("==> EpisodeWriter initializing...\n")
        self.task_dir = task_dir
        self.frequency = frequency
        self.image_size = image_size

        self.rerun_log = rerun_log
        
        self.data = {}
        self.episode_data = []
        self.item_id = -1
        self.episode_id = -1
        if os.path.exists(self.task_dir):
            episode_dirs = [episode_dir for episode_dir in os.listdir(self.task_dir) if 'episode_' in episode_dir]
            episode_last = sorted(episode_dirs)[-1] if len(episode_dirs) > 0 else None
            self.episode_id = 0 if episode_last is None else int(episode_last.split('_')[-1])
            print(f"==> task_dir directory already exist, now self.episode_id is:{self.episode_id}\n")
        else:
            os.makedirs(self.task_dir)
            print(f"==> episode directory does not exist, now create one.\n")
        self.data_info()
        self.text_desc()

        self.is_available = True  # Indicates whether the class is available for new operations
        # Initialize the queue and worker thread
        self.item_data_queue = Queue(maxsize=100)
        self.stop_worker = False
        self.need_save = False  # Flag to indicate when save_episode is triggered
        # self.worker_thread = Thread(target=self.process_queue)
        # self.worker_thread.start()

        print("==> EpisodeWriter initialized successfully.\n")

    def data_info(self, version='1.0.0', date=None, author=None):
        self.info = {
                # "version": "1.0.0" if version is None else version, 
                "date": datetime.date.today().strftime('%Y-%m-%d') if date is None else date,
                "author": "cutting" if author is None else author,
                "image": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "depth": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "pointcloud": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "joint_names":{
                    "psm1_joint_state":   ['psm1_yaw_joint', 'psm1_pitch_end_joint', 'psm1_main_insertion_joint', 'psm1_tool_roll_joint', 'psm1_tool_pitch_joint', 'psm1_tool_yaw_joint', 'psm1_tool_gripper1_joint', 'psm1_tool_gripper2_joint'],
                    "psm2_joint_state":  ['psm2_yaw_joint', 'psm2_pitch_end_joint', 'psm2_main_insertion_joint', 'psm2_tool_roll_joint', 'psm2_tool_pitch_joint', 'psm2_tool_yaw_joint', 'psm2_tool_gripper1_joint', 'psm2_tool_gripper2_joint'],
                    "psm1_ee": ['PSM1PosX', 'PSM1PosY', 'PSM1PosZ', 'PSM1QuatW', 'PSM1QuatX', 'PSM1QuatY', 'PSM1QuatZ'],
                    "psm2_ee": ['PSM2PosX', 'PSM2PosY', 'PSM2PosZ', 'PSM2QuatW', 'PSM2QuatX', 'PSM2QuatY', 'PSM2QuatZ'],
                },

            }
    def text_desc(self):
        self.text = {
            "goal": "Retraction and cutting",
        }

 
    def create_episode(self):
        """
        Create a new episode.
        Returns:
            bool: True if the episode is successfully created, False otherwise.
        Note:
            Once successfully created, this function will only be available again after save_episode complete its save task.
        """
        if not self.is_available:
            print("==> The class is currently unavailable for new operations. Please wait until ongoing tasks are completed.")
            return False  # Return False if the class is unavailable

        # Reset episode-related data and create necessary directories
        self.item_id = -1
        self.episode_data = []
        self.episode_id = self.episode_id + 1
        
        self.episode_dir = os.path.join(self.task_dir, f"episode_{str(self.episode_id).zfill(4)}")
        self.color_dir = os.path.join(self.episode_dir, 'colors')
        self.pc_dir = os.path.join(self.episode_dir, 'pointclouds')
        self.audio_dir = os.path.join(self.episode_dir, 'audios')
        self.json_path = os.path.join(self.episode_dir, 'data.json')
        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(self.color_dir, exist_ok=True)

        # if self.rerun_log:
        #     self.online_logger = RerunLogger(prefix="online/", IdxRangeBoundary = 60, memory_limit="300MB")

        self.is_available = False  # After the episode is created, the class is marked as unavailable until the episode is successfully saved
        print(f"==> New episode created: {self.episode_dir}")
        return True  # Return True if the episode is successfully created
        
    def add_item(self, colors=None, states=None,):
        # Increment the item ID
        self.item_id += 1
        # Create the item data dictionary
        item_data = {
            'idx': self.item_id,
            'colors': colors,
            'states': states,
        }
        # Enqueue the item data
        self._process_item_data(item_data)
        # self.item_data_queue.put(item_data)

    # def process_queue(self):
    #     while not self.stop_worker or not self.item_data_queue.empty():
    #         # Process items in the queue
    #         try:
    #             item_data = self.item_data_queue.get(timeout=1)
    #             try:
    #                 self._process_item_data(item_data)
    #             except Exception as e:
    #                 print(f"Error processing item_data (idx={item_data['idx']}): {e}")
    #             self.item_data_queue.task_done()
    #         except Empty:
    #             pass
        
    #         # Check if save_episode was triggered
    #         if self.need_save and self.item_data_queue.empty():
    #             self._save_episode()

    def _process_item_data(self, item_data):
        idx = item_data['idx']
        colors = item_data.get('colors', {})

        # Save images
        if colors:
            for idx_color, (color_key, color) in enumerate(colors.items()):
                color_name = f'{color_key}_{str(idx).zfill(6)}.jpg'
                if not cv2.imwrite(os.path.join(self.color_dir, color_name), color):
                    print(f"Failed to save color image.")
                item_data['colors'][color_key] = os.path.join(
                    'colors', color_name)

        # Update episode data
        self.episode_data.append(item_data)

    # def save_episode(self):
    #     """
    #     Trigger the save operation. This sets the save flag, and the process_queue thread will handle it.
    #     """
    #     self.need_save = True  # Set the save flag
    #     print(f"==> Episode saved start...")

    def save_episode(self):
        """
        Save the episode data to a JSON file.
        """
        self.data['info'] = self.info
        self.data['text'] = self.text
        self.data['data'] = self.episode_data
        with open(self.json_path, 'w', encoding='utf-8') as jsonf:
            jsonf.write(json.dumps(self.data, indent=4, ensure_ascii=False))
        self.need_save = False     # Reset the save flag
        self.is_available = True   # Mark the class as available after saving
        print(f"==> Episode saved successfully to {self.json_path}.")

    def close(self):
        """
        Stop the worker thread and ensure all tasks are completed.
        """
        if not self.is_available:  # If self.is_available is False, it means there is still data not saved.
            self.save_episode()
        while not self.is_available:
            time.sleep(0.01)
        self.stop_worker = True
