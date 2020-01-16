import habitat

import cv2
import numpy as np
import matplotlib.pyplot as plt

from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
#from habitat.datas
import quaternion as q
import os
from cube_projection import Cube, generate_projection_matrix, draw_cube, get_cube_depth_and_faces
from skimage import transform as sktransform
from copy import copy
from habitat.sims.habitat_simulator.actions import HabitatSimActions

class VisualTargetNavEnv(habitat.RLEnv):
    def __init__(self, config):
        super(VisualTargetNavEnv,self).__init__(config)
        self.cube_size = 0.2
        self.cube_resolution = 256
        self.use_texture = True if self.load_texture('wood.jpg', self.cube_resolution) else False
        self.collision_penalty = 0.0
        self.detect_threshold = 1000
        self.distance_threshold = 0.5
        self.success_reward = 1.0
        self.living_penalty = 0.1
        self.cfg = config

        self.debug_mode = False

    def load_texture(self, texture_path, size):
        try :
            wood_texture = plt.imread(texture_path)
            self.texture_image = cv2.resize(wood_texture[:, 42:208], (size, size))
            self.texture_points = np.array([[0., 0., size, size], [size, 0., 0., size]]).T
            return True
        except FileNotFoundError:
            print("Image of Wood box doesn't exists!")
            return False

    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, detected, success):
        collided = self._env.sim.previous_step_collided
        reward = collided * self.collision_penalty# + self._env.sim.geodesic_distance()
        reward += success * self.success_reward
        reward -= self.living_penalty
        return reward

    def step(self,action):
        obs = self.habitat_env.step(action)
        self.depth = obs['depth'].squeeze()
        self.depth[self.depth == 0] = np.inf
        goal_pose = np.array(self.habitat_env.current_episode.goals[0].position)[[0,2,1]]
        obs['rgb'], detected = self._add_cube_into_image(obs, goal_pose)
        done, success = self.get_done(detected)
        reward = self.get_reward(detected, success)
        info = self.habitat_env.get_metrics()
        return obs, reward, done, info

    def _add_cube_into_image(self, obs, location, color=[20, 200, 200]):
        cube = Cube(origin=location, scale=self.cube_size)
        self.cube_image = final = copy(obs["rgb"])
        roll, yaw, pitch = q.as_euler_angles(self.habitat_env.sim.get_agent_state().rotation)
        x,z,y = self.habitat_env.sim.get_agent_state().position
        if self.debug_mode :
            print('goal_pose : {}, agent_pose: {}'.format(location, [x,y,z]))
        size = self.cube_resolution // 2
        fov = 1.57#self.cfg.SIMULATOR.RGB_SENSOR.HFOV
        world_to_image_mat = generate_projection_matrix(x,y,z, -obs['heading']-np.pi/2, pitch, roll, fov, fov, size, size)
        detected = 0
        if self.use_texture:
            masks, xx_faces, yy_faces = get_cube_depth_and_faces(cube, world_to_image_mat, size * 2, size * 2)
            for face_mask, xx, yy in zip(masks, xx_faces, yy_faces):
                transform = sktransform.ProjectiveTransform()
                dest = np.array([yy, xx]).T
                try:
                    transform.estimate(self.texture_points, dest)
                    self.new_img = sktransform.warp(self.texture_image, transform.inverse)
                except:
                    continue
                img_mask = face_mask < self.depth
                self.cube_image[img_mask] = np.array(self.new_img[img_mask] * 255, dtype=np.uint8)
                self.depth[img_mask] = face_mask[img_mask]
                detected += len(np.where(img_mask)[0])
        else:
            cube_idx = draw_cube(cube, world_to_image_mat, size * 2, size * 2, fast_depth=True)
            self.cube_image[cube_idx[0] < self.depth] = np.array(color)
        return self.cube_image, detected

    def get_done(self, detected):
        time_over = self.habitat_env.episode_over
        collision = self.habitat_env.sim.previous_step_collided
        success = self.is_success(detected)
        done = time_over or collision or success
        if self.debug_mode:
            if time_over : print('Time over!')
            if collision : print('Collision!')
            if success : print('Success!')
        return done, success

    def is_success(self, detected):
        agent_x, _, agent_y = self.habitat_env.sim.get_agent_state().position
        target_x, _, target_y = self.habitat_env.current_episode.goals[0].position
        distance = np.power((agent_x - target_x) ** 2 + (agent_y - target_y) ** 2, 0.5)
        if self.debug_mode :
            print('dist {} detect {}'.format(distance, detected))
        close_enough = distance < self.distance_threshold
        detect = detected > self.detect_threshold
        return close_enough and detect


if __name__ == '__main__':
    cfg = habitat.get_config('vistargetnav_mp3d.yaml')
    cfg.defrost()
    cfg.DATASET.DATA_PATH = '../habitat-api/data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz'
    cfg.DATASET.SCENES_DIR = '../habitat-api/data/scene_datasets'
    cfg.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    cfg.freeze()
    print(cfg)

    env = VisualTargetNavEnv(cfg)
    obs = env.reset()
    follower = ShortestPathFollower(env.habitat_env.sim, 0.4, False)

    from habitat.utils.visualizations import maps
    def draw_top_down_map(info, heading, output_size):
        top_down_map = maps.colorize_topdown_map(
            info["top_down_map"]["map"], info["top_down_map"]["fog_of_war_mask"]
        )
        original_map_size = top_down_map.shape[:2]
        map_scale = np.array(
            (1, original_map_size[1] * 1.0 / original_map_size[0])
        )
        new_map_size = np.round(output_size * map_scale).astype(np.int32)
        # OpenCV expects w, h but map size is in h, w
        top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

        map_agent_pos = info["top_down_map"]["agent_map_coord"]
        map_agent_pos = np.round(
            map_agent_pos * new_map_size / original_map_size
        ).astype(np.int32)
        top_down_map = maps.draw_agent(
            top_down_map,
            map_agent_pos,
            heading - np.pi / 2,
            agent_radius_px=top_down_map.shape[0] / 40,
        )
        return top_down_map


    for episode in range(100):
        env.reset()
        images = []
        end = False
        action =  1#None
        while not env.habitat_env.episode_over:
            if action is not None :
                best_action = action
            observations, reward, done, info = env.step(best_action)
            print('reward {} , done {}'.format(reward, done))
            if done : break
            depth = np.concatenate([np.expand_dims(env.depth,2)]*3, 2)

            depth[np.where(depth == np.inf)] = np.nanmax(depth[depth != np.inf])
            im = observations["rgb"]
            top_down_map = draw_top_down_map(info, observations['heading'], 256)
            view_img = np.concatenate([im/255., depth/np.nanmax(depth), top_down_map/255.0],1)
            cv2.imshow('hi',view_img[:,:,[2,1,0]])
            key = cv2.waitKey(0)
            action = None
            if key == ord('q') :
                end = True
                break
            elif key == ord('p'):
                end = False
                break
            elif key == ord('w'): action = 'MOVE_FORWARD'
            elif key == ord('a'): action = 'TURN_LEFT'
            elif key == ord('d'): action = 'TURN_RIGHT'
        if end : break
