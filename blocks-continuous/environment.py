import airsim
import numpy as np
import time
import config
import cv2

clockspeed = 10
timeslice = 1 / clockspeed
speed_limit = 0.2

d_v = d_vx = d_vy = d_vz = 10.0
yaw_dv = 5.0

class Env:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.image_error_count = 0
    
    def reset(self):
        self.client.reset()
        self.step_count = 0
        target_pos = [np.random.uniform(34, 38), np.random.uniform(30, 47), -10]
        init_pos = [np.random.uniform(-95, -90), np.random.uniform(-20, 20), -10]
        # print(f'target_pos {target_pos}\ninit_pos{init_pos}')
        self.target_loc = airsim.Vector3r(target_pos[0], target_pos[1], target_pos[2])
        self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(init_pos[0], init_pos[1], init_pos[2])), False)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        self.last_distance = np.linalg.norm([self.target_loc.x_val - init_pos[0], self.target_loc.y_val - init_pos[1], self.target_loc.z_val - init_pos[2]])
        
        return self._cal_observation()
        
        
    def _cal_observation(self):
        state = self.client.getMultirotorState()
        cur_loc = state.kinematics_estimated.position
        cur_loc = np.array([cur_loc.x_val, cur_loc.y_val, cur_loc.z_val])
        target = np.array([self.target_loc.x_val, self.target_loc.y_val, self.target_loc.z_val])
        try:
            responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
            response = responses[0]
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
            img_rgb = img1d.reshape(response.height, response.width, 3)
            img_resize = cv2.resize(img_rgb, (64, 64))
            img_resize = np.transpose(img_resize, (2, 0, 1))
        except:
            img_resize = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
            self.image_error_count += 1
            print(f'image error {self.image_error_count}')
        observation = [img_resize, np.concatenate([cur_loc, target])]
        return observation
        
    
    def step(self, action):
        self.step_count += 1

        vx = action[0] * d_vx
        vy = action[1] * d_vy
        vz = action[2] * d_vz
        yaw_speed = action[3] * yaw_dv

        self.client.simPause(False)
        self.client.moveByVelocityBodyFrameAsync(vx=vx, vy=vy, vz=vz, 
                                                 duration=timeslice, yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_speed))
        landed = False
        has_collided = False
        collision_count = 0
        start_time = time.time()
        while time.time() - start_time < timeslice: # ~= sleep(timeslice)
            quad_pos = self.client.getMultirotorState().kinematics_estimated.position
            quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
            
            collided = self.client.simGetCollisionInfo().has_collided
            
            # alternative solution for simGetCollisionInfo().has_collided api always return false
            close_to_wall = False
            distance_sensor_front = self.client.getDistanceSensorData(distance_sensor_name='Front')
            distance_sensor_left = self.client.getDistanceSensorData(distance_sensor_name='Left')
            distance_sensor_right = self.client.getDistanceSensorData(distance_sensor_name='Right')            
            distance_sensor_back = self.client.getDistanceSensorData(distance_sensor_name='Back')    
            distance_sensors = np.array([distance_sensor_front.distance, distance_sensor_left.distance, distance_sensor_right.distance, distance_sensor_back.distance])
            if distance_sensors.min() < 0.1:
                close_to_wall = True
            collided = collided or close_to_wall
            
            landed = (quad_vel.x_val == 0 and quad_vel.y_val == 0 and quad_vel.z_val == 0)
            landed = landed or np.absolute(quad_pos.z_val) < 2 
            collision = collided or landed
            if collision:
                has_collided = True                
                print(f'collision_count {collision_count}')
                break
         
        self.client.simPause(True)
         
        dead = has_collided    
        observation = self._cal_observation()
        reward = self._cal_reward(observation, dead)
        done = self._is_done(observation, reward) or dead
        
        return observation, reward, done

    
    def _cal_reward(self, observation, dead):
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val], dtype=np.float64)
        speed = np.linalg.norm(vel)
        
        cur_loc = observation[1][:3]
        target = np.array([self.target_loc.x_val, self.target_loc.y_val, self.target_loc.z_val])
        current_distance = np.linalg.norm(target - cur_loc)
        
        distance_sensor_front = self.client.getDistanceSensorData(distance_sensor_name='Front')
        distance_sensor_left = self.client.getDistanceSensorData(distance_sensor_name='Left')
        distance_sensor_right = self.client.getDistanceSensorData(distance_sensor_name='Right')            
        distance_sensor_back = self.client.getDistanceSensorData(distance_sensor_name='Back')    
        distance_sensors = np.array([distance_sensor_front.distance, distance_sensor_left.distance, distance_sensor_right.distance, distance_sensor_back.distance])       
        
        
        reward = 0
        # reward += config.reward['close'] / (distance_sensors.min() + 1e-6)
        
        if dead:
            reward += config.reward['dead']
        
        if current_distance < self.last_distance:
            reward += config.reward['forward'] * (self.last_distance - current_distance)
        else:
            reward += config.reward['backward'] * (current_distance - self.last_distance)
        
        if speed < speed_limit:
            reward += config.reward['slow']
        
        if current_distance <= 10:
            reward += config.reward['goal'] / current_distance
            
        self.last_distance = current_distance
        return reward
        
        
    def _is_done(self, observation, reward):
        cur_loc = observation[1][:3]
        target = np.array([self.target_loc.x_val, self.target_loc.y_val, self.target_loc.z_val])
        distance = np.linalg.norm(target - cur_loc)

        if distance <= 5:
            print('solved')
            return True
        if reward < -30:
            print('reward too low') 
            return True
        if self.step_count > 2000:
            print('timeout')   
            return True
        if np.absolute(observation[1][2]) < 1:
            print('too low')
            return True
          
        return False        

