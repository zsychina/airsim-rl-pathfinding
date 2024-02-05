import airsim
import numpy as np
import time
import config

clockspeed = 10
timeslice = 1 / clockspeed
floorZ = 2
speed_limit = 0.2

d_v = d_vx = d_vy = d_vz = 2.0

target_pos = [30, -35, -6]
init_pos = [-65, 0, -10]

class Env:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        # should be random
        self.target_loc = airsim.Vector3r(target_pos[0], target_pos[1], target_pos[2])
        self.step_count = 0
    
    def reset(self):
        self.client.reset()
        self.step_count = 0
        # should be random
        self.target_loc = airsim.Vector3r(target_pos[0], target_pos[1], target_pos[2])
        self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(init_pos[0], init_pos[1], init_pos[2])), False)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        self.last_distance = np.linalg.norm([self.target_loc.x_val - init_pos[0], self.target_loc.y_val - init_pos[1]])
        
        return self._cal_observation()
        
        
    def _cal_observation(self):
        # [DX, DY, DZ, X, Y, Z, Front, Left, Right, Distance]
        state = self.client.getMultirotorState()
        cur_loc = state.kinematics_estimated.position
        cur_loc = np.array([cur_loc.x_val, cur_loc.y_val, cur_loc.z_val])
        target = np.array([self.target_loc.x_val, self.target_loc.y_val, self.target_loc.z_val])
        distance_vec = target - cur_loc
        distance = np.linalg.norm(distance_vec)
        distance_sensor_front = self.client.getDistanceSensorData(distance_sensor_name='Front')
        distance_sensor_left = self.client.getDistanceSensorData(distance_sensor_name='Left')
        distance_sensor_right = self.client.getDistanceSensorData(distance_sensor_name='Right')
        distance_sensors = np.array([distance_sensor_front.distance, distance_sensor_left.distance, distance_sensor_right.distance])
        
        observation = np.concatenate((distance_vec, cur_loc, distance_sensors, np.array([distance])))
        return observation
        
    
    def step(self, action):
        self.step_count += 1
        # action := int \in [0, 3]
        # [+X, -X, +Y, -Y]
        velocity = [0, 0]
        
        try:
            if action in [0, 2]: # add velocity
                velocity[action // 2] = d_v
            else: # minus velocity
                velocity[action // 2] = -d_v
        except:
            print(action)
        
        self.client.simPause(False)
        self.client.moveByVelocityBodyFrameAsync(vx=velocity[0], vy=velocity[1], vz=0, 
                                                 duration=timeslice, yaw_mode=airsim.YawMode(True))
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
            landed = landed or quad_pos.z_val > floorZ 
            collision = collided or landed
            if collision:
                collision_count += 1
                print(f'collision_count {collision_count}')
            if collision_count > 10:
                has_collided = True
                print('collided!')
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
        
        cur_loc = observation[3: 5]
        target = np.array([self.target_loc.x_val, self.target_loc.y_val])
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
            reward += config.reward['backward']
        
        if speed < speed_limit:
            reward += config.reward['slow']
        
        if current_distance <= 5:
            reward += config.reward['goal']
            
        self.last_distance = current_distance
        return reward
        
        
    def _is_done(self, observation, reward):
        cur_loc = observation[3: 6]
        target = np.array([self.target_loc.x_val, self.target_loc.y_val, self.target_loc.z_val])
        distance = np.linalg.norm(target - cur_loc)
        if distance < 3 or reward < -5 or self.step_count > 1500 or np.absolute(observation[5]) < 2:
            return True
        
        return False        

