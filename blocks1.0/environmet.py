import airsim
import numpy as np

d_v = d_vx = d_vy = d_vz = 5.0
class Env:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        # should be random
        self.target_loc = airsim.Vector3r(100, 0, -5)
        self.step_count = 0
    
    def reset(self):
        self.client.reset()
        self.step_count = 0
        # should be random
        self.target_loc = airsim.Vector3r(100, 0, -5)
        self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, -10)), False)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
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
        # action := int \in [0, 5]
        # [+X, -X, +Y, -Y, +Z, -Z]
        velocity = [0, 0, 0]
        
        if action in [0, 2, 4]: # add velocity
            velocity[action // 2] = d_v
        else: # minus velocity
            velocity[action // 2] = -d_v
        
        self.client.moveByVelocityBodyFrameAsync(vx=velocity[0], vy=velocity[1], vz=velocity[2], 
                                                 duration=0.02, yaw_mode=airsim.YawMode(True))
        observation = self._cal_observation()
        reward = self._cal_reward(observation)
        done = self._is_done(observation, reward)
        
        return observation, reward, done


    def _cal_reward(self, observation):
        '''
        + -1000 if crashed
        + 100/ distance
        + -1/ distance_sensor_min
        '''
        reward = 0
        cur_loc = observation[3: 6]
        target = np.array([self.target_loc.x_val, self.target_loc.y_val, self.target_loc.z_val])
        distance = np.linalg.norm(target - cur_loc)
        reward += 10000/distance
        
        distance_sensors = observation[6: 9]
        reward += -1/distance_sensors.min()
        
        state = self.client.getMultirotorState()
        is_crashed = state.collision.has_collided
        is_landed = state.landed_state
        if is_crashed:
            reward += -100000000
            
        return reward
        
        
    def _is_done(self, observation, reward):
        '''
        too long
        too close
        '''
        cur_loc = observation[3: 6]
        target = np.array([self.target_loc.x_val, self.target_loc.y_val, self.target_loc.z_val])
        distance = np.linalg.norm(target - cur_loc)
        if distance < 5 or reward < -1000 or self.step_count > 2000:
            return True
        
        return False        
        
        
    
if __name__ == '__main__':
    env = Env()
    print(env.target_loc)
    print(env.target_loc.x_val)
    
    