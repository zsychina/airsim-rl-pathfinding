import airsim
import numpy as np

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
        
        observation = np.concatenate((distance_vec, cur_loc, distance_sensors, distance))
        return observation
        
    
    def step(self, action):
        self.step_count += 1
        # action := tensor([X, Y, Z])
        self.client.moveByVelocityBodyFrameAsync(vx=action[0], vy=action[1], vz=action[2], 
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
        reward += 100/ distance
        
        distance_sensors = observation[6: 9]
        reward += -1/ distance_sensors.min()
        
        state = self.client.getMultirotorState()
        is_crash = state.collision.has_collided
        if is_crash:
            reward += -1000
            
        return reward
        
        
    def _is_done(self, observation, reward):
        '''
        too long
        too close
        '''
        cur_loc = observation[3: 6]
        target = np.array([self.target_loc.x_val, self.target_loc.y_val, self.target_loc.z_val])
        distance = np.linalg.norm(target - cur_loc)
        if distance < 5 or reward < -1000:
            return True
        
        return False        
        
        
    
if __name__ == '__main__':
    env = Env()
    print(env.target_loc)
    print(env.target_loc.x_val)
    
    