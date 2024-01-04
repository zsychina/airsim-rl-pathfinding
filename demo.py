import airsim
import pygame
import time

pygame.init()
display = pygame.display.set_mode((300, 300))

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Take off the drone
client.takeoffAsync().join()

# Set the drone to hover in place
client.hoverAsync().join()

# Define the velocity for movements
velocity = 5 # m/s

while True:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            print(pygame.key.name(event.key))
            # press ESC to end program
            if event.key == pygame.K_ESCAPE:
                client.armDisarm(False)
                client.enableApiControl(False)
                pygame.quit()
                quit()
            
            # Increase altitude
            if event.key == pygame.K_w:
                client.moveByVelocityZAsync(0, 0, -velocity, 1).join()
                
            # Decrease altitude
            if event.key == pygame.K_s:
                client.moveByVelocityZAsync(0, 0, velocity, 1).join()
            
            # Move forward
            if event.key == pygame.K_UP:
                client.moveByVelocityAsync(velocity, 0, 0, 1).join()
            
            # Move backward
            if event.key == pygame.K_DOWN:
                client.moveByVelocityAsync(-velocity, 0, 0, 1).join()
            
            # Move left
            if event.key == pygame.K_LEFT:
                client.moveByVelocityAsync(0, -velocity, 0, 1).join()
            
            # Move right
            if event.key == pygame.K_RIGHT:
                client.moveByVelocityAsync(0, velocity, 0, 1).join()
            
            # TODO: Add more control options if needed
                
client.armDisarm(False)
client.enableApiControl(False)
pygame.quit()