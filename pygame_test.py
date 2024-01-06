import pygame
import sys
import time

pygame.init()
display = pygame.display.set_mode((300, 300))

while True:
    time.sleep(0.02)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
    
    scan_wrapper = pygame.key.get_pressed()
    
    if scan_wrapper[pygame.K_SPACE]:
        print('space')
