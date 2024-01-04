import pygame

pygame.init()
display = pygame.display.set_mode((300, 300))

while True:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            print(pygame.key.name(event.key))
            if event.key == pygame.K_ESCAPE:
                quit()
