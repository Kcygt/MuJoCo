import pygame
import random

# Initialize Pygame
pygame.init()

# Set up joystick
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    print("No joystick connected.")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()

# Window setup
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Xbox Controller Game")

# Square properties
square_size = 50
x, y = WIDTH // 2, HEIGHT // 2
color = (255, 0, 0)  # Start red
speed = 50

clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Button events
        elif event.type == pygame.JOYBUTTONDOWN:
            if event.button == 0:  # A button
                # Change square color randomly
                color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
            elif event.button == 1:  # B button
                running = False

    # Get joystick axes for movement
    axis_x = joystick.get_axis(0)  # Left stick horizontal
    axis_y = joystick.get_axis(1)  # Left stick vertical

    # Deadzone to prevent drift
    deadzone = 0.1
    if abs(axis_x) < deadzone:
        axis_x = 0
    if abs(axis_y) < deadzone:
        axis_y = 0

    # Move the square
    x += int(axis_x * speed)
    y += int(axis_y * speed)

    # Keep square inside window
    x = max(0, min(WIDTH - square_size, x))
    y = max(0, min(HEIGHT - square_size, y))

    # Drawing
    screen.fill((30, 30, 30))  # Dark background
    pygame.draw.rect(screen, color, (x, y, square_size, square_size))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
