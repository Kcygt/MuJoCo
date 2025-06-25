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
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Xbox Controller Game")

# Square properties
square_size = 50
x, y = WIDTH // 2, HEIGHT // 2
color = (255, 0, 0)  # Start red
speed = 5

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

# import pygame

# # Initialize Pygame
# pygame.init()

# # Initialize joystick module
# pygame.joystick.init()

# # Check if any joystick is connected
# if pygame.joystick.get_count() == 0:
#     print("No joystick connected.")
#     exit()

# # Use the first joystick
# joystick = pygame.joystick.Joystick(0)
# joystick.init()
# print(f"Joystick name: {joystick.get_name()}")

# # Create a window (needed for event processing)
# screen = pygame.display.set_mode((400, 300))
# pygame.display.set_caption("Xbox Controller Test")

# running = True
# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#         # Joystick button pressed
#         elif event.type == pygame.JOYBUTTONDOWN:
#             print(f"Button {event.button} pressed")

#         # Joystick button released
#         elif event.type == pygame.JOYBUTTONUP:
#             print(f"Button {event.button} released")

#         # Joystick axis motion (for joystick movement)
#         elif event.type == pygame.JOYAXISMOTION:
#             axis = event.axis
#             value = event.value
#             # Axis 0 and 1 usually correspond to left joystick X and Y
#             # Axis 2 and 3 usually correspond to right joystick X and Y
#             print(f"Axis {axis} moved to {value:.2f}")

#             # For example, detect if left joystick is pushed significantly in any direction
#             if axis == 0 and abs(value) > 0.5:
#                 print("Left joystick moved horizontally")
#             if axis == 1 and abs(value) > 0.5:
#                 print("Left joystick moved vertically")

# pygame.quit()
