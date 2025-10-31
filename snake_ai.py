import random
import sys

GAME_WIDTH = 700
GAME_HEIGHT = 500
SPACE_SIZE = 25
BODY_PARTS = 3

class Snake:
    def __init__(self):
        self.body_size = BODY_PARTS
        self.coordinates = []

        for i in range(0, BODY_PARTS):
            self.coordinates.append([0, 0])


class Food:
    def __init__(self):
        x = random.randint(0, (GAME_WIDTH//SPACE_SIZE)-1) * SPACE_SIZE
        y = random.randint(0, (GAME_HEIGHT//SPACE_SIZE)-1) * SPACE_SIZE

        self.coordinates = [x, y]


def next_turn(snake, food):
    x, y = snake.coordinates[0]
    
    if direction == "up":
        y -= SPACE_SIZE

    elif direction == "down":
        y += SPACE_SIZE

    elif direction == "left":
        x -= SPACE_SIZE

    elif direction == "right":
        x += SPACE_SIZE
    
    snake.coordinates.insert(0, (x, y))

    if x == food.coordinates[0] and y == food.coordinates[1]:
        global score
        score += 1

        food = Food()
    
    else:
        del snake.coordinates[-1]
        
    if check_collisions(snake):
        game_over()

def change_direction(new_direction, snake):
    global direction

    head_x, head_y = snake.coordinates[0]
    neck_x, neck_y = snake.coordinates[1]

    if new_direction == 'left':
        new_x, new_y = head_x - SPACE_SIZE, head_y

    elif new_direction == 'right':
        new_x, new_y = head_x + SPACE_SIZE, head_y
    
    elif new_direction == 'up':
        new_x, new_y = head_x, head_y - SPACE_SIZE

    elif new_direction == 'down':
        new_x, new_y = head_x, head_y + SPACE_SIZE

    if (new_x, new_y) != (neck_x, neck_y):
        direction = new_direction


def check_collisions(snake):
    x, y = snake.coordinates[0]
    
    if x < 0 or x >= GAME_WIDTH:
        return True
    
    elif y < 0 or y >= GAME_HEIGHT:
        return True
    
    for body_part in snake.coordinates[1:]:
        if x == body_part[0] and y == body_part[1]:
            return True
    
    return False

def game_over():
    sys.exit(0)