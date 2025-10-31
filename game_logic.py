import random
import sys

GAME_WIDTH = 700
"""
Módulo de lógica pura (headless) para o jogo Snake.
Contém classes: Snake, Food, Game
Nenhuma dependência de tkinter aqui.
Retorna snapshots de estado que podem ser usados por uma UI ou por um runner headless.
"""
import random
from typing import List, Tuple, Dict, Optional

GAME_WIDTH = 700
GAME_HEIGHT = 500
SPACE_SIZE = 25
BODY_PARTS = 3

Direction = str  # 'up'|'down'|'left'|'right'


class Snake:
    def __init__(self, initial_length: int = BODY_PARTS, start_at: Optional[Tuple[int,int]] = None):
        # armazenamos uma lista de tuplas (x,y)
        if start_at is None:
            start_at = (0, 0)
        self.coordinates: List[Tuple[int,int]] = [start_at for _ in range(initial_length)]

    def head(self) -> Tuple[int,int]:
        return self.coordinates[0]

    def grow_head(self, pos: Tuple[int,int]):
        self.coordinates.insert(0, pos)

    def pop_tail(self):
        if self.coordinates:
            self.coordinates.pop()

    def length(self) -> int:
        return len(self.coordinates)


class Food:
    def __init__(self, coordinates: Tuple[int,int]):
        self.coordinates = coordinates


class Game:
    """Classe principal da lógica do jogo.

    Métodos principais:
      - change_direction(new_direction)
      - step() -> Dict snapshot do estado após o passo
      - get_state() -> Dict snapshot sem avançar
    """

    def __init__(self, width: int = GAME_WIDTH, height: int = GAME_HEIGHT, space: int = SPACE_SIZE, initial_length: int = BODY_PARTS):
        self.width = width
        self.height = height
        self.space = space
        self.score = 0
        self.direction: Direction = 'down'
        # iniciar a cobra no centro (opcional)
        start_x = (width // (2*space)) * space
        start_y = (height // (2*space)) * space
        self.snake = Snake(initial_length, start_at=(start_x, start_y))
        self.food = self._spawn_food()
        self.game_over = False
        self.reason: Optional[str] = None

    def _all_cells(self):
        xs = range(0, self.width, self.space)
        ys = range(0, self.height, self.space)
        for x in xs:
            for y in ys:
                yield (x, y)

    def _spawn_food(self) -> Food:
        # Gera comida em célula livre; se não houver espaço, marca game_over
        occupied = set(self.snake.coordinates)
        all_cells = [(x, y) for x in range(0, self.width, self.space) for y in range(0, self.height, self.space)]
        free = [c for c in all_cells if c not in occupied]
        if not free:
            # tabuleiro cheio -> vitória/termina
            self.game_over = True
            self.reason = 'board_full'
            return Food((-1, -1))
        coord = random.choice(free)
        return Food(coord)

    def change_direction(self, new_direction: Direction) -> bool:
        """Atualiza direção evitando reverso direto. Retorna True se aceito."""
        opposites = {'up':'down', 'down':'up', 'left':'right', 'right':'left'}
        if new_direction not in opposites:
            return False
        if opposites[new_direction] == self.direction:
            return False
        self.direction = new_direction
        return True

    def _compute_next_head(self) -> Tuple[int,int]:
        x, y = self.snake.head()
        if self.direction == 'up':
            y -= self.space
        elif self.direction == 'down':
            y += self.space
        elif self.direction == 'left':
            x -= self.space
        elif self.direction == 'right':
            x += self.space
        return (x, y)

    def _check_collisions(self) -> bool:
        x, y = self.snake.head()
        # bordas
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            self.reason = 'wall'
            return True
        # corpo
        for part in self.snake.coordinates[1:]:
            if (x, y) == part:
                self.reason = 'self'
                return True
        return False

    def step(self) -> Dict:
        """Avança um turno e retorna um snapshot do estado.

        Snapshot contém chaves: status ('moved'|'ate'|'game_over'),
        'snake': list of [x,y], 'food': [x,y], 'score': int
        """
        if self.game_over:
            return self.get_state()

        next_head = self._compute_next_head()
        # inserir nova cabeça
        self.snake.grow_head(next_head)

        status = 'moved'
        # comeu?
        if next_head == tuple(self.food.coordinates):
            self.score += 1
            status = 'ate'
            self.food = self._spawn_food()
        else:
            # remove cauda
            self.snake.pop_tail()

        # checar colisões
        if self._check_collisions():
            self.game_over = True
            status = 'game_over'

        return {
            'status': status,
            'snake': [list(c) for c in self.snake.coordinates],
            'food': list(self.food.coordinates),
            'score': self.score,
            'game_over': self.game_over,
            'reason': self.reason,
        }

    def get_state(self) -> Dict:
        return {
            'status': 'game_over' if self.game_over else 'idle',
            'snake': [list(c) for c in self.snake.coordinates],
            'food': list(self.food.coordinates),
            'score': self.score,
            'game_over': self.game_over,
            'reason': self.reason,
        }

    def reset(self):
        self.score = 0
        self.direction = 'down'
        start_x = (self.width // (2*self.space)) * self.space
        start_y = (self.height // (2*self.space)) * self.space
        self.snake = Snake(BODY_PARTS, start_at=(start_x, start_y))
        self.food = self._spawn_food()
        self.game_over = False
        self.reason = None