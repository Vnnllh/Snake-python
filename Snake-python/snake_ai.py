# import neat
# import os
# import game_logic as game


# def distanceToFood(snake_head, food_position):
#     """Calculate the Manhattan distance from the snake's head to the food."""
#     return abs(snake_head[0] - food_position[0]) + abs(snake_head[1] - food_position[1])

# def distancesToObstacles(snake_head, snake_body, grid_size):
#     """
#     Calculate the distance from the snake's head to the nearest obstacle (wall or snake body)
#     in 8 directions: up, down, left, right, and 4 diagonals.

#     Args:
#         snake_head (tuple): (x, y) position of the snake's head.
#         snake_body (list): List of (x, y) positions occupied by the snake.
#         grid_size (tuple): (width, height) of the grid.

#     Returns:
#         list: Distances to the nearest obstacle in each direction.
#               Order: [up, down, left, right, up-left, up-right, down-left, down-right]
#     """
#     directions = [
#         (0, -1),  # up
#         (0, 1),   # down
#         (-1, 0),  # left
#         (1, 0),   # right
#         (-1, -1), # up-left
#         (1, -1),  # up-right
#         (-1, 1),  # down-left
#         (1, 1),   # down-right
#     ]
#     distances = []
#     for dx, dy in directions:
#         x, y = snake_head
#         distance = 0
#         while True:
#             x += dx
#             y += dy
#             distance += 1
#             if x < 0 or x >= grid_size[0] or y < 0 or y >= grid_size[1]:
#                 break
#             if (x, y) in snake_body:
#                 break
#         distances.append(distance)
#     return distances

# def get_snake_inputs(snake, food, grid_size):
#     """Get the inputs for the neural network based on the snake's state."""
#     snake_head = snake.coordinates[0]
#     snake_body = snake.coordinates[1:]

#     distance_to_food = distanceToFood(snake_head, food.coordinates)
#     distances_to_obstacles = distancesToObstacles(snake_head, snake_body, grid_size)

#     return [distance_to_food] + distances_to_obstacles

# def create_neural_network():
#     local_dir = os.path.dirname(__file__)
#     config_path = os.path.join(local_dir, 'neat_config.txt')

#     config = neat.Config(
#         neat.DefaultGenome,
#         neat.DefaultReproduction,
#         neat.DefaultSpeciesSet,
#         neat.DefaultStagnation,
#         config_path
#     )

#     genome = config.genome_type(1)
#     genome.configure_new(config.genome_config)

#     net = neat.nn.FeedForwardNetwork.create(genome, config)

#     return net

# def test_neural_network():
#     """Testa a rede neural jogando Snake."""
#     print("=== CRIANDO REDE NEURAL ===\n")
    
#     # Cria a rede neural
#     net = create_neural_network()
    
#     # Cria o jogo
#     g = game.Game()
#     grid_size = (g.width // g.space, g.height // g.space)
    
#     print("=== INICIANDO JOGO ===\n")
    
#     # Loop do jogo
#     moves = 0
#     max_moves = 100
    
#     while not g.game_over and moves < max_moves:
#         # Coletar inputs do estado do jogo
#         inputs = get_snake_inputs(g.snake, g.food, grid_size)
        
#         # Passar inputs para a rede neural
#         outputs = net.activate(inputs)
        
#         # Escolher direção com maior output
#         direction_index = outputs.index(max(outputs))
#         directions = ['up', 'down', 'left', 'right']
#         chosen_direction = directions[direction_index]
        
#         # Executar movimento
#         g.change_direction(chosen_direction)
#         state = g.step()
        
#         # Debug
#         print(f"Move {moves + 1}: {chosen_direction} | Score: {state['score']}")
        
#         moves += 1
    
#     print(f"\n=== FIM DO JOGO ===")
#     print(f"Score final: {g.score}")
#     print(f"Motivo: {g.reason if g.game_over else 'Limite de movimentos'}")

# if __name__ == '__main__':
#     test_neural_network()

import neat
import os
import pickle
import game_logic as game
import threading
import time



def distanceToFood(snake_head, food_position):
    """Calculate the Manhattan distance from the snake's head to the food."""
    return abs(snake_head[0] - food_position[0]) + abs(snake_head[1] - food_position[1])


def distancesToObstacles(snake_head, snake_body, grid_size):
    """
    Calculate the distance from the snake's head to the nearest obstacle (wall or snake body)
    in 8 directions: up, down, left, right, and 4 diagonals.

    Args:
        snake_head (tuple): (x, y) position of the snake's head.
        snake_body (list): List of (x, y) positions occupied by the snake.
        grid_size (tuple): (width, height) of the grid.

    Returns:
        list: Distances to the nearest obstacle in each direction.
              Order: [up, down, left, right, up-left, up-right, down-left, down-right]
    """
    directions = [
        (0, -1),  # up
        (0, 1),   # down
        (-1, 0),  # left
        (1, 0),   # right
        (-1, -1), # up-left
        (1, -1),  # up-right
        (-1, 1),  # down-left
        (1, 1),   # down-right
    ]
    distances = []
    for dx, dy in directions:
        x, y = snake_head
        distance = 0
        while True:
            x += dx
            y += dy
            distance += 1
            if x < 0 or x >= grid_size[0] or y < 0 or y >= grid_size[1]:
                break
            if (x, y) in snake_body:
                break
        distances.append(distance)
    return distances


def get_snake_inputs(g):
    """Get the inputs for the neural network based on the game state."""
    snake_head = g.snake.head()
    snake_body = [tuple(coord) for coord in g.snake.coordinates[1:]]
    grid_size = (g.width // g.space, g.height // g.space)

    # 1. Distância à comida (normalizada)
    distance_to_food = distanceToFood(snake_head, tuple(g.food.coordinates))
    max_distance = g.width + g.height
    normalized_distance = distance_to_food / max_distance
    
    # 2. Direção da comida (4 valores binários: up, down, left, right)
    food_x, food_y = g.food.coordinates
    head_x, head_y = snake_head
    
    food_up = 1.0 if food_y < head_y else 0.0
    food_down = 1.0 if food_y > head_y else 0.0
    food_left = 1.0 if food_x < head_x else 0.0
    food_right = 1.0 if food_x > head_x else 0.0
    
    # 3. Distâncias aos obstáculos
    distances_to_obstacles = distancesToObstacles(snake_head, snake_body, grid_size)
    
    # Normalizar distâncias (0 a 1)
    max_grid_distance = max(grid_size)
    normalized_obstacles = [d / max_grid_distance for d in distances_to_obstacles]

    # Total: 1 + 4 + 8 = 13 inputs
    return [normalized_distance, food_up, food_down, food_left, food_right] + normalized_obstacles


# ==================== ESTATÍSTICAS ====================
class TrainingStats:
    """Armazena estatísticas do treinamento para visualização futura."""
    def __init__(self):
        self.generation_best_fitness = []
        self.generation_avg_fitness = []
        self.generation_best_score = []
        self.best_genome_per_generation = []
        self.total_generations = 0
        self.best_overall_fitness = 0
        self.best_overall_score = 0
        self.best_overall_genome = None
    
    def update(self, generation, best_genome, best_fitness, avg_fitness, best_score):
        """Atualiza estatísticas de uma geração."""
        self.generation_best_fitness.append(best_fitness)
        self.generation_avg_fitness.append(avg_fitness)
        self.generation_best_score.append(best_score)
        self.best_genome_per_generation.append(best_genome)
        self.total_generations = generation + 1
        
        if best_fitness > self.best_overall_fitness:
            self.best_overall_fitness = best_fitness
            self.best_overall_genome = best_genome
        
        if best_score > self.best_overall_score:
            self.best_overall_score = best_score
    
    def save(self, filename='training_stats.pickle'):
        """Salva estatísticas em arquivo."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filename='training_stats.pickle'):
        """Carrega estatísticas de arquivo."""
        with open(filename, 'rb') as f:
            return pickle.load(f)


# ==================== AVALIAÇÃO DE GENOMA ====================
def eval_genome(genome, config):
    """
    Avalia um único genoma (uma cobra) jogando Snake.
    Retorna (fitness, score, moves) para estatísticas.
    """
    # Criar a rede neural a partir do genoma
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Criar o jogo
    g = game.Game()
    
    # Variáveis de controle
    fitness = 0
    moves_without_food = 0
    max_moves_without_food = 150  # Aumentado para dar mais tempo
    total_moves = 0
    
    # Distância inicial à comida
    initial_distance = distanceToFood(g.snake.head(), tuple(g.food.coordinates))
    
    # Loop do jogo
    while not g.game_over and moves_without_food < max_moves_without_food:
        # Coletar inputs
        inputs = get_snake_inputs(g)
        
        # Obter decisão da rede neural
        outputs = net.activate(inputs)
        direction_index = outputs.index(max(outputs))
        directions = ['up', 'down', 'left', 'right']
        chosen_direction = directions[direction_index]
        
        # Executar movimento
        g.change_direction(chosen_direction)
        state = g.step()
        
        total_moves += 1
        moves_without_food += 1
        
        # Calcular fitness baseado no movimento
        if state['status'] == 'ate':
            # COMEU COMIDA: RECOMPENSA GIGANTE!
            fitness += 50000  # Massivo!
            moves_without_food = 0
            # Resetar distância inicial para a nova comida
            initial_distance = distanceToFood(g.snake.head(), tuple(g.food.coordinates))
        else:
            # Recompensar aproximação da comida
            current_distance = distanceToFood(g.snake.head(), tuple(g.food.coordinates))
            
            # Se aproximou da comida, ganhar pontos
            if current_distance < initial_distance:
                fitness += 10  # Bônus por se aproximar
            else:
                fitness -= 5  # Pequena penalidade por se afastar
            
            initial_distance = current_distance
    
    # Bônus MASSIVO e exponencial por cada comida
    if g.score > 0:
        fitness += (g.score ** 2.5) * 10000
    
    # Penalidade severa se não comeu nada
    if g.score == 0:
        fitness = fitness * 0.1  # Reduz fitness drasticamente
    
    return fitness, g.score, total_moves


# ==================== TREINAMENTO ====================
# Variáveis globais para tracking
current_generation = 0
stats_tracker = TrainingStats()
# GUI instance
gui = None
best_genome_this_gen = None
best_fitness_this_gen = 0
best_score_this_gen = 0


def eval_genomes(genomes, config):
    """
    Avalia todos os genomas da população.
    Chamado pelo NEAT a cada geração.
    """
    global current_generation, stats_tracker
    global best_genome_this_gen, best_fitness_this_gen, best_score_this_gen
    
    fitness_scores = []
    game_scores = []
    
    best_genome_this_gen = None
    best_fitness_this_gen = 0
    best_score_this_gen = 0
    
    for genome_id, genome in genomes:
        fitness, score, moves = eval_genome(genome, config)
        genome.fitness = fitness
        
        fitness_scores.append(fitness)
        game_scores.append(score)
        
        # Rastrear melhor da geração
        if fitness > best_fitness_this_gen:
            best_fitness_this_gen = fitness
            best_genome_this_gen = genome
            best_score_this_gen = score
    
    # Atualizar estatísticas
    avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0
    stats_tracker.update(
        current_generation,
        best_genome_this_gen,
        best_fitness_this_gen,
        avg_fitness,
        best_score_this_gen
    )
    
    print(f"\n📊 Geração {current_generation}:")
    print(f"   Melhor Fitness: {best_fitness_this_gen:.2f}")
    print(f"   Fitness Médio: {avg_fitness:.2f}")
    print(f"   Melhor Score: {best_score_this_gen}")
    print(f"   Melhor Score Geral: {stats_tracker.best_overall_score}")


# ==================== CHECKPOINTER CUSTOMIZADO ====================
class CustomCheckpointer(neat.Checkpointer):
    """Checkpointer que mantém apenas os 2 últimos checkpoints."""
    
    def __init__(self, generation_interval=1, filename_prefix='neat-checkpoint-'):
        super().__init__(generation_interval, filename_prefix=filename_prefix)
        self.checkpoint_prefix = filename_prefix
    
    def end_generation(self, config, population, species_set):
        # Salvar checkpoint atual
        super().end_generation(config, population, species_set)
        
        # Limpar checkpoints antigos, mantendo apenas os 2 últimos
        import glob
        checkpoints = sorted(glob.glob(f'{self.checkpoint_prefix}*'))
        if len(checkpoints) > 2:
            for old_checkpoint in checkpoints[:-2]:
                try:
                    os.remove(old_checkpoint)
                except:
                    pass


def run_neat(config_file, generations=50, infinite=False):
    """
    Executa o algoritmo NEAT para treinar a IA.
    Integrado com TrainingGUI (GUI roda no thread principal; NEAT em background).
    """
    global current_generation, stats_tracker, gui

    print("=" * 60)
    if infinite:
        print("TREINAMENTO INFINITO - Pressione Ctrl+C para parar")
    else:
        print("INICIANDO TREINAMENTO DA IA COM NEAT")
    print("=" * 60)

    # Resetar estatísticas
    stats_tracker = TrainingStats()
    current_generation = 0

    # Carregar configuração
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    # Criar população
    population = neat.Population(config)

    # Adicionar reporters para acompanhar o progresso
    population.add_reporter(neat.StdOutReporter(True))

    # Adicionar checkpointer customizado
    population.add_reporter(CustomCheckpointer(generation_interval=10, filename_prefix='neat-checkpoint-'))

    # Criar GUI (no thread principal). Tentamos carregar `gui.py` (mais novo),
    # senão tentamos manter compatibilidade com `gui_trainer.py` se existir.
    gui = None
    try:
        # importar módulo gui dinamicamente para evitar import circular durante
        # carregamento do módulo principal
        import gui as gui_module
        gui = gui_module.TrainingUI()
    except Exception:
        try:
            from gui_trainer import TrainingGUI
            gui = TrainingGUI()
        except Exception:
            gui = None

    if infinite:
        print(f"\nTreinamento infinito iniciado...")
        print(f"População: {config.pop_size} cobras por geração")
        print(f"Checkpoint a cada 10 gerações (mantém últimos 2)")
        print(f"\n⚠️  Pressione Ctrl+C quando quiser parar o treinamento\n")
    else:
        print(f"\nTreinando por {generations} gerações...")
        print(f"População: {config.pop_size} cobras por geração")
        print(f"Checkpoint a cada 10 gerações (mantém últimos 2)\n")

    # Executar NEAT em thread separada; GUI ficará no thread principal.
    winner = None

    def run_generation(genomes, cfg):
        global current_generation, best_fitness_this_gen, best_score_this_gen
        eval_genomes(genomes, cfg)
        # calc average fitness safely
        fitnesses = [g.fitness for _, g in genomes if hasattr(g, 'fitness')]
        avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0

        # Agendar atualização da GUI no thread do Tk (thread-safe)
        try:
            if gui and hasattr(gui, 'root'):
                gen = current_generation
                bf = best_fitness_this_gen
                af = avg_fitness
                bs = best_score_this_gen
                try:
                    gui.root.after(0, lambda: gui.update_stats(gen, bf, af, bs))
                except Exception:
                    pass

                # Se existir um melhor genoma nesta geração, pedir para visualizá-lo no canvas
                try:
                    if best_genome_this_gen is not None and hasattr(gui, 'canvas_play') and hasattr(gui, 'show_genome'):
                        def show_best():
                            try:
                                # ajustar canvas (opcional)
                                try:
                                    gtmp = game.Game()
                                    gui.canvas_play.config(width=gtmp.width, height=gtmp.height)
                                except Exception:
                                    pass
                                gui.show_genome(best_genome_this_gen, cfg, speed=150, max_moves=400)
                            except Exception:
                                pass
                        try:
                            gui.root.after(0, show_best)
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass

        current_generation += 1

    def neat_runner():
        nonlocal winner
        try:
            if infinite:
                winner_local = population.run(run_generation, n=None)
            else:
                winner_local = population.run(run_generation, generations)
            winner = winner_local
        except KeyboardInterrupt:
            print("\n\n⚠️  Treinamento interrompido pelo usuário!")
            if stats_tracker.best_overall_genome:
                winner = stats_tracker.best_overall_genome
        except Exception as e:
            print("Erro no runner NEAT:", e)

    t_neat = threading.Thread(target=neat_runner, daemon=True)
    t_neat.start()

    # Rodar GUI (bloqueante). Se não há GUI, apenas aguarda o fim do treinamento.
    try:
        if gui:
            gui.start()  # mainloop() — deve rodar no thread principal
        else:
            # sem GUI, aguardar finalização do thread NEAT
            t_neat.join()
    except KeyboardInterrupt:
        print("\nGUI interrompida pelo usuário.")
    except Exception:
        pass

    # Após GUI fechar, aguardar finalização do NEAT se ainda estiver rodando
    if t_neat.is_alive():
        try:
            t_neat.join(timeout=1)
        except Exception:
            pass

    # Se o runner terminou, winner estará preenchido
    # Salvar resultados (se houver)
    if winner:
        print("\n" + "=" * 60)
        print("TREINAMENTO CONCLUÍDO!")
        print("=" * 60)

        with open('best_snake.pickle', 'wb') as f:
            pickle.dump((winner, config), f)
        stats_tracker.save('training_stats.pickle')

        print("\n✅ Melhor genoma salvo em 'best_snake.pickle'")
        print("✅ Estatísticas salvas em 'training_stats.pickle'")
        print(f"\n📊 Estatísticas Finais:")
        print(f"   Melhor Fitness: {stats_tracker.best_overall_fitness:.2f}")
        print(f"   Melhor Score: {stats_tracker.best_overall_score}")
        print(f"   Gerações: {stats_tracker.total_generations}")

        # Mostrar melhor indivíduo na interface, se GUI está ativa
        try:
            if gui and winner:
                viewer = GameVisualizer(winner, config, stats_tracker)
                threading.Thread(
                    target=viewer.play_on_canvas,
                    args=(gui.canvas_play,),
                    daemon=True
                ).start()
        except Exception as e:
            print("Não foi possível exibir o melhor indivíduo na GUI:", e)

    return winner, config, stats_tracker


# ==================== VISUALIZAÇÃO (PREPARADO PARA FUTURO) ====================
class GameVisualizer:
    """Classe preparada para visualizar o melhor genoma jogando.
    
    TODO: Implementar usando tkinter ou pygame para mostrar:
    - O jogo sendo jogado pelo melhor genoma
    - Estatísticas em tempo real
    - Gráficos de evolução
    """
    
    def __init__(self, genome, config, stats=None):
        self.genome = genome
        self.config = config
        self.stats = stats
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.game = None
    
    def play_on_canvas(self, tk_canvas, delay=50, max_moves=1000):
        """
        Visualiza o genoma jogando diretamente em um Canvas Tkinter.
        Usa snapshots retornados por game.step() e desenha como em main.draw_state.
        """
        # Always create a fresh headless Game instance (Game doesn't accept canvas param)
        try:
            self.game = game.Game()
        except Exception:
            self.game = game.Game()

        self.net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        self._play_moves = 0
        self._play_max = max_moves
        self._play_delay = int(delay)  # milliseconds
        self._play_canvas = tk_canvas

        def _draw_state(state):
            c = self._play_canvas
            try:
                c.delete('all')
            except Exception:
                return
            # draw food
            fx, fy = state.get('food', [-1, -1])
            if fx >= 0:
                c.create_rectangle(fx, fy, fx + self.game.space, fy + self.game.space, fill="#FF0000", tag='food')
            # draw snake
            for i, (sx, sy) in enumerate(state.get('snake', [])):
                color = "#008CFF" if i == 0 else "#00FF00"
                c.create_rectangle(sx, sy, sx + self.game.space, sy + self.game.space, fill=color, tag='snake')
            # score (top-left)
            try:
                c.create_text(6, 6, anchor='nw', fill='white', text=f"Score: {self.game.score}")
            except Exception:
                pass

        def _step():
            # Stop if game over or reached max moves
            if self.game.game_over or self._play_moves >= self._play_max:
                # show final message
                try:
                    if self.game.game_over:
                        self._play_canvas.create_text(
                            int(self._play_canvas.winfo_width()/2),
                            int(self._play_canvas.winfo_height()/2),
                            text="GAME OVER", fill="red", font=("Consolas", 28)
                        )
                except Exception:
                    pass
                return  # stop scheduling further frames

            try:
                inputs = get_snake_inputs(self.game)
                outputs = self.net.activate(inputs)
                direction_index = outputs.index(max(outputs))
                directions = ['up', 'down', 'left', 'right']
                chosen_direction = directions[direction_index]
                self.game.change_direction(chosen_direction)
                state = self.game.step()
                _draw_state(state)
            except Exception as e:
                print("Erro durante step do play_on_canvas:", e)
                return

            self._play_moves += 1
            try:
                self._play_canvas.after(self._play_delay, _step)
            except Exception:
                pass

        # start
        try:
            self._play_canvas.after(0, _step)
        except Exception as e:
            print("Não foi possível iniciar play_on_canvas:", e)

        
    def play_game(self, visual=False):
        """Joga uma partida com o genoma.
        
        Args:
            visual: Se True, mostra interface gráfica (TODO: implementar)
        """
        self.game = game.Game()
        
        moves = 0
        max_moves = 1000
        
        while not self.game.game_over and moves < max_moves:
            inputs = get_snake_inputs(self.game)
            outputs = self.net.activate(inputs)
            direction_index = outputs.index(max(outputs))
            directions = ['up', 'down', 'left', 'right']
            chosen_direction = directions[direction_index]
            
            self.game.change_direction(chosen_direction)
            state = self.game.step()
            
            if state['status'] == 'ate':
                print(f"🍎 Move {moves + 1}: COMEU! Score: {self.game.score}")
            
            moves += 1
            
            # TODO: Se visual=True, atualizar interface gráfica aqui
        
        return {
            'score': self.game.score,
            'moves': moves,
            'reason': self.game.reason if self.game.game_over else 'max_moves'
        }
    
    def show_statistics(self):
        """Mostra estatísticas do treinamento.
        
        TODO: Implementar gráficos usando matplotlib
        """
        if not self.stats:
            print("⚠️  Nenhuma estatística disponível")
            return
        
        print("\n" + "=" * 60)
        print("ESTATÍSTICAS DO TREINAMENTO")
        print("=" * 60)
        print(f"Gerações: {self.stats.total_generations}")
        print(f"Melhor Fitness: {self.stats.best_overall_fitness:.2f}")
        print(f"Melhor Score: {self.stats.best_overall_score}")
        print(f"\nEvolução do Fitness:")
        for i, (best, avg) in enumerate(zip(self.stats.generation_best_fitness, 
                                             self.stats.generation_avg_fitness)):
            print(f"  Gen {i}: Best={best:.2f}, Avg={avg:.2f}")


def test_best_genome():
    """Testa o melhor genoma treinado jogando uma partida."""
    print("=" * 60)
    print("TESTANDO MELHOR GENOMA")
    print("=" * 60)
    
    # Carregar o melhor genoma
    try:
        with open('best_snake.pickle', 'rb') as f:
            winner, config = pickle.load(f)
    except FileNotFoundError:
        print("\n❌ Erro: Arquivo 'best_snake.pickle' não encontrado!")
        print("   Execute o treinamento primeiro: python snake_ai.py train")
        return
    
    # Carregar estatísticas (opcional)
    try:
        stats = TrainingStats.load('training_stats.pickle')
    except FileNotFoundError:
        stats = None
    
    # Criar visualizador
    visualizer = GameVisualizer(winner, config, stats)
    
    # Mostrar estatísticas
    if stats:
        visualizer.show_statistics()
    
    # Jogar partida de teste
    print(f"\n🎮 Iniciando partida de teste...\n")
    result = visualizer.play_game(visual=False)  # TODO: mudar para True quando implementar GUI
    
    print(f"\n{'=' * 60}")
    print(f"RESULTADO FINAL")
    print(f"{'=' * 60}")
    print(f"Score: {result['score']}")
    print(f"Movimentos: {result['moves']}")
    if result['reason'] != 'max_moves':
        print(f"Motivo da morte: {result['reason']}")
    print(f"{'=' * 60}\n")


# ==================== MAIN ====================
if __name__ == '__main__':
    import sys
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    
    # Verificar argumentos da linha de comando
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            test_best_genome()
        elif sys.argv[1] == 'train':
            if len(sys.argv) > 2 and sys.argv[2] == 'infinite':
                # Modo infinito
                run_neat(config_path, infinite=True)
            else:
                # Modo normal com número de gerações
                generations = int(sys.argv[2]) if len(sys.argv) > 2 else 50
                run_neat(config_path, generations)
        elif sys.argv[1] == 'gui':
            # Abrir a UI de visualização (usa gui.py se existir)
            try:
                import gui as gui_module
                gui_module.start_gui_mode()
            except Exception:
                try:
                    from gui_trainer import TrainingGUI
                    g = TrainingGUI()
                    g.start()
                except Exception:
                    print('Não foi possível iniciar a GUI (nenhum módulo de GUI disponível)')
        else:
            print("Uso:")
            print("  python snake_ai.py train [gerações]  - Treina a IA por N gerações")
            print("  python snake_ai.py train infinite    - Treina até Ctrl+C")
            print("  python snake_ai.py test              - Testa a melhor IA")
    else:
        # Padrão: treinar
        run_neat(config_path, generations=50)
