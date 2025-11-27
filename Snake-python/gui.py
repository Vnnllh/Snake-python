import tkinter as tk
from tkinter import ttk
import pickle
import os
import game_logic as game
import neat

# matplotlib for plotting training stats
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


class VisualPlayer:
    """Plays a genome on a Tkinter Canvas using snake game snapshots."""
    def __init__(self, genome, config, canvas, speed=100, max_moves=1000):
        self.genome = genome
        self.config = config
        self.canvas = canvas
        self.speed = int(speed)
        self.max_moves = int(max_moves)
        self.game = None
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self._moves = 0
        self._running = False

    def start(self):
        self.game = game.Game()
        self._moves = 0
        self._running = True
        # adjust canvas size to game
        try:
            self.canvas.config(width=self.game.width, height=self.game.height)
        except Exception:
            pass
        self.canvas.after(0, self._step)

    def stop(self):
        self._running = False

    def _draw_state(self, state):
        c = self.canvas
        try:
            c.delete('all')
        except Exception:
            return
        # draw food
        fx, fy = state.get('food', (-1, -1))
        if fx >= 0:
            c.create_rectangle(fx, fy, fx + self.game.space, fy + self.game.space, fill="#FF0000", tag='food')
        # draw snake
        for i, (sx, sy) in enumerate(state.get('snake', [])):
            color = "#008CFF" if i == 0 else "#00FF00"
            c.create_rectangle(sx, sy, sx + self.game.space, sy + self.game.space, fill=color, tag='snake')
        # score
        try:
            c.create_text(6, 6, anchor='nw', fill='white', text=f"Score: {self.game.score}")
        except Exception:
            pass

    def _step(self):
        if not self._running or self.game.game_over or self._moves >= self.max_moves:
            if self.game and self.game.game_over:
                try:
                    self.canvas.create_text(self.canvas.winfo_width()//2, self.canvas.winfo_height()//2,
                                            text='GAME OVER', fill='red', font=("Consolas", 28))
                except Exception:
                    pass
            return

        try:
            # lazy import to avoid circular import when gui is imported from snake_ai
            import snake_ai
            inputs = snake_ai.get_snake_inputs(self.game)
            outputs = self.net.activate(inputs)
            direction_index = outputs.index(max(outputs))
            directions = ['up', 'down', 'left', 'right']
            chosen_direction = directions[direction_index]
            self.game.change_direction(chosen_direction)
            state = self.game.step()
            self._draw_state(state)
        except Exception as e:
            print('VisualPlayer error:', e)
            return

        self._moves += 1
        try:
            self.canvas.after(self.speed, self._step)
        except Exception:
            pass




class TrainingUI:
    """Simple Tkinter UI to show training stats and play the best genome."""
    def __init__(self, stats_file='training_stats.pickle', best_file='best_snake.pickle'):
        self.stats_file = stats_file
        self.best_file = best_file
        self.root = tk.Tk()
        self.root.title('Snake AI - Training Viewer')
        self._build_layout()
        self.stats = None
        self.winner = None
        self.config = None
        self.player = None
        self._load_data()

    def _build_layout(self):
        top = ttk.Frame(self.root)
        top.pack(side='top', fill='x', padx=6, pady=6)
        mid = ttk.Frame(self.root)
        mid.pack(side='top', fill='both', expand=True, padx=6, pady=6)
        bot = ttk.Frame(self.root)
        bot.pack(side='bottom', fill='x', padx=6, pady=6)

        self.lbl_generations = ttk.Label(top, text='Gerações: -')
        self.lbl_generations.pack(side='left', padx=8)
        self.lbl_best_score = ttk.Label(top, text='Melhor Score: -')
        self.lbl_best_score.pack(side='left', padx=8)
        self.lbl_best_fitness = ttk.Label(top, text='Melhor Fitness: -')
        self.lbl_best_fitness.pack(side='left', padx=8)

        # Canvas to play the game
        self.canvas_play = tk.Canvas(mid, bg='black', width=700, height=400)
        self.canvas_play.pack(fill='both', expand=True)

        # Plot frame (matplotlib) below the canvas
        self.plot_frame = ttk.Frame(mid)
        self.plot_frame.pack(side='top', fill='x', expand=False, pady=(6, 0))

        # initialize plot data
        self.plot_best = []
        self.plot_avg = []
        self.plot_generations = []

        if MATPLOTLIB_AVAILABLE:
            self.fig = Figure(figsize=(6, 2.0), dpi=100)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_title('Evolução do Fitness')
            self.ax.set_xlabel('Geração')
            self.ax.set_ylabel('Fitness')
            self.line_best, = self.ax.plot([], [], label='Best')
            self.line_avg, = self.ax.plot([], [], label='Avg')
            self.ax.legend(loc='upper left')
            self.fig.tight_layout()
            self.tk_fig = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
            self.tk_fig.get_tk_widget().pack(fill='x', expand=True)
        else:
            self.fig = None

        self.btn_play = ttk.Button(bot, text='Play Best', command=self.toggle_play)
        self.btn_play.pack(side='left', padx=6)
        self.btn_reload = ttk.Button(bot, text='Reload Best/Stats', command=self._load_data)
        self.btn_reload.pack(side='left', padx=6)
        ttk.Label(bot, text='Speed (ms)').pack(side='left', padx=6)
        self.speed_scale = ttk.Scale(bot, from_=20, to=500, orient='horizontal')
        self.speed_scale.set(100)
        self.speed_scale.pack(side='left', padx=6, fill='x', expand=True)

    def _load_data(self):
        # load stats via snake_ai.TrainingStats if possible (lazy import)
        try:
            import snake_ai
            try:
                self.stats = snake_ai.TrainingStats.load(self.stats_file)
            except Exception:
                # fallback to raw pickle
                with open(self.stats_file, 'rb') as f:
                    self.stats = pickle.load(f)
        except Exception:
            try:
                with open(self.stats_file, 'rb') as f:
                    self.stats = pickle.load(f)
            except Exception:
                self.stats = None

        if self.stats:
            self.lbl_generations.config(text=f"Gerações: {self.stats.total_generations}")
            self.lbl_best_score.config(text=f"Melhor Score: {self.stats.best_overall_score}")
            try:
                self.lbl_best_fitness.config(text=f"Melhor Fitness: {self.stats.best_overall_fitness:.2f}")
            except Exception:
                self.lbl_best_fitness.config(text=f"Melhor Fitness: -")
            # populate plot arrays from stats if available
            try:
                self.plot_best = list(self.stats.generation_best_fitness)
                self.plot_avg = list(self.stats.generation_avg_fitness)
                self.plot_generations = list(range(len(self.plot_best)))
                self._redraw_plot()
            except Exception:
                pass
        else:
            self.lbl_generations.config(text='Gerações: -')
            self.lbl_best_score.config(text='Melhor Score: -')
            self.lbl_best_fitness.config(text='Melhor Fitness: -')

        # load best genome
        try:
            with open(self.best_file, 'rb') as f:
                self.winner, self.config = pickle.load(f)
        except Exception:
            self.winner, self.config = None, None

        if not self.winner:
            self.canvas_play.delete('all')
            self.canvas_play.create_text(350, 250, text='Nenhum best_snake.pickle encontrado', fill='white', font=("Consolas", 18))

    def update_stats(self, generation, best_fitness, avg_fitness, best_score):
        try:
            self.lbl_generations.config(text=f"Gerações: {generation + 1}")
            self.lbl_best_score.config(text=f"Melhor Score: {best_score}")
            self.lbl_best_fitness.config(text=f"Melhor Fitness: {best_fitness:.2f}")
        except Exception:
            pass
        # update internal plot data and redraw
        try:
            self.plot_generations.append(generation)
            self.plot_best.append(best_fitness)
            self.plot_avg.append(avg_fitness)
            self._redraw_plot()
        except Exception:
            pass

    def _redraw_plot(self):
        if not MATPLOTLIB_AVAILABLE or not getattr(self, 'fig', None):
            return
        try:
            self.line_best.set_data(self.plot_generations, self.plot_best)
            self.line_avg.set_data(self.plot_generations, self.plot_avg)
            self.ax.relim()
            self.ax.autoscale_view()
            self.tk_fig.draw()
        except Exception:
            pass

    def show_genome(self, genome, config, speed=150, max_moves=400, autostop_seconds=6):
        """Play a genome on the canvas, stopping any existing player first.

        autostop_seconds: stop playback after this many seconds to avoid overlapping
        multiple generations playing continuously.
        """
        try:
            # stop previous player if running
            if self.player and getattr(self.player, '_running', False):
                try:
                    self.player.stop()
                except Exception:
                    pass

            self.player = VisualPlayer(genome, config, self.canvas_play, speed=speed, max_moves=max_moves)
            self.player.start()

            # schedule an automatic stop after autostop_seconds
            try:
                self.root.after(int(autostop_seconds * 1000), lambda: self.player.stop())
            except Exception:
                pass
        except Exception as e:
            print('Error in show_genome:', e)

    def toggle_play(self):
        if not self.winner or not self.config:
            return
        if self.player and getattr(self.player, '_running', False):
            self.player.stop()
            self.btn_play.config(text='Play Best')
        else:
            speed = int(self.speed_scale.get())
            self.player = VisualPlayer(self.winner, self.config, self.canvas_play, speed=speed, max_moves=1000)
            self.player.start()
            self.btn_play.config(text='Stop')

    def start(self):
        self.root.mainloop()


def start_gui_mode():
    ui = TrainingUI()
    ui.start()


if __name__ == '__main__':
    start_gui_mode()
