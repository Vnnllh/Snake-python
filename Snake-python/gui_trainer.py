import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class TrainingGUI:
    """
    GUI for displaying training stats, 3 stacked matplotlib plots (Best fitness,
    Avg fitness, Best score) and a Tkinter canvas below for showing the best
    individual playing.
    Note: this class does NOT call mainloop() in __init__. Call start() to run
    the Tk mainloop (so it can be started in a separate thread).
    """
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Treinamento Snake IA - NEAT")
        self.root.geometry("780x900")

        # Frames principais
        self.stats_frame = tk.Frame(self.root)
        self.stats_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.graph_frame = tk.Frame(self.root)
        self.graph_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.game_frame = ttk.LabelFrame(self.root, text="Melhor indivíduo jogando")
        self.game_frame.pack(side=tk.TOP, pady=10, padx=10)

        # Labels de estatísticas
        self.gen_label = tk.Label(self.stats_frame, text="Geração: 0", font=("Arial", 12))
        self.gen_label.grid(row=0, column=0, padx=10, sticky="w")

        self.best_fitness_label = tk.Label(self.stats_frame, text="Melhor Fitness: 0", font=("Arial", 12))
        self.best_fitness_label.grid(row=0, column=1, padx=10, sticky="w")

        self.avg_fitness_label = tk.Label(self.stats_frame, text="Fitness Médio: 0", font=("Arial", 12))
        self.avg_fitness_label.grid(row=0, column=2, padx=10, sticky="w")

        self.best_score_label = tk.Label(self.stats_frame, text="Melhor Score: 0", font=("Arial", 12))
        self.best_score_label.grid(row=0, column=3, padx=10, sticky="w")

        # Cria figura de 3 gráficos empilhados
        self.fig = Figure(figsize=(7.5, 8), dpi=100)

        self.ax_best = self.fig.add_subplot(311)
        self.ax_best.set_title("Melhor Fitness por Geração")
        self.ax_best.grid(True)

        self.ax_avg = self.fig.add_subplot(312)
        self.ax_avg.set_title("Fitness Médio por Geração")
        self.ax_avg.grid(True)

        self.ax_score = self.fig.add_subplot(313)
        self.ax_score.set_title("Melhor Score por Geração")
        self.ax_score.grid(True)

        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Canvas para jogar (abaixo dos gráficos)
        self.canvas_play = tk.Canvas(self.game_frame, width=400, height=400, bg="black")
        self.canvas_play.pack(padx=10, pady=10)

        # Armazena histórico
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_score_history = []

        # Control flags
        self._running = False

    def start(self):
        """Start the Tk mainloop. Call this in a separate thread."""
        if not self._running:
            self._running = True
            try:
                self.root.mainloop()
            finally:
                self._running = False

    def update_stats(self, generation, best_fitness, avg_fitness, best_score):
        """Update labels and graphs with the latest stats."""
        try:
            self.gen_label.config(text=f"Geração: {generation}")
            self.best_fitness_label.config(text=f"Melhor Fitness: {best_fitness:.2f}")
            self.avg_fitness_label.config(text=f"Fitness Médio: {avg_fitness:.2f}")
            self.best_score_label.config(text=f"Melhor Score: {best_score}")

            # Salva para plotar
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            self.best_score_history.append(best_score)

            # Atualiza gráficos
            self._update_graphs()
        except tk.TclError:
            # GUI foi fechada
            pass

    def _update_graphs(self):
        self.ax_best.clear()
        self.ax_avg.clear()
        self.ax_score.clear()

        self.ax_best.set_title("Melhor Fitness por Geração")
        self.ax_avg.set_title("Fitness Médio por Geração")
        self.ax_score.set_title("Melhor Score por Geração")

        self.ax_best.grid(True)
        self.ax_avg.grid(True)
        self.ax_score.grid(True)

        gens = list(range(len(self.best_fitness_history)))

        if gens:
            self.ax_best.plot(gens, self.best_fitness_history, linewidth=2)
            self.ax_avg.plot(gens, self.avg_fitness_history, linewidth=2)
            self.ax_score.plot(gens, self.best_score_history, linewidth=2)

        self.fig.tight_layout()
        try:
            self.canvas_fig.draw()
        except Exception:
            pass

    def close(self):
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass
