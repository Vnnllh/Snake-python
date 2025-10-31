
from tkinter import *
import game_logic

# UI-specific constants (colors)
SNAKE_COLOR = "#00FF00"
HEAD_COLOR = "#008CFF"
FOOD_COLOR = "#FF0000"
BACKGROUND_COLOR = "#000000"
# Intervalo em ms
SPEED = 100


def center_window(window):
    window.update_idletasks()
    window_width = window.winfo_width()
    window_height = window.winfo_height()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    x = int((screen_width/2) - (window_width/2))
    y = int((screen_height/2) - (window_height/2))

    window.geometry(f"{window_width}x{window_height}+{x}+{y}")


game = game_logic.Game()


window = Tk()
window.title("Snake game")
window.resizable(False, False)

label = Label(window, text=f"score:{game.score}", font=('consolas', 24))
label.pack()

canvas = Canvas(window, bg=BACKGROUND_COLOR, height=game.height, width=game.width)
canvas.pack()

center_window(window)


def draw_state(state: dict):
    # Simples: limpar e redesenhar tudo a cada frame.
    canvas.delete('all')
    # desenhar food
    fx, fy = state['food']
    if fx >= 0:
        canvas.create_rectangle(fx, fy, fx + game.space, fy + game.space, fill=FOOD_COLOR, tag='food')
    # desenhar snake
    snake = state['snake']
    for i, (sx, sy) in enumerate(snake):
        color = HEAD_COLOR if i == 0 else SNAKE_COLOR
        canvas.create_rectangle(sx, sy, sx + game.space, sy + game.space, fill=color, tag='snake')


def next_turn():
    state = game.step()
    label.config(text=f"score:{state['score']}")
    draw_state(state)
    if state.get('game_over'):
        canvas.create_text(canvas.winfo_width()/2, canvas.winfo_height()/2, font=('consolas',70), text="GAME OVER", fill="red", tag="gameover")
        return
    window.after(SPEED, next_turn)


def on_key(event):
    key = event.keysym
    mapping = {'Left':'left', 'Right':'right', 'Up':'up', 'Down':'down'}
    if key in mapping:
        game.change_direction(mapping[key])


window.bind('<Key>', on_key)

next_turn()

window.mainloop()