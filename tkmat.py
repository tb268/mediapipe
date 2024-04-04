import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import numpy as np

# データを生成する関数
def generate_data():
    num_lines = 5
    num_points = 100
    lines = []

    for _ in range(num_lines):
        x = np.linspace(0, 10, num_points)
        y = np.random.rand(num_points)
        z = np.random.rand(num_points)
        lines.append((x, y, z))

    return lines

# アニメーションを更新する関数
def update_plot(frame):
    lines = generate_data()

    for i, line in enumerate(lines):
        x, y, z = line
        lines_3d[i].set_data(x, y)
        lines_3d[i].set_3d_properties(z)

    return lines_3d

# Tkinterウィンドウの作成
root = tk.Tk()
root.geometry("800x600")

# MatplotlibのFigureを作成
fig = Figure(figsize=(6, 6), dpi=100)
ax = fig.add_subplot(111, projection='3d')

# 空の3Dラインを作成
lines_3d = [ax.plot([], [], [])[0] for _ in range(5)]

# MatplotlibのFigureをTkinterウィンドウに埋め込む
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# アニメーションの設定
ani = animation.FuncAnimation(fig, update_plot, frames=100, interval=100)

# ウィンドウのループ
root.mainloop()
