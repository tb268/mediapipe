import tkinter as tk

def on_configure(event):
    """Canvasのスクロール領域を調整するコールバック関数"""
    canvas.configure(scrollregion=canvas.bbox("all"))

root = tk.Tk()
root.title("Scrollable Frame Example")

# Canvasウィジェットを作成します
canvas = tk.Canvas(root)
canvas.pack(side="left", fill="both", expand=True)

# スクロールバーを作成し、Canvasに配置します
scrollbar = tk.Scrollbar(root, command=canvas.yview)
scrollbar.pack(side="right", fill="y")
canvas.configure(yscrollcommand=scrollbar.set)

# FrameをCanvas内に配置します
frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=frame, anchor="nw")

# Canvasのサイズが変更されたときにスクロール領域を調整するように設定します
frame.bind("<Configure>", on_configure)

# ダミーのラベルをFrame内に配置します
for i in range(50):
    label = tk.Label(frame, text=f"Label {i}")
    label.pack()

root.mainloop()
