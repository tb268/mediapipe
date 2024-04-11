import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# データの準備
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z1 = np.sin(X) * np.cos(Y)
Z2 = np.cos(X) * np.sin(Y)

# アニメーションの初期化
def init():
    ax.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.5)
    ax.plot_surface(X, Y, Z2, cmap='plasma', alpha=0.5)
    return fig,

# アニメーションの更新
def update(frame):
    ax.view_init(30, frame)
    return fig,

# アニメーションの作成
ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 2), init_func=init, blit=True)

# アニメーションの保存
ani.save('3d_animation.mp4', writer='ffmpeg')

plt.show()
