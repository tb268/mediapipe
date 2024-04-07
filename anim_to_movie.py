import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class makemovie():
    def __init__(self):
        # データの準備
        x, y, z = self.generate_data()

        # プロットの準備
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')

        # アニメーションを作成する
        ani = FuncAnimation(fig, update, frames=len(x), blit=True)

        # アニメーションを保存する
        ani.save('./outmovie/animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

        #plt.show()

        pass
    # データを生成する関数（ランダムウォーク）
    def generate_data(self,num_points=1000, num_steps=100):
        x = np.zeros(num_points)
        y = np.zeros(num_points)
        z = np.zeros(num_points)

        for i in range(1, num_points):
            for j in range(num_steps):
                x[i] = x[i-1] + np.random.normal()
                y[i] = y[i-1] + np.random.normal()
                z[i] = z[i-1] + np.random.normal()
        return x, y, z

    # アニメーションを更新する関数
    def update(self,frame):
        ax.clear()
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_zlim(-50, 50)
        ax.scatter(x[:frame], y[:frame], z[:frame], c='b', marker='o')
        return fig,

