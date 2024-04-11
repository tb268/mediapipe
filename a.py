#matplotlibはダウングレードが必要
#!pip install matplotlib==3.5.1
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import ArtistAnimation

# データの準備
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# パラメータの設定
t = np.linspace(0, 2 * np.pi, 100)
x1 = np.sin(t)
y1 = np.cos(t)
z1 = t

x2 = np.cos(t)
y2 = np.sin(t)
z2 = t

x3 = np.tan(t)
y3 = np.cos(t)
z3 = t

# プロットの初期化
plots = []

# フレームごとのプロットデータを作成
for i in range(len(t) - 1):

    plot_data = []
    s=ax.plot(x1[i:i+2], y1[i:i+2], z1[i:i+2], color='r')[0]
    #print(x2[i:i+2], y2[i:i+2], z2[i:i+2])
    plot_data.append(s)
    plot_data.append(ax.plot(x2[i:i+2], y2[i:i+2], z2[i:i+2], color='g')[0])
    plot_data.append(ax.plot(x3[i:i+2], y3[i:i+2], z3[i:i+2], color='b')[0])
    
    plots.append(plot_data)

# アーティストのアニメーションを作成

print(len(plots))
print(len(plots[0]))
print(plots[0])
print(plots[0][0])
ani = ArtistAnimation(fig, plots, interval=100, repeat=False)

# アニメーションの保存
ani.save('3d_lines_animation.mp4', writer='ffmpeg')

#plt.show()
