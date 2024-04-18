import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

from time import time
import numpy as np
import matplotlib.pyplot as plt

# データの準備
x = np.linspace(0, 2*np.pi, 1000)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x + np.pi/2)

# 新しいFigureとAxesを作成
fig, ax = plt.subplots()

# ArtistAnimation インスタンスを作成
a=time()
# 各ラインをプロット
line1, = ax.plot(x, y1, color='blue', label='sin(x)')
line2, = ax.plot(x, y2, color='red', label='cos(x)')
line3, = ax.plot(x, y3, color='green', label='sin(x + π/2)')

# グラフにタイトルを追加
ax.set_title('Multiple Line Plots')

# 凡例を表示
ax.legend()

# x軸とy軸の範囲を設定
ax.set_xlim(0, 2*np.pi)
ax.set_ylim(-1.5, 1.5)

b=time()
print(b-a)
plt.show()
