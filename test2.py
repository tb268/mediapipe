"""

import tensorflow as tf

import numpy as np

print("＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄")
mnist = tf.keras.datasets.mnist 

(x_train, y_train), (x_test, y_test) = mnist.load_data() #データの読み込み
print(type(y_train))
x_train, x_test = x_train / 255.0, x_test / 255.0 #データの正規化


# モデルの構築
# Sequential API
model = tf.keras.models.Sequential([
tf.keras.layers.Flatten(input_shape=(28,28 )),#入力値のサイズを定義
tf.keras.layers.Dense(128, activation='relu'),#隠れ層のサイズを定義
tf.keras.layers.Dropout(0.2),#訓練中に「()内の割合」のユニットをランダムに無効化します。
tf.keras.layers.Dense(10, activation='softmax')#出力層のサイズを定義
])


# optimizer, loss, metricsの設定（学習設定）
model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# 学習
model.fit(x_train, y_train, epochs=5)#学習のバッチ数はbatch_sizeで定義できる。デフォルトでは32個に分かれる
print(x_train.shape,y_train.shape)

# 評価
#test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

#print('\nTest accuracy、学習制度:', test_acc)
#モデルでスコア化
#print(x_train[0])
ans=x_train[10].reshape((1, 28, 28))
pred = model.predict(ans, batch_size=1, verbose=0)
#判別結果で最も高い数値を抜き出し
score = np.argmax(pred)

print('\n推定結果:', score)
"""