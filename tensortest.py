

import tensorflow as tf
import pandas as pd
import numpy as np

def ScoreLearning(data1,score):

    print("＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄")
    

    #(x_train, y_train), (x_test, y_test) = mnist.load_data() #データの読み込み
    #x_train, x_test = x_train / 255.0, x_test / 255.0 #データの正規化
    
    # データの形状を変換
    train = data1

    print(train.shape)
    print(type(train))
    print(type(score))

    print("＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄＄")

    # モデルの構築
    # Sequential API
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(25, )),#入力値のサイズを定義
    tf.keras.layers.Dense(128, activation='relu'),#隠れ層のサイズを定義
    tf.keras.layers.Dropout(0.2),#訓練中に「()内の割合」のユニットをランダムに無効化します。
    tf.keras.layers.Dense(10, activation='softmax')#出力層のサイズを定義
    ])


    # optimizer, loss, metricsの設定（学習設定）
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # 学習
    #model.fit(x_train, y_train)
    model.fit(train, score, batch_size=1)

    # 評価
    #test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
    #test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

    #モデルでスコア化
    ans=train[0].reshape((1,25))
    pred = model.predict(ans, batch_size=1, verbose=0)
    #判別結果で最も高い数値を抜き出し
    score = np.argmax(pred)

    print('\nTest accuracy、学習制度:', score)

    return score

