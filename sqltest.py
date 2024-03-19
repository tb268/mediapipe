#sqlite3 main.db
#データベースに保存用のシステム

import sqlite3
import json
#
def sql_set(times,landmark_point,pose,isFirst):
    #DBの名前。同一ディレクトリ内に生成
    dbname = 'main.db'

    with sqlite3.connect(dbname) as conn:

        # SQLiteを操作するためのカーソルを作成
        cur = conn.cursor()
        if isFirst:
            cur.execute(f'DELETE FROM {pose}')

        # テーブルの作成
        #cur.execute(
        #   'CREATE TABLE items(id INTEGER PRIMARY KEY AUTOINCREMENT, name STRING, price INTEGER)'
        #)
        #cur.execute(
        #    'CREATE TABLE pose(id INTEGER PRIMARY KEY AUTOINCREMENT, name STRING, price INTEGER)'
        #)
        # 50個の数値型カラムの定義を生成
        #columns = [f"col{i} FLOAT" for i in range(1, 66)]

        # 完全なテーブル作成クエリを構築
        #DB内にposeテーブルがない時に作成
        
        cur.execute(f"CREATE TABLE IF NOT EXISTS {pose} (id INTEGER PRIMARY KEY,data TEXT)")
        # データ登録
        # 挿入したい50個のデータ（例）
        data = landmark_point  # このタプルに50個の要素が含まれていると仮定

        # 配列全体をJSON文字列に変換
        json_data = json.dumps(data)

        # データベースに挿入
        cur.execute(f"INSERT INTO {pose} (data) VALUES (?);", (json_data,))

        # 複数データ登録
        #cur.executemany('INSERT INTO items values(?, ?, ?)', inserts)
        
        # コミットしないと登録が反映されない
        conn.commit()
        # データ検索
        #cur.execute('SELECT * FROM pose')

        # 取得したデータはカーソルの中に入る
        #for row in cur:
        #   print(row)
        #closeするとDBが閉じるので行わない。最後にやるかも
        #conn.close()


#DBのテーブル全削除
"""
dbname = 'main.db'
with sqlite3.connect(dbname) as conn:
    # SQLiteを操作するためのカーソルを作成
    cur = conn.cursor()
    cur.execute("DELETE FROM pose")
    # コミットしないと登録が反映されない
    conn.commit()
    # データ検索
    #cur.execute('SELECT * FROM pose')

    # 取得したデータはカーソルの中に入る
    #for row in cur:
    #   print(row)
    #conn.close()
"""