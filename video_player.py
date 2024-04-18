import sys
import time
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter.filedialog import askopenfile

from tkinter import messagebox
import cv2

from PIL import Image, ImageOps, ImageTk

# from GUI_control import GUI_control

import threading
from tensortest import ScoreLearning

import numpy as np


#mediapipe使用用
from comparison import main as comparison

#mediapipe使用用
from mediapipe_convert import maindef as mc

#pythonからコマンド実行する用
import subprocess


class create_group:

    def __init__(self,parents,frame):
        # Variable setting
        self.file_filter = [("Movie file", ".mp4")]
        self.set_movie = True
        self.thread_set = False
        self.start_movie = False
        self.video_frame = None
        super().__init__()  #親クラスのinitを使用
        self.parents=parents    #親クラス
        self.frame=frame    #親フレーム
        # Sub window=======================================
        #動画キャンバス
        self.canvas_frame = tk.Frame(self.frame, height=450, width=400)
        #シークバー
        self.scale_frame = tk.Frame(self.frame, height=1000, width=1200)
        #動画ファイルパス
        self.path_frame = tk.Frame(self.frame, height=100, width=400)
        #ボタンリスト
        self.opr_frame = tk.Frame(self.frame, height=100, width=400)

        # Widgetsmith
        self.canvas_frame.place(relx=0.05, rely=0.05)
        self.path_frame.place(relx=0.60, rely=0.2)
        self.opr_frame.place(relx=0.60, rely=0.5)
        self.scale_frame.place(relx=0.05, rely=0.8)

        # 1.1 canvas_frame (label)
        self.label = tk.Label(
            self.canvas_frame, text="Movie", bg="white", relief=tk.RIDGE
        )
        self.label.grid(row=0, column=0, sticky=tk.W + tk.E)

        # 1.2 動画表示ファイル (canvas)
        self.canvas = tk.Canvas(self.canvas_frame, width=500, height=500, bg="#A9A9A9")
        self.canvas.grid(row=1, column=0)
        # 1.2 動画表示ファイル (canvas)
        #self.canvas = tk.Canvas(self.canvas_frame, width=700, height=500, bg="#A9A9A9")
        #self.canvas.grid(row=1, column=0)
        # Scale。スクロールバー（デフォルトで作成）

        self.scaleH = tk.Scale(
                                self.scale_frame,
                                command = self.slider_scroll,
                                orient=tk.HORIZONTAL,
                                length = 700,           # 全体の長さ
                                width = 20,             # 全体の太さ
                                from_ = 0,            # 最小値（開始の値）
                                to = 100,               # 最大値
                                )
        
        self.scaleH.grid(row=2, column=0)

        # 2 path_frame
        self.button = self.opr_btn(self.path_frame, "File", self.on_click_path)
        self.button.grid(row=3, column=0, sticky=tk.W, padx=10, pady=10)

        self.path_stvar = tk.StringVar()
        path_entry = tk.Entry(
            self.path_frame, textvariable=self.path_stvar, width=70
        )
        path_entry.grid(row=3, column=1, sticky=tk.EW, padx=10)

        # 3 opr_frame
        self.button = self.opr_btn(self.opr_frame, "Start", self.on_click_start)
        self.button.grid(row=3, column=2, sticky=tk.SE, padx=10, pady=10)

        self.button = self.opr_btn(self.opr_frame, "Stop", self.on_click_stop)
        self.button.grid(row=3, column=3, sticky=tk.SE, padx=10, pady=10)

        self.button = self.opr_btn(self.opr_frame, "Reset", self.on_click_reset)
        self.button.grid(row=4, column=2, sticky=tk.SE, padx=10, pady=10)

        self.button = self.opr_btn(self.opr_frame, "Exit", self.on_click_close)
        self.button.grid(row=4, column=3, sticky=tk.SE, padx=10, pady=10)
        
        

    def on_key_press(self,event):
        # キー入力に応じてスライダーの値を変更する
        if event.char == "a":
            self.scaleH.set(self.scaleH.get() - 1)
        elif event.char == "d":
            self.scaleH.set(self.scaleH.get() + 1)

        # キー入力イベントにイベントハンドラーをバインド
        self.scaleH.bind("<KeyPress>", on_key_press)


    def slider_scroll(self, event=None):
        '''スライダーを移動したとき'''
        #スラーダーをフォーカス
        self.scaleH.focus_set()
        #startの指定
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.scaleH.get())
        #self.start_movie = False
        print(str(self.scaleH.get()))


    def opr_btn(self, set_frame, btn_name, act_command):
        return tk.Button(set_frame, text=btn_name, width=10, command=act_command)

    def on_click_path(self):
        self.movie_path = self.get_path()
        self.path_stvar.set(self.movie_path)
        self.run_one_frame()
        # Movie standby.
        self.thread_set = True
        self.thread_main = threading.Thread(target=self.main_thread_func)
        self.thread_main.start()
        

    def on_click_start(self):
        #startの指定
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.scaleH.get())
        self.start_movie = True

    def on_click_stop(self):
        self.start_movie = False
    #リセットボタン挙動
    def on_click_reset(self):
        self.start_movie = False
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        #シークバーを0にせ設定
        self.scaleH.set(0)
        self.run_one_frame()

    def on_click_close(self):
        self.set_movie = False

        # Block the calling thread until the thread represented by this instance end.
        if self.thread_set == True:
            self.thread_main.join()

        self.video_cap.release()
        self.parents.main_window.destroy() 



    def main_thread_func(self):

        self.video_cap = cv2.VideoCapture(self.movie_path)
        ret, self.video_frame = self.video_cap.read()
        #self.sliderlength=self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # フレームレートを取得
        fps = self.video_cap.get(cv2.CAP_PROP_FPS)

        # 動画の総フレーム数を取得
        total_frames = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # 動画の総時間（秒）を計算
        duration_sec = total_frames / fps

        self.scaleH.config(to=total_frames)

        if self.video_frame is None:
            print("None")

        while self.set_movie:

            if self.start_movie:

                ret, self.video_frame = self.video_cap.read()

                if ret:
                    # convert color order from BGR to RGB
                    #動画フレームに対してシークバーを移動
                    self.scaleH.set(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
                    #print(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
                    #動画時間に対してシークバーを移動
                    #self.scaleH.set(self.video_cap.get(cv2.CAP_PROP_POS_MSEC))
                    #print(self.video_cap.get(cv2.CAP_PROP_POS_MSEC))
                    pil = self.cvtopli_color_convert(self.video_frame)

                    self.effect_img, self.canvas_create = self.resize_image(
                        pil, self.canvas
                    )
                    self.replace_canvas_image(
                        self.effect_img, self.canvas, self.canvas_create
                    )
                else:
                    self.start_movie = False

    def run_one_frame(self):
        self.video_cap = cv2.VideoCapture(self.movie_path)
        ret, self.video_frame = self.video_cap.read()

        if self.video_frame is None:
            print("動画が見当たりません")

        else:
            ret, self.video_frame = self.video_cap.read()
            # convert color order from BGR to RGB
            pil = self.cvtopli_color_convert(self.video_frame)

            self.effect_img, self.canvas_create = self.resize_image(pil, self.canvas)
            # scale value intialize
            self.replace_canvas_image(self.effect_img, self.canvas, self.canvas_create)

    def replace_canvas_image(self, pic_img, canvas_name, canvas_name_create):
        canvas_name.photo = ImageTk.PhotoImage(pic_img)
        canvas_name.itemconfig(canvas_name_create, image=canvas_name.photo)

    def cvtopli_color_convert(self, video):
        rgb = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    # Model
    def resize_image(self, img, canvas):

        w = img.width
        h = img.height
        w_offset = 250 - (w * (500 / h) / 2)
        h_offset = 250 - (h * (700 / w) / 2)

        if w > h:
            resized_img = img.resize((int(w * (700 / w)), int(h * (700 / w))))
        else:
            resized_img = img.resize((int(w * (500 / h)), int(h * (500 / h))))

        self.pil_img = ImageTk.PhotoImage(resized_img)
        canvas.delete("can_pic")

        if w > h:
            resized_img_canvas = canvas.create_image(
                0, h_offset, anchor="nw", image=self.pil_img, tag="can_pic"
            )

        else:
            resized_img_canvas = canvas.create_image(
                w_offset, 0, anchor="nw", image=self.pil_img, tag="can_pic"
            )

        return resized_img, resized_img_canvas

    def get_path(self):
        return filedialog.askopenfilename(
            title="Please select image file,", filetypes=self.file_filter
        )
    
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================

#create_groupからの継承。
class create_group_plt(create_group):
    def on_click_path(self):
        #self.movie_path = self.get_path()
        self.movie_path = "./outmovie/animation.mp4"
        self.path_stvar.set(self.movie_path)
        self.run_one_frame()
        # Movie standby.
        self.thread_set = True
        self.thread_main = threading.Thread(target=self.main_thread_func)
        self.thread_main.start()



   

#＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

class Set_gui():
    def __init__(self, main_window):


        # メインウインドウ
        self.main_window = main_window
        self.main_window.geometry("2000x800")
        self.main_window.title("動画内の行動比較システム")
        self.canvas= tk.Canvas(self.main_window)
        self.main_frame = tk.Frame(self.canvas)

        # Canvasを親とした縦方向のScrollbar
        self.scrollbar = tk.Scrollbar(
            self.canvas, orient=tk.VERTICAL, command=self.canvas.yview
        )
        # Frameを親とした動作確認用のLabel
        label = tk.Label(self.main_frame, text="動作確認用ラベル")

        # スクロールの設定
        self.canvas.configure(scrollregion=(0, 0, 900, 1500))
        self.canvas.configure(yscrollcommand=self.scrollbar.set)


        #動画ウインドウ1（左）
        self.frame1=tk.Frame(self.main_frame)
        #動画ウインドウ2（右）
        self.frame2=tk.Frame(self.main_frame)

        #matplotlibウインドウ（中央）
        self.frame_plt=tk.Frame(self.main_frame)

        #点数表示用テキスト
        
        self.scoretext = tk.Label(self.main_frame, text="スコア表示位置")
        self.scoretext.pack(side=tk.BOTTOM)

        #得点予想処理ボタン（下）
        self.estimate = tk.Button(self.main_frame,  text="得点を予想",width=20, command=self.get_entry)
        self.estimate.pack(side=tk.BOTTOM)

        # 数値入力用の入力エントリーを作成
        self.entry = tk.Entry(self.main_frame,width=20)
        self.entry.pack(side=tk.BOTTOM)


        #比較処理ボタン（下）
        self.cbutton=tk.Button(self.main_frame, text="比較", width=10, command=self.comp)
        self.cbutton.pack(side=tk.BOTTOM)

        #変換処理ボタン（下）
        self.mbutton=tk.Button(self.main_frame, text="姿勢推定", width=10, command=self.mediapipe_start)
        self.mbutton.pack(side=tk.BOTTOM)



        button = tk.Button(self.main_frame, text='予測送信', command=self.get_entry)
        button.pack(side=tk.BOTTOM)


        
        self.frame_plt.pack(side=tk.BOTTOM)
        #スライダーの長さ
        self.sliderlength=100
        self.path_stvar = tk.StringVar()
        #表示＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
        self.movie1=create_group(self,self.frame1)
        self.movie2=create_group(self,self.frame2)

        #各動画のシークバーの現在地を取得
        self.movie1_frame=self.movie1.scaleH.get()
        self.movie2_frame=self.movie2.scaleH.get()



        #比較動画用
        self.frame_plt=create_group_plt(self,self.frame_plt)
        self.movie1.canvas_frame.pack()
        self.movie2.canvas_frame.pack()
        self.frame_plt.canvas_frame.pack()
        

        self.frame1.pack(side=tk.LEFT, padx=0)
        self.frame2.pack(side=tk.LEFT, padx=1)
    
        # Canvas上の座標(0, 0)に対してFrameの左上（nw=north-west）をあてがうように、Frameを埋め込む
        self.canvas.create_window((0, 0), window=self.main_frame, anchor="nw", width=900, height=900)

        # 諸々を配置
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        label.pack(expand=1)
    

    def on_configure(self,event):
        """Canvasのスクロール領域を調整するコールバック関数"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


    def mediapipe_start(self):
        print("姿勢推定の処理を開始します。")
        #動画パスを取得
        movie_path1=self.movie1.movie_path
        movie_path2=self.movie2.movie_path

        #ターミナルコマンド入力
        #movie1をマスターデータとしてデータベース化
        subprocess.call(["python3", "sample_holistic_get1.py" ,movie_path1 ,"-m" ,"1"])

        #movie2をサブデータとしてデータベース化
        subprocess.call(["python3", "sample_holistic_get1.py" ,movie_path2 ,"-m" ,"0"])

        print("姿勢推定の処理を終了しました。")
    
    def comp(self):
        print("比較処理を開始します。")
        self.train_data=comparison()

        print("比較処理を終了しました。")

    #点数評価＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    def get_entry(self):
        try:
            # 入力された値を取得
            value = int(self.entry.get())
            # 入力が正しい場合の処理
            if value<=100:
                #self.scoretext["text"]="点数："+str(value)
                score_est=np.array([self.train_data]*len(self.train_data)) #ユーザーの推定したスコア
                score=ScoreLearning(np.array(self.train_data),score_est)
                self.scoretext["text"]="点数："+str(score)

        except ValueError:
            # 入力が数値ではない場合の処理
            messagebox.showerror("エラー", "数値を入力してください")

def main():

    # 　Tk MainWindow
    main_window = tk.Tk()
    # ウィンドウを閉じた際にPythonを終了する
    def close_window():
        main_window.destroy()
        main_window.quit()
    # Viewクラス生成
    Set_gui(main_window)
    # ウィンドウを閉じるイベントの設定
    main_window.protocol("WM_DELETE_WINDOW", close_window)
    # 　フレームループ処理
    main_window.mainloop()


if __name__ == "__main__":
    main()

