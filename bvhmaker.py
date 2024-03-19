# Re-creating the complex BVH file with comments in Japanese

# Defining the structure of the BVH file with Japanese comments
#bvhファイル作成用システム

#blenderのインポートエラーで「could not convert string to float: Xrotation」が出る時→中括弧などの改行が原因


# coding: utf-8
import sqlite3
import numpy as np

#numpy配列をスペースありのstring文字列に変換
def np2str(array):
    out=" ".join(map(str,array))
    return out

#xyzの角度を取得
def rotxyz(vec1,vec2):
    vec1-vec2
    return

#DB取得
conn = sqlite3.connect(r'main.db')
#カーソル作成
c = conn.cursor()
#テーブルを指定しデータを取得
c.execute("select * from pose")

#データベース取得
#poselistはフレームごとの情報=====================
#各フレームには各関節の情報
#各関節には[信頼度,(x座標,y座標,z座標)]の情報が入る========

frames = c.fetchall()

#フレームごとに取得
for index, landmarks in enumerate(frames):
    print(landmarks)
    landmark=eval(landmarks[1])
    #位置の設定。（L,Rは絶対位置、Left,Rightは相対位置）
    print ((landmark))
    #print (len(landmark))
    #print ((landmark[0]))
    #部位ごとに取得＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    NoseAbsol=np.array(landmark[0][1])# 鼻
    LHip=np.array(landmark[23][1])# 腰(左側)
    RHip=np.array(landmark[24][1])# 腰(右側)
    Base=((LHip+RHip)/2)#腰(左右腰の中間)
    #上半身ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    LShoulder=np.array(landmark[11][1])# 左肩
    RShoulder=np.array(landmark[12][1])# 右肩
    LElbow=np.array(landmark[13][1])# 左肘
    RElbow=np.array(landmark[14][1])# 右肘
    LWrist=np.array(landmark[15][1])# 左手首
    RWrist=np.array(landmark[16][1])# 右手首
    LPinky=np.array(landmark[17][1])# 左子指
    RPinky=np.array(landmark[18][1])# 右子指
    LIndex=np.array(landmark[19][1])# 左人差し指
    RIndex=np.array(landmark[20][1])# 左人差し指
    LThumb=np.array(landmark[21][1])# 左親指
    RThumb=np.array(landmark[22][1])# 右親指
    #下半身ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    LKnee=np.array(landmark[25][1])# 左ひざ
    RKnee=np.array(landmark[26][1])# 右ひざ
    LAnkle=np.array(landmark[27][1])# 左足首
    RAnkle=np.array(landmark[28][1])# 右足首
    LHeel=np.array(landmark[29][1])# 左かかと
    RHeel=np.array(landmark[30][1])# 右かかと
    LToes=np.array(landmark[31][1])# 左つま先
    RToes=np.array(landmark[32][1])# 右つま先
    HeartAbsol=((LShoulder+RShoulder)/2)#胸（絶対位置）

    if index==0:#初期オフセット設定

        #腰から上半身、右足、左足の三つに分かれる
        #左足================================================================
        LeftHip_origin=LHip-Base#左腰
        LeftKnee_origin=LKnee-LHip#左ひざ
        LeftAnkle_origin=LAnkle-LKnee#左足首
        LeftHeel_origin=LHeel-LAnkle#左かかと
        LeftToes_origin=LToes-LAnkle#左つま先
        #右足================================================================
        RightHip_origin=RHip-Base#右腰
        RightKnee_origin=RKnee-RHip#右ひざ
        RightAnkle_origin=RAnkle-RKnee#右足首
        RightHeel_origin=RHeel-RAnkle#右かかと
        RightToes_origin=RToes-RAnkle#右つま先
        #上半身================================================================
        Heart_origin=HeartAbsol-Base#胸（相対位置）
        Nose_origin=NoseAbsol-HeartAbsol#鼻（相対位置）
        #左腕
        LeftShoulder_origin=LShoulder-Heart_origin#左肩
        LeftElbow_origin=LElbow-LShoulder#左ひじ
        LeftWrist_origin=LWrist-LElbow#左手首
        LeftThumb_origin=LThumb-LWrist#左親指
        LeftIndex_origin=LIndex-LWrist#左人差し指
        LeftPinky_origin=LPinky-LWrist#左小指
        #右腕
        RightShoulder_origin=RShoulder-Heart_origin#左肩
        RightElbow_origin=RElbow-RShoulder#左ひじ
        RightWrist_origin=RWrist-RElbow#左手首
        RightThumb_origin=RThumb-RWrist#左親指
        RightIndex_origin=RIndex-RWrist#左人差し指
        RightPinky_origin=RPinky-RWrist#左小指
    else:#初期以降の角度取得
        #腰から上半身、右足、左足の三つに分かれる
        #左足================================================================
        LeftHip_origin=LHip-Base#左腰
        LeftKnee=LKnee-LHip#左ひざ
        LeftAnkle=LAnkle-LKnee#左足首
        LeftHeel=LHeel-LAnkle#左かかと
        LeftToes=LToes-LAnkle#左つま先
        #右足================================================================
        RightHip=RHip-Base#右腰
        RightKnee=RKnee-RHip#右ひざ
        RightAnkle=RAnkle-RKnee#右足首
        RightHeel=RHeel-RAnkle#右かかと
        RightToes=RToes-RAnkle#右つま先
        #上半身================================================================
        Heart=HeartAbsol-Base#胸（相対位置）
        Nose=NoseAbsol-HeartAbsol#鼻（相対位置）
        #左腕
        LeftShoulder=LShoulder-Heart#左肩
        LeftElbow=LElbow-LShoulder#左ひじ
        LeftWrist=LWrist-LElbow#左手首
        LeftThumb=LThumb-LWrist#左親指
        LeftIndex=LIndex-LWrist#左人差し指
        LeftPinky=LPinky-LWrist#左小指
        #右腕
        RightShoulder=RShoulder-Heart#左肩
        RightElbow=RElbow-RShoulder#左ひじ
        RightWrist=RWrist-RElbow#左手首
        RightThumb=RThumb-RWrist#左親指
        RightIndex=RIndex-RWrist#左人差し指
        RightPinky=RPinky-RWrist#左小指
        """

    




    




#print (len(frames)) 
#print (frames[0]) 


#conn.close() 
a="ああああ"
bvh_content_complex_japanese = f"""
HIERARCHY
ROOT Hip
{{
    OFFSET 0.00 0.00 0.00 	
    CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation
    JOINT LeftHip_origin
    {{
        OFFSET 0.00 0.00 0.00
        CHANNELS 3 Xrotation Yrotation Zrotation
        JOINT LeftKnee
        {{
            OFFSET {np2str(LeftHip_origin)}
            CHANNELS 3 Xrotation Yrotation Zrotation
            JOINT LeftAnkle
            {{
                OFFSET {np2str(LeftKnee_origin)}
                CHANNELS 3 Xrotation Yrotation Zrotation
                JOINT LeftHeel
                    {{
                        OFFSET {np2str(LeftAnkle_origin)}
                        CHANNELS 3 Xrotation Yrotation Zrotation
                        End Site
                        {{
                            OFFSET {np2str(LeftHeel_origin)}
                        }}
                    }}
                JOINT LeftToes
                {{
                    OFFSET {np2str(LeftAnkle_origin)}
                    CHANNELS 3 Xrotation Yrotation Zrotation
                    End Site
                    {{
                        OFFSET {np2str(LeftToes_origin)}
                    }}
                }}
            }}
        }}
    }}
    JOINT RightHip
    {{
        OFFSET 0.00 0.00 0.00
        CHANNELS 3 Xrotation Yrotation Zrotation
        JOINT RightKnee
        {{
            OFFSET {np2str(RightHip_origin)}
            CHANNELS 3 Xrotation Yrotation Zrotation
            JOINT RightAnkle
            {{
                OFFSET {np2str(RightKnee_origin)}
                CHANNELS 3 Xrotation Yrotation Zrotation
                JOINT RightHeel
                {{
                    OFFSET {np2str(RightAnkle_origin)}
                    CHANNELS 3 Xrotation Yrotation Zrotation
                    End Site
                    {{
                        OFFSET {np2str(RightHeel_origin)}
                    }}
                }}
                JOINT RightToes
                {{
                    OFFSET {np2str(RightAnkle_origin)}
                    CHANNELS 3 Xrotation Yrotation Zrotation
                    End Site
                    {{
                        OFFSET {np2str(RightToes_origin)}
                    }}
                }}
            }}
        }}
    }}
    JOINT Heart
    {{
        OFFSET 0.00 0.00 0.00
        CHANNELS 3 Xrotation Yrotation Zrotation
        JOINT RightShoulder
        {{
            OFFSET {np2str(Heart_origin)}
            CHANNELS 3 Xrotation Yrotation Zrotation
            JOINT RightElbow
            {{
                OFFSET {np2str(RightShoulder_origin)}
                CHANNELS 3 Xrotation Yrotation Zrotation
                JOINT RightWrist
                {{
                    OFFSET {np2str(RightElbow_origin)}
                    CHANNELS 3 Xrotation Yrotation Zrotation
                    JOINT RightPinky
                    {{
                        OFFSET {np2str(RightWrist_origin)}
                        CHANNELS 3 Xrotation Yrotation Zrotation
                        End Site
                        {{
                            OFFSET {np2str(RightPinky_origin)}
                        }}
                    }}
                    JOINT RightThumb
                    {{
                        OFFSET {np2str(RightWrist_origin)}
                        CHANNELS 3 Xrotation Yrotation Zrotation
                        End Site
                        {{
                            OFFSET {np2str(RightThumb_origin)}
                        }}
                    }}
                }}
            }}
        }}
        JOINT Nose
        {{
            OFFSET {np2str(Heart_origin)}
            CHANNELS 3 Xrotation Yrotation Zrotation
            End Site
            {{
                OFFSET {np2str(Nose_origin)}
            }}
        }}
        JOINT LeftShoulder
        {{
            OFFSET {np2str(Heart_origin)}
            CHANNELS 3 Xrotation Yrotation Zrotation
            JOINT LeftElbow
            {{
                OFFSET {np2str(LeftShoulder_origin)}
                CHANNELS 3 Xrotation Yrotation Zrotation
                JOINT LeftWrist
                {{
                    OFFSET {np2str(LeftElbow_origin)}
                    CHANNELS 3 Xrotation Yrotation Zrotation
                    JOINT LeftPinky
                    {{
                        OFFSET {np2str(LeftWrist_origin)}
                        CHANNELS 3 Xrotation Yrotation Zrotation
                        End Site
                        {{
                            OFFSET {np2str(LeftPinky_origin)}
                        }}
                    }}
                    JOINT LeftThumb
                    {{
                        OFFSET {np2str(LeftWrist_origin)}
                        CHANNELS 3 Xrotation Yrotation Zrotation
                        {{
                            OFFSET {np2str(LeftThumb_origin)}
                        }}
                    }}
                }}
            }}
        }}
    }}
}}
MOTION
Frames: {len(frames)}
Frame Time: 0.0333333
"""



print(bvh_content_complex_japanese)

"""
# Adding the motion data with Japanese comments
for landmark in range(num_frames):
    hip_rotation = 10 * math.sin(2 * math.pi * (landmark / num_frames))  # 腰の揺れ
    head_rotation = 10 * math.cos(2 * math.pi * (landmark / num_frames))  # 頭の動き

    frame_data = f"0.00 0.00 0.00 {hip_rotation:.2f} 0.00 0.00 0.00 0.00 0.00 {hip_rotation:.2f} 0.00 0.00 0.00 0.00 {head_rotation:.2f} 0.00 0.00 0.00\n"
    bvh_content_complex_japanese += frame_data
"""

# Saving the content to a BVH file
#BVHファイル書き込み
bvh_complex_japanese_file_path = './BVHfile/test.bvh'
with open(bvh_complex_japanese_file_path, 'w') as file:
    file.write(bvh_content_complex_japanese.strip())


