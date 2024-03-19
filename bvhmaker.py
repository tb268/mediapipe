# Re-creating the complex BVH file with comments in Japanese

# Defining the structure of the BVH file with Japanese comments
#bvhファイル作成用システム

#blenderのインポートエラーで「could not convert string to float: Zrotation」が出る時→中括弧などX改行が原因


# coding: utf-8
import sqlite3
import numpy as np
import math
import itertools

#np配列をスペースありのstring文字列に変換
def np2str(array):
    for i in range(len(array)):
        array[i]=round(array[i], 3)
        
    #array[1],array[2]=array[2],array[1]

    out=" ".join(map(str,array))
    
    return out


def rotminus(x,y):
    
    a=[0]*len(x)
    for i in range(len(x)):
        a[i]=[x[i][0]-y[i][0],x[i][1]-y[i][1],x[i][2]-y[i][2]]

    return a

#np配列をスペースありのstring文字列に変換
def np2strForList(array):
    out=[]
    
    for arr in array:
        out.append(" ".join(map(str,arr)))
    out=" ".join(map(str,out))
    return out




#ベクトル間の角度を取得
def rotxyz(vec1,vec2):
    
    #各2次元ベクトル
    vec1xy=np.array([vec1[0],vec1[1],0])
    vec1yz=np.array([0,vec1[1],vec1[2]])
    vec1zx=np.array([vec1[0],0,vec1[2]])
    vec2xy=np.array([vec2[0],vec2[1],0])
    vec2yz=np.array([0,vec2[1],vec2[2]])
    vec2zx=np.array([vec2[0],0,vec2[2]])
    #各長さ
    vec1xyLen=np.linalg.norm(vec1xy)
    vec1yzLen=np.linalg.norm(vec1yz)
    vec1zxLen=np.linalg.norm(vec1zx)
    vec2xyLen=np.linalg.norm(vec2xy)
    vec2yzLen=np.linalg.norm(vec2yz)
    vec2zxLen=np.linalg.norm(vec2zx)
    #各内積
    xInner=np.dot(vec1yz,vec2yz)
    yInner=np.dot(vec1zx,vec2zx)
    zInner=np.dot(vec1xy,vec2xy)

    #各角度
    #ゼロ除算対策
    with np.errstate(all='ignore'):
        xrot=(xInner/(vec1yzLen*vec2yzLen))
        yrot=(yInner/(vec1zxLen*vec2zxLen))
        zrot=(zInner/(vec1xyLen*vec2xyLen))

    if math.isnan(xrot):
        xrot=0
    if math.isnan(yrot):
        yrot=0
    if math.isnan(zrot):
        zrot=0
    
    xDeg=math.degrees(xrot)
    yDeg=math.degrees(yrot)
    zDeg=math.degrees(zrot)
    xDeg=round(xDeg, 3)
    yDeg=round(yDeg, 3)
    zDeg=round(zDeg, 3)
    #帰り値は全てfloat型
    return xDeg,yDeg,zDeg

 

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

VectorUp=np.array([0,0,0])#上ベクトル

Motions=[]
sub=[]
#フレームごとに取得
for index, landmarks in enumerate(frames):

    landmark=eval(landmarks[1])
    
    #位置の設定。（L,Rは絶対位置、Left,Rightは相対位置）
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
    

    #1フレーム目であれば
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
    else:
        #1フレーム前の角度を取得
        Brfore_rot_Lists=rot_Lists.copy()
        #腰から上半身、右足、左足の三つに分かれる
        #左足===============================================================
        bLeftHip=LeftHip#左腰
        bLeftKnee=LeftKnee#左ひざ
        bLeftAnkle=LeftAnkle#左足首
        bLeftHeel=LeftHeel#左かかと
        bLeftToes=LeftToes#左つま先
        #右足================================================================
        bRightHip=RightHip#右腰
        bRightKnee=RightKnee#右ひざ
        bRightAnkle=RightAnkle#右足首
        bRightHeel=RightHeel#右かかと
        bRightToes=RightToes#右つま先
        #上半身================================================================
        bHeart=Heart#胸（相対位置）
        bNose=Nose#鼻（相対位置）
        #左腕
        bLeftShoulder=LeftShoulder#左肩
        bLeftElbow=LeftElbow#左ひじ
        bLeftWrist=LeftWrist#左手首
        bLeftThumb=LeftThumb#左親指
        bLeftIndex=LeftIndex#左人差し指
        bLeftPinky=LeftPinky#左小指
        #右腕
        bRightShoulder=RightShoulder#右肩
        bRightElbow=RightElbow#右ひじ
        bRightWrist=RightWrist#右手首
        bRightThumb=RightThumb#右親指
        bRightIndex=RightIndex#右人差し指
        bRightPinky=RightPinky#右小指

        

    root=[0,0,0,0,0,0]#座標、回転
    #ベクトル計算
    #腰から上半身、右足、左足の三つに分かれる
    #左足================================================================
    LeftHip=LHip-Base#左腰
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
    RightShoulder=RShoulder-Heart#右肩
    RightElbow=RElbow-RShoulder#右ひじ
    RightWrist=RWrist-RElbow#右手首
    RightThumb=RThumb-RWrist#右親指
    RightIndex=RIndex-RWrist#右人差し指
    RightPinky=RPinky-RWrist#右小指

    #角度
    #左下半身
    LeftHip_rot=rotxyz(VectorUp,LeftHip)
    LeftKnee_rot=rotxyz(LeftHip,LeftKnee)
    LeftAnkle_rot=rotxyz(LeftKnee,LeftAnkle)
    LeftToes_rot=rotxyz(LeftAnkle,LeftToes)
    LeftHeel_rot=rotxyz(LeftAnkle,LeftHeel)
    #右下半身
    RightHip_rot=rotxyz(VectorUp,RightHip)
    RightKnee_rot=rotxyz(RightHip,RightKnee)
    RightAnkle_rot=rotxyz(RightKnee,RightAnkle)
    RightToes_rot=rotxyz(RightAnkle,RightToes)
    RightHeel_rot=rotxyz(RightToes,RightHeel)
    #上半身
    Heart_rot=rotxyz(VectorUp,Heart)
    Nose_rot=rotxyz(Heart,Nose)
    #左上半身
    LeftShoulder_rot=rotxyz(Heart,LeftShoulder)
    LeftElbow_rot=rotxyz(LeftShoulder,LeftElbow)
    LeftWrist_rot=rotxyz(LeftElbow,LeftWrist)
    LeftThumb_rot=rotxyz(LeftWrist,LeftThumb)
    LeftIndex_rot=rotxyz(LeftWrist,LeftIndex)
    LeftPinky_rot=rotxyz(LeftWrist,LeftPinky)
    #右上半身
    RightShoulder_rot=rotxyz(Heart,RightShoulder)
    RightElbow_rot=rotxyz(RightShoulder,RightElbow)
    RightWrist_rot=rotxyz(RightElbow,RightWrist)
    RightThumb_rot=rotxyz(RightWrist,RightThumb)
    RightIndex_rot=rotxyz(RightWrist,RightIndex)
    RightPinky_rot=rotxyz(RightWrist,RightPinky)


    rot_Lists=[
            LeftHip_rot,LeftKnee_rot,LeftAnkle_rot,LeftHeel_rot,LeftToes_rot,
            RightHip_rot,RightKnee_rot,RightAnkle_rot,RightHeel_rot,RightToes_rot,
            RightShoulder_rot,RightElbow_rot,RightWrist_rot,RightPinky_rot,RightThumb_rot,
            Heart_rot,Nose_rot,
            LeftShoulder_rot,LeftElbow_rot,LeftWrist_rot,LeftPinky_rot,LeftThumb_rot
        ]

    """
    #モーション作成
    Motions.append(np2strForList([
                    np.array([0,0,0]),np.array([0,0,0]),
                    LeftHip_rot,LeftKnee_rot,LeftAnkle_rot,LeftHeel_rot,LeftToes_rot,
                    RightHip_rot,RightKnee_rot,RightAnkle_rot,RightHeel_rot,RightToes_rot,
                    RightShoulder_rot,RightElbow_rot,RightWrist_rot,RightPinky_rot,RightThumb_rot,
                    Heart_rot,Nose_rot,
                    LeftShoulder_rot,LeftElbow_rot,LeftWrist_rot,LeftPinky_rot,LeftThumb_rot
                    ]))
    """
    if not index==0:

        #モーション作成
        Motions.append(np2strForList([
            np.array([0,0,0]),np.array([0,0,0]),
            list(itertools.chain.from_iterable(rotminus(rot_Lists,Brfore_rot_Lists)))
        ]))
    else:
        #モーション作成(1フレーム目)
        Motions.append(np2strForList([
            np.array([0,0,0]),np.array([0,0,0]),
            list(itertools.chain.from_iterable(rotminus(rot_Lists,rot_Lists)))
        ]))





    





#conn.close() 
bvh_content_complex_japanese = f"""
HIERARCHY
ROOT Base
{{
    OFFSET 0.00 0.00 0.00 	
    CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
    JOINT LeftHip
    {{
        OFFSET 0.00 0.00 0.00
        CHANNELS 3 Zrotation Yrotation Xrotation
        JOINT LeftKnee
        {{
            OFFSET {np2str(LeftHip_origin)}
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT LeftAnkle
            {{
                OFFSET {np2str(LeftKnee_origin)}
                CHANNELS 3 Zrotation Yrotation Xrotation
                JOINT LeftHeel
                    {{
                        OFFSET {np2str(LeftAnkle_origin)}
                        CHANNELS 3 Zrotation Yrotation Xrotation
                        End Site
                        {{
                            OFFSET {np2str(LeftHeel_origin)}
                        }}
                    }}
                JOINT LeftToes
                {{
                    OFFSET {np2str(LeftAnkle_origin)}
                    CHANNELS 3 Zrotation Yrotation Xrotation
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
        CHANNELS 3 Zrotation Yrotation Xrotation
        JOINT RightKnee
        {{
            OFFSET {np2str(RightHip_origin)}
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT RightAnkle
            {{
                OFFSET {np2str(RightKnee_origin)}
                CHANNELS 3 Zrotation Yrotation Xrotation
                JOINT RightHeel
                {{
                    OFFSET {np2str(RightAnkle_origin)}
                    CHANNELS 3 Zrotation Yrotation Xrotation
                    End Site
                    {{
                        OFFSET {np2str(RightHeel_origin)}
                    }}
                }}
                JOINT RightToes
                {{
                    OFFSET {np2str(RightAnkle_origin)}
                    CHANNELS 3 Zrotation Yrotation Xrotation
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
        CHANNELS 3 Zrotation Yrotation Xrotation
        JOINT RightShoulder
        {{
            OFFSET {np2str(Heart_origin)}
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT RightElbow
            {{
                OFFSET {np2str(RightShoulder_origin)}
                CHANNELS 3 Zrotation Yrotation Xrotation
                JOINT RightWrist
                {{
                    OFFSET {np2str(RightElbow_origin)}
                    CHANNELS 3 Zrotation Yrotation Xrotation
                    JOINT RightPinky
                    {{
                        OFFSET {np2str(RightWrist_origin)}
                        CHANNELS 3 Zrotation Yrotation Xrotation
                        End Site
                        {{
                            OFFSET {np2str(RightPinky_origin)}
                        }}
                    }}
                    JOINT RightThumb
                    {{
                        OFFSET {np2str(RightWrist_origin)}
                        CHANNELS 3 Zrotation Yrotation Xrotation
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
            CHANNELS 3 Zrotation Yrotation Xrotation
            End Site
            {{
                OFFSET {np2str(Nose_origin)}
            }}
        }}
        JOINT LeftShoulder
        {{
            OFFSET {np2str(Heart_origin)}
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT LeftElbow
            {{
                OFFSET {np2str(LeftShoulder_origin)}
                CHANNELS 3 Zrotation Yrotation Xrotation
                JOINT LeftWrist
                {{
                    OFFSET {np2str(LeftElbow_origin)}
                    CHANNELS 3 Zrotation Yrotation Xrotation
                    JOINT LeftPinky
                    {{
                        OFFSET {np2str(LeftWrist_origin)}
                        CHANNELS 3 Zrotation Yrotation Xrotation
                        End Site
                        {{
                            OFFSET {np2str(LeftPinky_origin)}
                        }}
                    }}
                    JOINT LeftThumb
                    {{
                        OFFSET {np2str(LeftWrist_origin)}
                        CHANNELS 3 Zrotation Yrotation Xrotation
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



#BVHファイル書き込み
bvh_complex_japanese_file_path = './BVHfile/test.bvh'
with open(bvh_complex_japanese_file_path, 'w') as file:
    file.write(bvh_content_complex_japanese.strip())
    file.write("\n")#改行

    for i in Motions:
        file.write(i)
        file.write("\n")#改行

        pass



