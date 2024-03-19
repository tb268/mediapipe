# Re-creating the complex BVH file with comments in Japanese

# Defining the structure of the BVH file with Japanese comments
#bvhファイル作成用システム

#blenderのインポートエラーで「could not convert string to float: Zrotation」が出る時→中括弧などX改行が原因

#xzyの順番で座標が入っている、yが上
# coding: utf-8
import sqlite3
import numpy as np
import math
import matplotlib.pyplot as plt
import keyboard

#np配列をスペースありのstring文字列に変換
def np2str(array):
    for i in range(len(array)):
        array[i]=round(array[i], 3)
        
    #array[1],array[2]=array[2],array[1]

    out=" ".join(map(str,array))
    
    return out


def rotminus(pose_index_set,y):
    
    a=[0]*len(pose_index_set)
    for i in range(len(pose_index_set)):
        a[i]=[pose_index_set[i][0]-y[i][0],pose_index_set[i][1]-y[i][1],pose_index_set[i][2]-y[i][2]]

    return a

#np配列をスペースありのstring文字列に変換
def np2strForList(array):
    out=[]
    
    for arr in array:
        out.append(" ".join(map(str,arr)))
    out=" ".join(map(str,out))
    return out




#ベクトル間の角度を取得
def rotxyz(vec_before,vec_after):
    #各長さ
    vec1Len=np.linalg.norm(vec_before)
    vec2Len=np.linalg.norm(vec_after)
    #各内積
    xInner=np.dot(vec_before,vec_after)

    #各角度
    #ゼロ除算対策
    with np.errstate(all='ignore'):
        rot=(xInner/(vec1Len*vec2Len))
    
    Deg=math.degrees(rot)
    Deg=round(Deg, 3)

    #帰り値は全てfloat型
    return Deg


#ベクトル間の角度を取得（前方調整用）vec1からvec2に回転する
def rotxyz2(vec_before,vec_after):
    #縦を0にする
    #各長さ
    vec1Len=np.linalg.norm(vec_before)
    vec2Len=np.linalg.norm(vec_after)
    #各内積
    xInner=np.dot(vec_before,vec_after)

    #各角度
    #ゼロ除算対策
    #θを取得
    #1を超えると逆余弦が計算できないため、丸め込み
    
    rot=(xInner/(vec1Len*vec2Len))
    if rot>1:
        rot=1
    rot=math.acos(rot)
    deg=math.degrees(rot)
    print(xInner/(vec1Len*vec2Len))
    #Deg=math.degrees(rot)
    #Deg=round(Deg, 3)
    #回転方向を計算
    """
    回転方向を判定する

    Args:
        vec_before: 回転前のベクトル (numpy.ndarray)
        vec_after: 回転後のベクトル (numpy.ndarray)

    Returns:
        回転方向 (1: 反時計回り, -1: 時計回り)
    """
    vec_before2=np.array([vec_before[0],vec_before[2]])
    vec_after2=np.array([vec_after[0],vec_after[2]])

    # 外積を求める
    cross_product = np.cross(vec_before2, vec_after2)
    
    print(cross_product)
    # y成分が正なら反時計回り、負なら時計回り
    if cross_product > 0:
        r=1
    else:
        r=-1
    #帰り値は全てfloat型

    return deg,r

#vec_before,2の垂線を外積で取得
def get_forwardvec(vec_before,vec_after):

    out=np.cross(vec_before,vec_after)
    #高さは0で前方ベクトルを取得 
    out[1]=0
    return out

#y軸回転
def yrot(vec,deg,r):
    #c = math.cos(theta)
    #s = math.sin(theta*r)
    theta_x = math.radians(0)
    theta_y = math.radians(deg)
    theta_z = math.radians(0)
    rot_x = np.array([[ 1,                 0,                  0],
                      [ 0, math.cos(theta_x), -math.sin(theta_x)],
                      [ 0, math.sin(theta_x),  math.cos(theta_x)]])

    rot_y = np.array([[ math.cos(theta_y), 0,  math.sin(theta_y)],
                      [                 0, 1,                  0],
                      [-math.sin(theta_y), 0, math.cos(theta_y)]])

    rot_z = np.array([[ math.cos(theta_z), -math.sin(theta_z), 0],
                      [ math.sin(theta_z),  math.cos(theta_z), 0],
                      [                 0,                  0, 1]])
    
    vec_rotated=rot_z.dot(rot_y.dot(rot_x.dot(vec)))
    return vec_rotated

#ベクトル形成
def vectorSet(start,vec_in,vec_out):
    """
    start : 開始位置
    vec_in : 元のベクトル
    vec_out : 目的の長さ取得用ベクトル
    """

    out=start+((vec_in/np.linalg.norm(vec_in))*np.linalg.norm(vec_out))

    return out

#DB取得
conn = sqlite3.connect(r'main.db')
#カーソル作成
c1 = conn.cursor()
c2 = conn.cursor()

#テーブルを指定しデータを取得
c1.execute("select * from pose1")#教師データ
c2.execute("select * from pose2")#生徒データ

#データベース取得
#poselistはフレームごとの情報=====================
#各フレームには各関節の情報
#各関節には[信頼度,(x座標,y座標,z座標)]の情報が入る========
#テーブル呼び出し
frames1 = c1.fetchall()
frames2 = c2.fetchall()

VectorUp=np.array([0,0,0])#上ベクトル

Motions_1=[]
Motions_2=[]

#フレームごとに取得
for index, (landmarks1,landmarks2) in enumerate(zip(frames1,frames2)):

    landmark=eval(landmarks1[1])
    
    #教師の位置の設定。（L,Rは絶対位置、Left,Rightは相対位置）
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
    #ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    #前方ベクトルを取得
    get_forwardvec(LHip-HeartAbsol,RHip-HeartAbsol)

    #教師のベクトル取得
    #腰から上半身、右足、左足の三つに分かれる
    #左足================================================================
    LeftHip=(LHip-Base)#左腰
    LeftKnee=(LKnee-LHip)#左ひざ
    LeftAnkle=(LAnkle-LKnee)#左足首
    LeftHeel=(LHeel-LAnkle)#左かかと
    LeftToes=(LToes-LAnkle)#左つま先
    #右足================================================================
    RightHip=(RHip-Base)#右腰
    RightKnee=(RKnee-RHip)#右ひざ
    RightAnkle=(RAnkle-RKnee)#右足首
    RightHeel=(RHeel-RAnkle)#右かかと
    RightToes=(RToes-RAnkle)#右つま先
    #上半身================================================================
    Heart=(HeartAbsol-Base)#胸（相対位置）
    Nose=(NoseAbsol-HeartAbsol)#鼻（相対位置）
    #左腕
    LeftShoulder=(LShoulder-Heart)#左肩
    LeftElbow=(LElbow-LShoulder)#左ひじ
    LeftWrist=(LWrist-LElbow)#左手首
    LeftThumb=(LThumb-LWrist)#左親指
    LeftIndex=(LIndex-LWrist)#左人差し指
    LeftPinky=(LPinky-LWrist)#左小指
    #右腕
    RightShoulder=(RShoulder-Heart)#右肩
    RightElbow=(RElbow-RShoulder)#右ひじ
    RightWrist=(RWrist-RElbow)#右手首
    RightThumb=(RThumb-RWrist)#右親指
    RightIndex=(RIndex-RWrist)#右人差し指
    RightPinky=(RPinky-RWrist)#右小指
    #ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
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

    #ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    #ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    #ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    #ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー

    landmark=eval(landmarks2[1])

    
    #教師の位置の設定。（L,Rは絶対位置、Left,Rightは相対位置）
    #部位ごとに取得＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    NoseAbsol2=np.array(landmark[0][1])# 鼻
    LHip2=np.array(landmark[23][1])# 腰(左側)
    RHip2=np.array(landmark[24][1])# 腰(右側)
    Base2=((LHip2+RHip2)/2)#腰(左右腰の中間)
    #上半身ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    LShoulder2=np.array(landmark[11][1])# 左肩
    RShoulder2=np.array(landmark[12][1])# 右肩
    LElbow2=np.array(landmark[13][1])# 左肘
    RElbow2=np.array(landmark[14][1])# 右肘
    LWrist2=np.array(landmark[15][1])# 左手首
    RWrist2=np.array(landmark[16][1])# 右手首
    LPinky2=np.array(landmark[17][1])# 左子指
    RPinky2=np.array(landmark[18][1])# 右子指
    LIndex2=np.array(landmark[19][1])# 左人差し指
    RIndex2=np.array(landmark[20][1])# 左人差し指
    LThumb2=np.array(landmark[21][1])# 左親指
    RThumb2=np.array(landmark[22][1])# 右親指
    #下半身ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    LKnee2=np.array(landmark[25][1])# 左ひざ
    RKnee2=np.array(landmark[26][1])# 右ひざ
    LAnkle2=np.array(landmark[27][1])# 左足首
    RAnkle2=np.array(landmark[28][1])# 右足首
    LHeel2=np.array(landmark[29][1])# 左かかと
    RHeel2=np.array(landmark[30][1])# 右かかと
    LToes2=np.array(landmark[31][1])# 左つま先
    RToes2=np.array(landmark[32][1])# 右つま先
    HeartAbsol2=((LShoulder2+RShoulder2)/2)#胸（絶対位置）
    #ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    #前方ベクトルを取得
    if index==0:
        vec_after=get_forwardvec(LHip-HeartAbsol,RHip-HeartAbsol)
        vec_before=get_forwardvec(LHip2-HeartAbsol2,RHip2-HeartAbsol2)

        #回転角、回転方向
        deg,turn_muki=rotxyz2(vec_before,vec_after)
        

    #回転ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    #部位ごとに取得＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    veclists=[
                NoseAbsol2,LHip2,RHip2,Base2,
              LShoulder2,RShoulder2,LElbow2,RElbow2,LWrist2,RWrist2,
              LPinky2,RPinky2,LIndex2,RIndex2,LThumb2,RThumb2,
              LKnee2,RKnee2,LAnkle2,RAnkle2,
              LHeel2,RHeel2,LToes2,RToes2,HeartAbsol2
              ]
    
    for i in range(len(veclists)):
        veclists[i]=yrot(veclists[i],deg,turn_muki)
        
    
    
    NoseAbsol2=veclists[0]# 鼻
    LHip2=veclists[1]# 腰(左側)
    RHip2=veclists[2]# 腰(右側)
    Base2=veclists[3]#腰(左右腰の中間)
    #上半身ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    LShoulder2=veclists[4]# 左肩
    RShoulder2=veclists[5]# 右肩
    LElbow2=veclists[6]# 左肘
    RElbow2=veclists[7]# 右肘
    LWrist2=veclists[8]# 左手首
    RWrist2=veclists[9]# 右手首
    LPinky2=veclists[10]# 左子指
    RPinky2=veclists[11]# 右子指
    LIndex2=veclists[12]# 左人差し指
    RIndex2=veclists[13]# 左人差し指
    LThumb2=veclists[14]# 左親指
    RThumb2=veclists[15]# 右親指
    #下半身ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    LKnee2=veclists[16]# 左ひざ
    RKnee2=veclists[17]# 右ひざ
    LAnkle2=veclists[18]# 左足首
    RAnkle2=veclists[19]# 右足首
    LHeel2=veclists[20]# 左かかと
    RHeel2=veclists[21]# 右かかと
    LToes2=veclists[22]# 左つま先
    RToes2=veclists[23]# 右つま先
    HeartAbsol2=veclists[24]#胸（絶対位置）








    #生徒のベクトル取得
    #腰から上半身、右足、左足の三つに分かれる
    #左足================================================================
    LeftHip2=(LHip2-Base2)#左腰
    LeftKnee2=(LKnee2-LHip2)#左ひざ
    LeftAnkle2=(LAnkle2-LKnee2)#左足首
    LeftHeel2=(LHeel2-LAnkle2)#左かかと
    LeftToes2=(LToes2-LAnkle2)#左つま先
    #右足================================================================
    RightHip2=(RHip2-Base2)#右腰
    RightKnee2=(RKnee2-RHip2)#右ひざ
    RightAnkle2=(RAnkle2-RKnee2)#右足首
    RightHeel2=(RHeel2-RAnkle2)#右かかと
    RightToes2=(RToes2-RAnkle2)#右つま先
    #上半身================================================================
    Heart2=(HeartAbsol2-Base2)#胸（相対位置）
    Nose2=(NoseAbsol2-HeartAbsol2)#鼻（相対位置）
    #左腕
    LeftShoulder2=(LShoulder2-Heart2)#左肩
    LeftElbow2=(LElbow2-LShoulder2)#左ひじ
    LeftWrist2=(LWrist2-LElbow2)#左手首
    LeftThumb2=(LThumb2-LWrist2)#左親指
    LeftIndex2=(LIndex2-LWrist2)#左人差し指
    LeftPinky2=(LPinky2-LWrist2)#左小指
    #右腕
    RightShoulder2=(RShoulder2-Heart2)#右肩
    RightElbow2=(RElbow2-RShoulder2)#右ひじ
    RightWrist2=(RWrist2-RElbow2)#右手首
    RightThumb2=(RThumb2-RWrist2)#右親指
    RightIndex2=(RIndex2-RWrist2)#右人差し指
    RightPinky2=(RPinky2-RWrist2)#右小指

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
    #ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    #ーーーーーーーーーーーーーーーーーーーーーーーーー
    #ーーーーーーーーーーーーーーーーーーーーーーーーー
    #ーーーーーーーーーーーーーーーーーーーーーーーーー
    #ーーーーーーーーーーーーーーーーーーーーーーーーー
    #ーーーーーーーーーーーーーーーーーーーーーーーーー
    #ーーーーーーーーーーーーーーーーーーーーーーーーー
    #ーーーーーーーーーーーーーーーーーーーーーーーーー
    #ーーーーーーーーーーーーーーーーーーーーーーーーー
    #骨格形成
    Base_out=((LHip+RHip)/2)#腰(左右腰の中間)
    LHip_out=vectorSet(Base_out,LeftHip2,LeftHip2)# 腰(左側)
    RHip_out=vectorSet(Base_out,RightHip2,RightHip)# 腰(右側)
    Heart_out=vectorSet(Base_out,Heart2,Heart)#胸（絶対位置）
    Nose_out=vectorSet(Heart_out,Nose2,Nose)# 鼻
    #上半身ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    LShoulder_out=vectorSet(Heart_out,LeftShoulder2,LeftShoulder)# 左肩
    RShoulder_out=vectorSet(Heart_out,RightShoulder2,RightShoulder)# 右肩
    LElbow_out=vectorSet(LShoulder_out,LeftElbow2,LeftElbow)# 左肘
    RElbow_out=vectorSet(RShoulder_out,RightElbow2,RightElbow)# 右肘
    LWrist_out=vectorSet(LElbow_out,LeftWrist2,LeftWrist)# 左手首
    RWrist_out=vectorSet(RElbow_out,RightWrist2,RightWrist)# 右手首

    LPinky_out=vectorSet(LWrist_out,LeftPinky2,LeftPinky)# 左子指
    RPinky_out=vectorSet(RWrist_out,RightPinky2,RightPinky)# 右子指
    LIndex_out=vectorSet(LWrist_out,LeftIndex2,LeftIndex)# 左人差し指
    RIndex_out=vectorSet(RWrist_out,RightIndex2,RightIndex)# 左人差し指
    LThumb_out=vectorSet(LWrist_out,LeftThumb2,LeftThumb)# 左親指
    RThumb_out=vectorSet(RWrist_out,RightThumb2,RightThumb)# 右親指
    #下半身ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    LKnee_out=vectorSet(LHip_out,LeftKnee2,LeftKnee)# 左ひざ
    RKnee_out=vectorSet(RHip_out,RightKnee2,RightKnee)# 右ひざ
    LAnkle_out=vectorSet(LKnee_out,LeftAnkle2,LeftAnkle)# 左足首
    RAnkle_out=vectorSet(RKnee_out,RightAnkle2,RightAnkle)# 右足首
    LHeel_out=vectorSet(LAnkle_out,LeftHeel2,LeftHeel) #左かかと
    RHeel_out=vectorSet(RAnkle_out,RightHeel2,RightHeel)# 右かかと
    LToes_out=vectorSet(LAnkle_out,LeftToes2,LeftToes)# 左つま先
    RToes_out=vectorSet(RAnkle_out,RightToes2,RightToes)# 右つま先
    
    #ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    #教師データをリスト化
    out_list_1=[
            Base,
              LHip,LKnee,LAnkle,LToes,LHeel,
              RHip,RKnee,RAnkle,RToes,RHeel,
              Heart,Nose,
              LShoulder,LElbow,LWrist,LThumb,LIndex,LPinky,
              RShoulder,RElbow,RWrist,RThumb,RIndex,RPinky
              ]
    #生徒データを形成後リスト化
    out_list_2=[
            Base_out,#0
              LHip_out,LKnee_out,LAnkle_out,LToes_out,LHeel_out,#1~5
              RHip_out,RKnee_out,RAnkle_out,RToes_out,RHeel_out,#6~10
              Heart_out,Nose_out,                               #11~12
              LShoulder_out,LElbow_out,LWrist_out,LThumb_out,LIndex_out,LPinky_out,#13~18
              RShoulder_out,RElbow_out,RWrist_out,RThumb_out,RIndex_out,RPinky_out#19~24
              ]
    


    #モーション作成
    Motions_1.append(out_list_1)
    Motions_2.append(out_list_2)

print("モーション計算完了")

#ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
#ポーズ判定用
def poseCheck(landmarks1,landmarks2):
    badlist=[]
    print(len(landmarks1))
    print(len(landmarks2))
    
    for index,(landmark1,landmark2) in enumerate(zip(landmarks1,landmarks2)):
        print("インデックスは"+str(index))
        #教師生徒間の同部位の距離を計算
        dif=np.linalg.norm(landmark1-landmark2)
        #間違っている部位はリストに入れる
        if dif>0.1:
            badlist.append(index)
    return badlist
    
def pose_index_set(pose_index_list,landmark_point):
    pose_x,pose_y,pose_z=[],[],[]
    for index in pose_index_list:
        point = landmark_point[index][1]
        pose_x.append(point[0])
        pose_y.append(point[2])
        pose_z.append(point[1] * (-1))

    return pose_x,pose_y,pose_z


#ポーズ描画用
def poseDraw(plt,ax,landmarks,visibility_th=0.5,badlist=[],pose_color="red"):
    landmark_point = []#有効性、xyz座標

    for index, landmark in enumerate(landmarks):
        landmark_point.append(
            [1, (landmark[0], landmark[1], landmark[2])])

    #center_index_list = [0, 11,12]#腰、胸、鼻
    #right_arm_index_list = [11, 19, 20, 21, 22, 23, 24] #胸、肩、肘、手首、外側端(小指)、先端(中指)、内側端(親指)
    #left_arm_index_list = [11, 13, 14, 15, 16, 17, 18]  #胸、肩、肘、手首、外側端(小指)、先端(中指)、内側端(親指)
    #right_reg_side_index_list = [0, 6, 7, 8, 10, 9] #腰、尻、膝、足首、足先、かかと
    #left_reg_side_index_list = [0, 1, 2, 3, 5,4]    #腰、尻、膝、足首、足先、かかと
    """
    # 顔
    center_x, center_y, center_z =pose_index_set(center_index_list,landmark_point)

    # 右腕
    right_arm_x, right_arm_y, right_arm_z =pose_index_set(right_arm_index_list,landmark_point)

    # 左腕
    left_arm_x, left_arm_y, left_arm_z = pose_index_set(left_arm_index_list,landmark_point)

    # 右半身
    right_body_side_x, right_body_side_y, right_body_side_z = pose_index_set(right_reg_side_index_list,landmark_point)

    # 左半身
    left_body_side_x, left_body_side_y, left_body_side_z = pose_index_set(left_reg_side_index_list,landmark_point)
    """

    nose_index = [11, 12]#胸、鼻
    spine_index = [0, 11]#腰、胸

    left_Shoulder_index = [11, 13]  #胸、肩
    left_upArm_index = [13, 14]  #肩、肘
    left_arm_index = [14, 15]  #肘、手首
    left_thumb_index = [15, 18]  #手首、内側端(親指)
    left_pinky_index = [15, 17]  #手首、先端(中指)
    left_index_index = [15, 16]  #手首、外側端(小指)

    right_Shoulder_index = [11, 19] #胸、肩
    right_upArm_index = [19, 20] #肩、肘
    right_arm_index = [20, 21] #肘、手首
    right_thumb_index = [21, 24] #手首、内側端(親指)
    right_pinky_index = [21, 23] #手首、先端(中指)
    right_index_index = [21, 22] #手首、外側端(小指)

    left_hip_index = [0, 1]    #腰、尻
    left_upReg_index = [1, 2]    #尻、膝
    left_reg_index = [2, 3]    #膝、足首
    left_heel_index = [3, 4]    #足首、足先
    left_toes_index = [3, 5]    #足首、かかと

    right_hip_index = [0, 6] #腰、尻
    right_upReg_index = [6, 7] #尻、膝
    right_reg_index = [7, 8] #膝、足首
    right_heel_index = [8, 9] #足首、足先
    right_toes_index = [8, 10] #足首、かかと


    #背骨、鼻
    nose_x, nose_y, nose_z =pose_index_set(nose_index,landmark_point)
    spine_x, spine_y, spine_z =pose_index_set(spine_index,landmark_point)
    #肩
    left_Shoulder_x, left_Shoulder_y, left_Shoulder_z =pose_index_set(left_Shoulder_index,landmark_point)
    right_Shoulder_x, right_Shoulder_y, right_Shoulder_z =pose_index_set(right_Shoulder_index,landmark_point)
    #上腕
    left_upArm_x, left_upArm_y, left_upArm_z =pose_index_set(left_upArm_index,landmark_point)
    right_upArm_x, right_upArm_y, right_upArm_z =pose_index_set(right_upArm_index,landmark_point)
    #前腕
    left_arm_x, left_arm_y, left_arm_z =pose_index_set(left_arm_index,landmark_point)
    right_arm_x, right_arm_y, right_arm_z =pose_index_set(right_arm_index,landmark_point)
    #親指
    left_thumb_x, left_thumb_y, left_thumb_z =pose_index_set(left_thumb_index,landmark_point)
    right_thumb_x, right_thumb_y, right_thumb_z =pose_index_set(right_thumb_index,landmark_point)
    #小指
    left_pinky_x, left_pinky_y, left_pinky_z =pose_index_set(left_pinky_index,landmark_point)
    right_pinky_x, right_pinky_y, right_pinky_z =pose_index_set(right_pinky_index,landmark_point)
    #中指
    left_index_x, left_index_y, left_index_z =pose_index_set(left_index_index,landmark_point)
    right_index_x, right_index_y, right_index_z =pose_index_set(right_index_index,landmark_point)
    #尻
    left_hip_x, left_hip_y, left_hip_z =pose_index_set(left_hip_index,landmark_point)
    right_hip_x, right_hip_y, right_hip_z =pose_index_set(right_hip_index,landmark_point)
    #上足
    left_upReg_x, left_upReg_y, left_upReg_z =pose_index_set(left_upReg_index,landmark_point)
    right_upReg_x, right_upReg_y, right_upReg_z =pose_index_set(right_upReg_index,landmark_point)
    #下足
    left_reg_x, left_reg_y, left_reg_z =pose_index_set(left_reg_index,landmark_point)
    right_reg_x, right_reg_y, right_reg_z =pose_index_set(right_reg_index,landmark_point)
    #つま先
    left_toes_x, left_toes_y, left_toes_z =pose_index_set(left_toes_index,landmark_point)
    right_toes_x, right_toes_y, right_toes_z =pose_index_set(right_toes_index,landmark_point)
    #かかと
    left_heel_x, left_heel_y, left_heel_z =pose_index_set(left_heel_index,landmark_point)
    right_heel_x, right_heel_y, right_heel_z =pose_index_set(right_heel_index,landmark_point)


    bad01_list=[0]*25    #合否を01で判定。0があっている、1が間違っているもの、

        


    #ax.cla()            #グラフをクリア
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    #pltで表示
    #scatterが点、plotが線で繋ぐ
    if len(badlist)>0:
        print("aaa")
    #ax.scatter(center_x, center_y, center_z,color="red")
    ax.plot(center_x, center_y, center_z,color=pose_color)
    ax.plot(right_arm_x, right_arm_y, right_arm_z,color=pose_color)
    ax.plot(left_arm_x, left_arm_y, left_arm_z,color=pose_color)
    ax.plot(right_body_side_x, right_body_side_y, right_body_side_z,color=pose_color)
    ax.plot(left_body_side_x, left_body_side_y, left_body_side_z,color=pose_color)

    plt.xlabel('X-label')
    plt.ylabel('Z-label')
    

    return


# World座標プロット ########################################################
#if plot_world_landmark:
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)

for motion_1,motion_2 in zip(Motions_1,Motions_2): 
    badlist=poseCheck(motion_1,motion_2)
    print(badlist)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    
    #描画
    poseDraw(plt,ax,motion_1,badlist=badlist,pose_color="grey")
    poseDraw(plt,ax,motion_2,badlist=badlist,pose_color="chartreuse")
    plt.pause(.001)
    ax.cla()
    if keyboard.is_pressed('escape'):
        print("中断")
        ax.cla()
        break










