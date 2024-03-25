# Re-creating the complex BVH file with comments in Japanese

# Defining the structure of the BVH file with Japanese comments
#bvhファイル作成用システム
#sudo python3 bvhmaker2.py
#blenderのインポートエラーで「could not convert string to float: Zrotation」が出る時→中括弧などX改行が原因

#xzyの順番で座標が入っている、yが上
# coding: utf-8
import sqlite3
import numpy as np
import math
import matplotlib.pyplot as plt
import keyboard
#from tensortest import ScoreLearning
import sys

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
    #*************************************************
    #
    #ベクトルの角度取得
    #
    #vec_before :角度を取りたい部分のより腰に近い方。膝部分なら上足
    #vec_after  :角度を取りたい部分のより腰から遠い方。膝部分なら下足
    #
    #
    #*************************************************
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
    #print(xInner/(vec1Len*vec2Len))
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

#フレーム数を合わせる
def fps_set(frames,fps):
    outlist=[]

    #行動のみに変換
    

    total_time=frames[-1][2]
    l=0
    n=0
    t=0
    while t<total_time:
        sublist=[]
        while True:
            if frames[l][2]<=t<frames[l+1][2]:
                time=frames[l+1][2]-frames[l][2]
                
                pose=[]
                for i in range(len(frames)):
                    pass
                #print(frames[l][1])
                pose=(np.array(frames[l][1])*(abs(frames[l+1][2]-time)/time))+(np.array(frames[l][1])*(abs(frames[l][2]-time)/time))
                
                
                sublist.append(n)       #インデックス
                sublist.append(pose)   #ポーズリスト
                sublist.append(t)               #経過時間


                break
                
            else:
                l+=1
        t+=1000/fps
        outlist.append(sublist)
        
        n+=1

    return outlist



#print("ーーーーーーーーーーーーーーーーーーーーーーーーーーーーー")
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
#各フレームにはの情報(インデックス、各関節の情報、現在時間の情報）が入っている
#各関節には[信頼度,(x座標,y座標,z座標)]の情報が入る========


#テーブル呼び出し
#fetchallで全呼び出しするとtupleで書き換え不可なのでfetchmanyで一つずつ変換して呼び出す
#frames1 = c1.fetchall()
#frames2 = c2.fetchall()
frames1 = []
frames2 = []
# 1行ずつlist型で取得
rows = []

while True:
    row1 = c1.fetchmany(size=1)
    row2 = c2.fetchmany(size=1)
    if not row1 or not row2:
        break
    frames1.append([str(row1[0][0]),eval(row1[0][1]),float(row1[0][2])])
    frames2.append([str(row2[0][0]),eval(row2[0][1]),float(row2[0][2])])
    
#print(frames1)


#nフレームの形に形成（30フレーム）
frames1=fps_set(frames1,30)
frames2=fps_set(frames2,30)

#print(frames1[0])




VectorUp=np.array([0,0,0])#上ベクトル

Motions_1=[]
Motions_2=[]
Motions_rot1=[]
Motions_rot2=[]

#print(frames1)


#フレームごとに取得
for index, (landmarks1,landmarks2) in enumerate(zip(frames1,frames2)):

    landmark=(landmarks1[1])
    #print(landmark)
    #教師の位置の設定。（L,Rは絶対位置、Left,Rightは相対位置）
    #部位ごとに取得＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    NoseAbsol=np.array(landmark[0])# 鼻
    LHip=np.array(landmark[23])# 腰(左側)
    RHip=np.array(landmark[24])# 腰(右側)
    Base=((LHip+RHip)/2)#腰(左右腰の中間)
    #上半身ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    LShoulder=np.array(landmark[11])# 左肩
    RShoulder=np.array(landmark[12])# 右肩
    LElbow=np.array(landmark[13])# 左肘
    RElbow=np.array(landmark[14])# 右肘
    LWrist=np.array(landmark[15])# 左手首
    RWrist=np.array(landmark[16])# 右手首
    LPinky=np.array(landmark[17])# 左子指
    RPinky=np.array(landmark[18])# 右子指
    LIndex=np.array(landmark[19])# 左人差し指
    RIndex=np.array(landmark[20])# 左人差し指
    LThumb=np.array(landmark[21])# 左親指
    RThumb=np.array(landmark[22])# 右親指
    #下半身ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    LKnee=np.array(landmark[25])# 左ひざ
    RKnee=np.array(landmark[26])# 右ひざ
    LAnkle=np.array(landmark[27])# 左足首
    RAnkle=np.array(landmark[28])# 右足首
    LHeel=np.array(landmark[29])# 左かかと
    RHeel=np.array(landmark[30])# 右かかと
    LToes=np.array(landmark[31])# 左つま先
    RToes=np.array(landmark[32])# 右つま先
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
    
    #上半身下半身をつなげる
    LeftSide=(LShoulder-LHip)
    RightSide=(RShoulder-RHip)

    #ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    #角度
    #左下半身
    LeftHip_rot=rotxyz(VectorUp,LeftHip)        #左尻角度
    LeftKnee_rot=rotxyz(LeftHip,LeftKnee)       #左足付け根角度
    LeftKnee_rot_2=rotxyz(-LeftSide,LeftKnee)       #左足付け根角度2(Leftsideは尻→肩なので逆にする)
    LeftAnkle_rot=rotxyz(LeftKnee,LeftAnkle)    #左ひざ角度
    LeftToes_rot=rotxyz(LeftAnkle,LeftToes)     #左足首つま先角度
    LeftHeel_rot=rotxyz(LeftAnkle,LeftHeel)     #左足首かかと角度

    #右下半身
    RightHip_rot=rotxyz(VectorUp,RightHip)          #右尻角度
    RightKnee_rot=rotxyz(RightHip,RightKnee)        #右ひざ角度
    RightKnee_rot_2=rotxyz(RightSide,RightKnee)        #右ひざ角度
    RightAnkle_rot=rotxyz(RightKnee,RightAnkle)     #右足首角度
    RightToes_rot=rotxyz(RightAnkle,RightToes)      #右足首つま先角度
    RightHeel_rot=rotxyz(RightToes,RightHeel)       #右足首かかと角度

    #上半身
    Heart_rot=rotxyz(VectorUp,Heart)    #心臓角度
    Nose_rot=rotxyz(Heart,Nose)         #鼻角度

    #左上半身
    LeftShoulder_rot=rotxyz(Heart,LeftShoulder)     #左までの角度 
    LeftElbow_rot=rotxyz(LeftShoulder,LeftElbow)    #左腕付け根角度
    LeftElbow_rot_2=rotxyz(LeftSide,LeftElbow)    #左腕付け根角度2
    LeftWrist_rot=rotxyz(LeftElbow,LeftWrist)       #左ひじ角度
    LeftThumb_rot=rotxyz(LeftWrist,LeftThumb)       #左手首親指角度
    LeftIndex_rot=rotxyz(LeftWrist,LeftIndex)       #左手首中指角度
    LeftPinky_rot=rotxyz(LeftWrist,LeftPinky)       #左手首小指角度

    #右上半身
    RightShoulder_rot=rotxyz(Heart,RightShoulder)       #右肩までの角度 
    RightElbow_rot=rotxyz(RightShoulder,RightElbow)     #右腕付け根角度
    RightElbow_rot_2=rotxyz(RightSide,RightElbow)     #右腕付け根角度2
    RightWrist_rot=rotxyz(RightElbow,RightWrist)        #右ひじ角度
    RightThumb_rot=rotxyz(RightWrist,RightThumb)        #右手首親指角度
    RightIndex_rot=rotxyz(RightWrist,RightIndex)        #右手首中指角度
    RightPinky_rot=rotxyz(RightWrist,RightPinky)        #右手首小指角度
    

    #ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    #ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    #ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    #ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー

    landmark=(landmarks2[1])

    
    #教師の位置の設定。（L,Rは絶対位置、Left,Rightは相対位置）
    #部位ごとに取得＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    NoseAbsol2=np.array(landmark[0])# 鼻
    LHip2=np.array(landmark[23])# 腰(左側)
    RHip2=np.array(landmark[24])# 腰(右側)
    Base2=((LHip2+RHip2)/2)#腰(左右腰の中間)
    #上半身ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    LShoulder2=np.array(landmark[11])# 左肩
    RShoulder2=np.array(landmark[12])# 右肩
    LElbow2=np.array(landmark[13])# 左肘
    RElbow2=np.array(landmark[14])# 右肘
    LWrist2=np.array(landmark[15])# 左手首
    RWrist2=np.array(landmark[16])# 右手首
    LPinky2=np.array(landmark[17])# 左子指
    RPinky2=np.array(landmark[18])# 右子指
    LIndex2=np.array(landmark[19])# 左人差し指
    RIndex2=np.array(landmark[20])# 左人差し指
    LThumb2=np.array(landmark[21])# 左親指
    RThumb2=np.array(landmark[22])# 右親指
    #下半身ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    LKnee2=np.array(landmark[25])# 左ひざ
    RKnee2=np.array(landmark[26])# 右ひざ
    LAnkle2=np.array(landmark[27])# 左足首
    RAnkle2=np.array(landmark[28])# 右足首
    LHeel2=np.array(landmark[29])# 左かかと
    RHeel2=np.array(landmark[30])# 右かかと
    LToes2=np.array(landmark[31])# 左つま先
    RToes2=np.array(landmark[32])# 右つま先
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

    #上半身下半身をつなげる
    LeftSide2=(LShoulder2-LHip2)
    RightSide2=(RShoulder2-RHip2)

    #角度
    #左下半身
    LeftHip_rot2=rotxyz(VectorUp,LeftHip2)        #左尻角度
    LeftKnee_rot2=rotxyz(LeftHip2,LeftKnee2)       #左足付け根角度
    LeftKnee_rot2_2=rotxyz(-LeftSide2,LeftKnee2)       #左足付け根角度2(Leftsideは尻→肩なので逆にする)
    LeftAnkle_rot2=rotxyz(LeftKnee2,LeftAnkle2)    #左ひざ角度
    LeftToes_rot2=rotxyz(LeftAnkle2,LeftToes2)     #左足首つま先角度
    LeftHeel_rot2=rotxyz(LeftAnkle2,LeftHeel2)     #左足首かかと角度

    #右下半身
    RightHip_rot2=rotxyz(VectorUp,RightHip2)          #右尻角度
    RightKnee_rot2=rotxyz(RightHip2,RightKnee2)        #右ひざ角度
    RightKnee_rot2_2=rotxyz(RightSide2,RightKnee2)        #右ひざ角度
    RightAnkle_rot2=rotxyz(RightKnee2,RightAnkle2)     #右足首角度
    RightToes_rot2=rotxyz(RightAnkle2,RightToes2)      #右足首つま先角度
    RightHeel_rot2=rotxyz(RightToes2,RightHeel2)       #右足首かかと角度

    #上半身
    Heart_rot2=rotxyz(VectorUp,Heart2)    #心臓角度
    Nose_rot2=rotxyz(Heart2,Nose2)         #鼻角度

    #左上半身
    LeftShoulder_rot2=rotxyz(Heart2,LeftShoulder2)     #左までの角度 
    LeftElbow_rot2=rotxyz(LeftShoulder2,LeftElbow2)    #左腕付け根角度
    LeftElbow_rot2_2=rotxyz(LeftSide2,LeftElbow2)    #左腕付け根角度2
    LeftWrist_rot2=rotxyz(LeftElbow2,LeftWrist2)       #左ひじ角度
    LeftThumb_rot2=rotxyz(LeftWrist2,LeftThumb2)       #左手首親指角度
    LeftIndex_rot2=rotxyz(LeftWrist2,LeftIndex2)       #左手首中指角度
    LeftPinky_rot2=rotxyz(LeftWrist2,LeftPinky2)       #左手首小指角度

    #右上半身
    RightShoulder_rot2=rotxyz(Heart2,RightShoulder2)       #右肩までの角度 
    RightElbow_rot2=rotxyz(RightShoulder2,RightElbow)     #右腕付け根角度
    RightElbow_rot2_2=rotxyz(RightSide2,RightElbow2)     #右腕付け根角度2
    RightWrist_rot2=rotxyz(RightElbow2,RightWrist2)        #右ひじ角度
    RightThumb_rot2=rotxyz(RightWrist2,RightThumb2)        #右手首親指角度
    RightIndex_rot2=rotxyz(RightWrist2,RightIndex2)        #右手首中指角度
    RightPinky_rot2=rotxyz(RightWrist2,RightPinky2)        #右手首小指角度
    #ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー

    #ーーーーーーーーーーーーーーーーーーーーーーーーー
    #骨格形成
    Base_out=((LHip+RHip)/2)#腰(左右腰の中間)（これが中心なのでマスターデータ側の座標に合わせる）
    LHip_out=vectorSet(Base_out,LeftHip2,LeftHip)# 腰(左側)
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
              HeartAbsol,NoseAbsol,
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
    
    #角度リスト
    rot_lists=[
        LeftHip_rot,[LeftKnee_rot,LeftKnee_rot_2],LeftAnkle_rot,LeftToes_rot,LeftHeel_rot,#左下半身
        RightHip_rot,[RightKnee_rot,RightKnee_rot_2],RightAnkle_rot,RightToes_rot,RightHeel_rot,#右下半身
        Heart_rot,Nose_rot,
        LeftShoulder_rot,[LeftElbow_rot,LeftElbow_rot_2],LeftWrist_rot,LeftThumb_rot,LeftIndex_rot,LeftPinky_rot,#左上半身
        RightShoulder_rot,[RightElbow_rot,RightElbow_rot_2],RightWrist_rot,RightThumb_rot,RightIndex_rot,RightPinky_rot#右上半身
    ]    
    
    #角度リスト
    rot_lists2=[
        LeftHip_rot2,[LeftKnee_rot2,LeftKnee_rot2_2],LeftAnkle_rot2,LeftToes_rot2,LeftHeel_rot2,#左下半身
        RightHip_rot2,[RightKnee_rot2,RightKnee_rot2_2],RightAnkle_rot2,RightToes_rot2,RightHeel_rot2,#右下半身
        Heart_rot2,Nose_rot2,
        LeftShoulder_rot2,[LeftElbow_rot2,LeftElbow_rot2_2],LeftWrist_rot2,LeftThumb_rot2,LeftIndex_rot2,LeftPinky_rot2,#左上半身
        RightShoulder_rot2,[RightElbow_rot2,RightElbow_rot2_2],RightWrist_rot2,RightThumb_rot2,RightIndex_rot2,RightPinky_rot2#右上半身
    ]


    #モーション作成
    Motions_1.append(out_list_1)
    Motions_2.append(out_list_2)
    Motions_rot1.append(rot_lists)
    Motions_rot2.append(rot_lists2)

print("モーション計算完了")

#ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
#ポーズ判定用（座標）
def poseCheck_xyz(landmarks1,landmarks2):

    length_list=[0]*len(landmarks1)
    
    for index,(landmark1,landmark2) in enumerate(zip(landmarks1,landmarks2)):
        dif=0
        #教師生徒間の同部位の距離を計算
        dif=np.linalg.norm(landmark1-landmark2)

        #長さをリストにいれる
        length_list[index]=dif

    return length_list

#ポーズ判定用（角度）
def poseCheck_rot(rot1,rot2):

    length_list=[0]*len(rot1)

    for index,(r1,r2) in enumerate(zip(rot1,rot2)):
        dif=0
        if type(r1)==list:
            for i in range(len(r1)):
                dif=max(dif,r1[i]-r2[i])

        else:   
            #教師生徒間の同部位の距離を計算
            dif=r1-r2

        #角度をリストにいれる
        length_list[index]=dif


    return length_list
    
def pose_index_set(pose_index_list,landmark_point):
    pose_x,pose_y,pose_z=[],[],[]
    for index in pose_index_list:
        point = landmark_point[index][1]
        pose_x.append(point[0])
        pose_y.append(point[2])
        pose_z.append(point[1] * (-1))

    return pose_x,pose_y,pose_z

def PlotAndColor(bonelist,ax,badlist,good_color,bad_color,isMaster,isbad):
    #*****************************************************************************
    #
    # bonelist:boneのxyz座標のリスト。[[x1,y1,z1],[x2,y2,z2],[x2,y2,z2],.....]のようになっており、
    # 座標1と2、2と3、3と4をそれぞれ繋いでいく
    # ax :pltの表示用
    # badlist:  座標がマスターの規定値よりずれているリスト
    # good_color:   合っている時の表示色
    # bad_color:    ずれている時の表示色
    # isMaster: マスターデータかどうか、マスターデータならすべて合っている判定
    # isbad :   親の骨格がずれているかどうか、ずれているならすべて間違っている判定
    #
    #*****************************************************************************
    
    parent_isbad=False #親ボーンが間違っているか
    for [i,[bone_x, bone_y, bone_z]] in bonelist:

        #マスターデータの場合、すべてデフォルト色で表示
        if isMaster:
            ax.plot(bone_x, bone_y, bone_z,color=good_color)
            continue

        if isbad:
            ax.plot(bone_x, bone_y, bone_z,color=bad_color)
            continue

        #間違っているリストにある。もしくは親ボーンが間違っている場合、赤色で表示
        if badlist[i] or parent_isbad:
            ax.plot(bone_x, bone_y, bone_z,color=bad_color)
            parent_isbad=True
        else:
            ax.plot(bone_x, bone_y, bone_z,color=good_color)
            
        
    return parent_isbad

#ポーズ描画用
def poseDraw(plt,ax,landmarks,visibility_th=0.5,badlist=[],pose_color="red",isMaster=False):
    landmark_point = []#有効性、xyz座標

    for index, landmark in enumerate(landmarks):
        landmark_point.append(
            [index, [landmark[0], landmark[1], landmark[2]]])

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

    #各座標をリスト化(0=x,1=y,2=z)

    #背骨、鼻
    nose_bone=pose_index_set(nose_index,landmark_point)
    spine_bone =pose_index_set(spine_index,landmark_point)

    
    #肩
    left_Shoulder_bone =pose_index_set(left_Shoulder_index,landmark_point)
    right_Shoulder_bone =pose_index_set(right_Shoulder_index,landmark_point)
    #上腕
    left_upArm_bone =pose_index_set(left_upArm_index,landmark_point)
    right_upArm_bone =pose_index_set(right_upArm_index,landmark_point)
    #前腕
    left_arm_bone =pose_index_set(left_arm_index,landmark_point)
    right_arm_bone =pose_index_set(right_arm_index,landmark_point)
    #親指
    left_thumb_bone =pose_index_set(left_thumb_index,landmark_point)
    right_thumb_bone =pose_index_set(right_thumb_index,landmark_point)
    #小指
    left_pinky_bone =pose_index_set(left_pinky_index,landmark_point)
    right_pinky_bone =pose_index_set(right_pinky_index,landmark_point)
    #中指
    left_index_bone =pose_index_set(left_index_index,landmark_point)
    right_index_bone =pose_index_set(right_index_index,landmark_point)


    
    #尻
    left_hip_bone =pose_index_set(left_hip_index,landmark_point)
    right_hip_bone =pose_index_set(right_hip_index,landmark_point)
    #上足
    left_upReg_bone =pose_index_set(left_upReg_index,landmark_point)
    right_upReg_bone =pose_index_set(right_upReg_index,landmark_point)
    #下足
    left_reg_bone =pose_index_set(left_reg_index,landmark_point)
    right_reg_bone =pose_index_set(right_reg_index,landmark_point)
    #つま先
    left_toes_bone =pose_index_set(left_toes_index,landmark_point)
    right_toes_bone =pose_index_set(right_toes_index,landmark_point)
    #かかと
    left_heel_bone =pose_index_set(left_heel_index,landmark_point)
    right_heel_bone =pose_index_set(right_heel_index,landmark_point)

    #＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

    #骨リストを作成。
    #
    #[骨のインデックス,骨の座標リスト]
    #
    spine_bonelist=[[11, spine_bone]]
    nose_bonelist=[[12, nose_bone]]
    #左上半身_骨リスト
    up_left_bonelist=[[13, left_Shoulder_bone],[14, left_upArm_bone],[15, left_arm_bone]]
    #左手指_骨リスト
    left_finger_bonelist=[[16, left_pinky_bone],[17, left_index_bone],[18, left_thumb_bone]]

    #右上半身_骨リスト
    up_right_bonelist=[[19, right_Shoulder_bone],[20, right_upArm_bone],[21, right_arm_bone]]
    #右手指_骨リスト
    right_finger_bonelist=[[22, right_pinky_bone],[23, right_index_bone],[24, right_thumb_bone]]

    #左下半身_骨リスト
    down_left_bonelist=[[1, left_hip_bone],[2, left_upReg_bone],[3, left_reg_bone]]
    #左つま先かかと_骨リスト
    left_toesheel_bonelist=[[4, left_toes_bone],[5, left_heel_bone]]

    #右下半身_骨リスト
    down_right_bonelist=[[6, right_hip_bone],[7, right_upReg_bone],[8, right_reg_bone]]
    #右つま先かかと_骨リスト
    right_toesheel_bonelist=[[9, right_toes_bone],[10, right_heel_bone]]


    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    #pltで表示
    #scatterが点、plotが線で繋ぐ
    bad_color="red"

    #背骨
    isSpine=PlotAndColor(spine_bonelist,ax,badlist,pose_color,bad_color,isMaster,False)
    #鼻
    PlotAndColor(nose_bonelist,ax,badlist,pose_color,bad_color,isMaster,isSpine)

    #左上半身
    isUpLeft=PlotAndColor(up_left_bonelist,ax,badlist,pose_color,bad_color,isMaster,isSpine)
    #右上半身
    isUpRight=PlotAndColor(up_right_bonelist,ax,badlist,pose_color,bad_color,isMaster,isSpine)

    #左下半身
    isDownLeft=PlotAndColor(down_left_bonelist,ax,badlist,pose_color,bad_color,isMaster,False)
    #右下半身
    isDownRight=PlotAndColor(down_right_bonelist,ax,badlist,pose_color,bad_color,isMaster,False)

    #左指
    PlotAndColor(left_finger_bonelist,ax,badlist,pose_color,bad_color,isMaster,isUpLeft)
    #右指
    PlotAndColor(right_finger_bonelist,ax,badlist,pose_color,bad_color,isMaster,isUpRight)   

    #つま先かかと
    PlotAndColor(right_toesheel_bonelist,ax,badlist,pose_color,bad_color,isMaster,isDownLeft)
    #つま先かかと
    PlotAndColor(left_toesheel_bonelist,ax,badlist,pose_color,bad_color,isMaster,isDownRight)


    plt.xlabel('X-label')
    plt.ylabel('Z-label')

    return

#良し悪しを判定する
def isGood(lists,bad):
    l=[bad]*len(lists)
    out=[False]*len(lists)
    for i in range(len(lists)):
        if lists[i]>l[i]:
            out[i]=True

    return out

# World座標プロット ########################################################
#if plot_world_landmark:
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)

train_data=[]

for motion_1,motion_2,rot1,rot2 in zip(Motions_1,Motions_2,Motions_rot1,Motions_rot2):

    #座標がずれている地点を取得 
    length_list=poseCheck_xyz(motion_1,motion_2)

    #角度がずれている点を取得
    length_rot_list=poseCheck_rot(rot1,rot2)

    #座標差、角度差を連結
    #r=np.concatenate(length_list,length_rot_list)

    train_data.append(length_list)

    



    
    #描画のための処理ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
    #角度か位置、どちらかがズレていれば間違いにする
    zahyou = isGood(length_list,0.1)
    rot = isGood(length_rot_list,10)
    badlist= zahyou or rot 

    #描画
    poseDraw(plt,ax,motion_1,badlist=badlist,pose_color="grey",isMaster=True)
    poseDraw(plt,ax,motion_2,badlist=badlist,pose_color="chartreuse",isMaster=False)

    plt.pause(.0001)

    #グラフ表示をクリア
    ax.cla()

    if keyboard.is_pressed('escape'):
        print("中断")
        ax.cla()
        sys.exit()
        break

#score=ScoreLearning(np.array(length_list),score_est)

#score_est=np.array([8]*len(train_data)) #ユーザーの推定したスコア
#score=ScoreLearning(np.array(train_data),score_est)

