# mp_to_blend2.py
# part 2 ... 動画

# ========================================================================
# 準備
# ========================================================================

# bpyを外部環境でインポートする時のメッセージを防ぐ
import os, ssl
if not os.path.exists("/run/user/1000/gvfs"):
    os.mkdir("/run/user/1000/gvfs")

# Blenderでmediapipeをインポートする時[SSL: CERTIFICATE_VERIFY_FAILED]を回避
ssl._create_default_https_context = ssl._create_unverified_context

# ...................................................................
# 必要なモジュールを読み込む
import cv2
import mediapipe as mp
import bpy
from mathutils import Quaternion, Vector, Matrix

# パス、ファイル名
PrjDir = "/Path/to/Project/"
vidName = "harehare.mp4"
vidIn = PrjDir + "vid/" + vidName
vidOut = PrjDir + "vid/Output." + vidName.split(".")[1]

# 動画は１回作ればよい
if os.path.exists(vidOut):
    vidOutExist = True
else:
    vidOutExist = False

# ...................................................................

# MediaPipeのオブジェクト
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ========================================================================
# サブ
# ========================================================================

def vertsCalcZdepth(h, w, lm):
    '''
    MediaPipeのデータに33以降を追加
    Pose用頂点計算：  1) 奥行きに定数を乗算して浅くする
                    2) 既存の線分の中点を計算し、体の中心線が引けるようにする
    '''
    z_depth = 0.15
    vList, vAdd = [], []
    Counter = 0
    for i in lm:
        V = str(i).split("\n")
        Vx = float(V[0].split(": ")[1]) * w * 0.001
        Vy = float(V[1].split(": ")[1]) * h * 0.001
        Vz = float(V[2].split(": ")[1]) * w * 0.001 * z_depth
        vList.append((Vx, Vz, Vy*-1))   # zが上、-yが前
        if Counter in [9,10,11,12,23,24]:
            vAdd.append((Vx,Vy,Vz))
        Counter += 1
    # 33〜35追加
    # 33=23,24の中点　34=11,12の中点　35=9,10の中点
    V33x = (vAdd[4][0] + vAdd[5][0]) / 2
    V33y = (vAdd[4][1] + vAdd[5][1]) / 2
    V33z = (vAdd[4][2] + vAdd[5][2]) / 2
    V33 = (V33x, V33z, V33y*-1)
    V34x = (vAdd[2][0] + vAdd[3][0]) / 2
    V34y = (vAdd[2][1] + vAdd[3][1]) / 2
    V34z = (vAdd[2][2] + vAdd[3][2]) / 2
    V34 = (V34x, V34z, V34y*-1)
    V35x = (vAdd[0][0] + vAdd[1][0]) / 2
    V35y = (vAdd[0][1] + vAdd[1][1]) / 2
    V35z = (vAdd[0][2] + vAdd[1][2]) / 2
    V35 = (V35x, V35z, V35y*-1)
    vList.append(V33)
    vList.append(V34)
    vList.append(V35)
    # 36, 37 (Stomach, Chest)
    V36x = (V33x + V34x) / 2
    V36y = (V33y + V34y) / 2
    V36z = (V33z + V34z) / 2
    V36 = (V36x, V36z, V36y*-1)
    vList.append(V36)
    V37x = (V33x + V36x) / 2
    V37y = (V33y + V36y) / 2
    V37z = (V33z + V36z) / 2
    V37 = (V37x, V37z, V37y*-1)
    vList.append(V37)
    return vList

def Bones(vList):
    '''
    ボーンの名前、ボーンのスタート位置、子ボーンのスタート位置
    rotation_differenceと行列の掛け算で回転させる
    '''
    # LocationZahyo
    Pelvis = Vector(vList[33])
    Stomach = Vector(vList[37])
    Chest = Vector(vList[36])
    Neck = Vector(vList[34])
    Head = Vector(vList[35])
    Clavicle_L = Vector(vList[34])
    Arm_L = Vector(vList[11])
    Forearm_L = Vector(vList[13])
    Hand_L = Vector(vList[15])
    Clavicle_R = Vector(vList[34])
    Arm_R = Vector(vList[12])
    Forearm_R = Vector(vList[14])
    Hand_R = Vector(vList[16])
    Thigh_L = Vector(vList[23])
    Calf_L = Vector(vList[25])
    Foot_L = Vector(vList[27])
    Thigh_R = Vector(vList[24])
    Calf_R = Vector(vList[26])
    Foot_R = Vector(vList[28])

    BornNames = ("Pelvis", "Stomach", "Chest", "Neck", "Head",
                 "Clavicle_L", "Arm_L", "Forearm_L", "Hand_L",
                 "Clavicle_R", "Arm_R", "Forearm_R", "Hand_R",
                 "Thigh_L", "Calf_L", "Foot_L",
                 "Thigh_R", "Calf_R", "Foot_R")

    # 画像の幅と高さからMediaPipeが計算した数値。ポーズモードでは直接は使えない。
    LocVec = (Pelvis, Stomach, Chest, Neck, Head,
              Clavicle_L, Arm_L, Forearm_L, Hand_L,
              Clavicle_R, Arm_R, Forearm_R, Hand_R,
              Thigh_L, Calf_L, Foot_L,
              Thigh_R, Calf_R, Foot_R)

    # ボーンのテールの位置（＝子のスタート位置）
    Pel_tail = Stomach
    Sto_tail = Chest
    Che_tail = Neck
    Nek_tail = Head
    Hed_tail = Vector(vList[0])
    ClaL_tail = Arm_L
    AmL_tail = Forearm_L
    FAmL_tail = Hand_L
    HndL_tail = Vector(vList[19])
    ClaR_tail = Arm_R
    AmR_tail = Forearm_R
    FAmR_tail = Hand_R
    HndR_tail = Vector(vList[20])
    ThiL_tail = Calf_L
    CalL_tail = Foot_L
    FtL_tail = Vector(vList[31])
    ThiR_tail = Calf_R
    CalR_tail = Foot_R
    FtR_tail = Vector(vList[32])

    TailV = (Pel_tail, Sto_tail, Che_tail, Nek_tail, Hed_tail,
             ClaL_tail, AmL_tail, FAmL_tail, HndL_tail,
             ClaR_tail, AmR_tail, FAmR_tail, HndR_tail,
             ThiL_tail, CalL_tail, FtL_tail,
             ThiR_tail, CalR_tail, FtR_tail)

    return [BornNames, LocVec, TailV]


def rotMat(pb, prvHead, prvTail, crntHead, crntTail):
    pb = bpy.context.active_pose_bone

    prvVec = prvTail - prvHead
    targetVec = crntTail - crntHead
    Q = prvVec.rotation_difference(targetVec)

    # ３個のマトリクス(行列)の掛け算
    M = (
        Matrix.Translation(pb.head) @
        Q.to_matrix().to_4x4() @
        Matrix.Translation(-pb.head)
        )

    # 元の行列にMを掛けたものを代入
    pb.matrix = M @ pb.matrix


# ========================================================================
# メイン
# ========================================================================

# ポーズ・オブジェクト
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# 動画を開く
cap = cv2.VideoCapture(vidIn)
if cap.isOpened() == False:
    print("入力動画だめぽ")
    raise TypeError
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
vLen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# ランドマーク付き動画がなければ書き出す
if not vidOutExist:
    out = cv2.VideoWriter(vidOut, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                          fps, (w, h))

lmAll = []      # 頂点データの入れ物。Blender用
while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not vidOutExist:
        mp_drawing.draw_landmarks(image,
                                  results.pose_landmarks,
                                  mp_pose.POSE_CONNECTIONS)
        out.write(image)

    # 頂点データをアペンド
    Pose_verts = vertsCalcZdepth(h, w,
                                 results.pose_landmarks.landmark)
    lmAll.append(Pose_verts)
pose.close()
cap.release()
if not vidOutExist:
    out.release()


# ...................................................................
# Blener内での処理
try:
    Arm = bpy.data.objects["Armature"]      # 存在しない場合は終了
except KeyError:
    print("Armatureがないの")
    exit()

Arm.select_set(True)
bpy.ops.object.posemode_toggle()
#bpy.context.object.data.show_axes = True
#bpy.context.object.data.show_names = True

FrmNo = 0
# １フレームごとの全身のランドマークの座標 x 動画のフレーム数
# Bones(vList) は３種類の情報を返す [BornNames, LocVec, Vecs]
for i in lmAll:
    Names = Bones(i)[0]
    LocVec = Bones(i)[1]
    TailVec = Bones(i)[2]

    if FrmNo == 0:
        PrvLoc = LocVec
        PrvTail = TailVec

    bpy.context.scene.frame_set(FrmNo)
    for j in Names:
        bpy.context.object.data.bones.active = Arm.data.bones[j]

        bpy.context.scene.transform_orientation_slots[0].type = "GLOBAL"
        jIndx = Names.index(j)

        LocDiff = LocVec[jIndx] - PrvLoc[jIndx]
        if j in ["Pelvis", "Arm_L", "Arm_R", "Thigh_L", "Thigh_R"]:
            # 位置
            if j == "Pelvis":
                bpy.context.active_pose_bone.location = LocVec[jIndx]
            else:
                bpy.context.active_pose_bone.location = LocDiff
            bpy.context.active_pose_bone.keyframe_insert(data_path="location")

        # 回転
        bpy.context.scene.transform_orientation_slots[0].type = "GLOBAL"
        bpy.context.active_pose_bone.rotation_mode = 'QUATERNION'
        bpy.context.active_bone.use_inherit_rotation = False
        # 回転行列を適用する
        # rotMat(pb, prvHead, prvTail, crntHead, crntTail):
        rotMat(Arm.data.bones[j], PrvLoc[jIndx], PrvTail[jIndx], LocVec[jIndx], TailVec[jIndx])
        bpy.context.active_pose_bone.keyframe_insert(data_path="rotation_quaternion")

    PrvLoc = LocVec
    PrvTail = TailVec
    FrmNo += 1
bpy.ops.object.posemode_toggle()