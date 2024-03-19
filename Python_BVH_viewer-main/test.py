# Re-creating the complex BVH file with comments in Japanese

# Defining the structure of the BVH file with Japanese comments
bvh_content_complex_japanese = """
HIERARCHY
ROOT Hips # ルートボーン: 腰
{
	OFFSET 0.00 0.00 0.00 # 初期オフセット
	CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation # モーションデータに影響するチャネル
	JOINT LeftUpLeg # 子ボーン: 左上脚
	{
		OFFSET 0.00 -40.00 0.00 # 初期オフセット
		CHANNELS 3 Zrotation Xrotation Yrotation # チャネル
		End Site
		{
			OFFSET 0.00 -40.00 0.00 # 終端オフセット
		}
	}
	JOINT RightUpLeg # 子ボーン: 右上脚
	{
		OFFSET 0.00 -40.00 0.00 # 初期オフセット
		CHANNELS 3 Zrotation Xrotation Yrotation # チャネル
		End Site
		{
			OFFSET 0.00 -40.00 0.00 # 終端オフセット
		}
	}
	JOINT Spine # 子ボーン: 脊椎
	{
		OFFSET 0.00 40.00 0.00 # 初期オフセット
		CHANNELS 3 Zrotation Xrotation Yrotation # チャネル
		JOINT Head # 子ボーン: 頭
		{
			OFFSET 0.00 30.00 0.00 # 初期オフセット
			CHANNELS 3 Zrotation Xrotation Yrotation # チャネル
			End Site
			{
				OFFSET 0.00 10.00 0.00 # 終端オフセット
			}
		}
	}
}
MOTION
Frames: 150 # フレーム数
Frame Time: 0.0333333 # フレーム時間
"""

# Adding the motion data with Japanese comments
for frame in range(num_frames):
    hip_rotation = 10 * math.sin(2 * math.pi * (frame / num_frames))  # 腰の揺れ
    head_rotation = 10 * math.cos(2 * math.pi * (frame / num_frames))  # 頭の動き

    frame_data = f"0.00 0.00 0.00 {hip_rotation:.2f} 0.00 0.00 0.00 0.00 0.00 {hip_rotation:.2f} 0.00 0.00 0.00 0.00 {head_rotation:.2f} 0.00 0.00 0.00\n"
    bvh_content_complex_japanese += frame_data

# Saving the content to a BVH file
bvh_complex_japanese_file_path = '/mnt/data/complex_motion_japanese.bvh'
with open(bvh_complex_japanese_file_path, 'w') as file:
    file.write(bvh_content_complex_japanese.strip())

bvh_complex_japanese_file_path

