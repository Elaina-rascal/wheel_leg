"""生成可碰撞的起伏路面；修改下方参数后运行本文件即可更新地形。"""

from array import array
import math
from pathlib import Path
import random
import struct


# 坐标为世界坐标，长度单位均为米。地形范围 x=[-12, 28]、y=[-15, 15]。
TERRAIN = {
    "center_x": 8.0,
    "half_x": 20.0,
    "half_y": 15.0,
    "grid_spacing": 0.10,
    "flat_radius": 0.55,      # 只保留放置机器人的一小块平地。
    "transition_width": 0.35, # 出发约半米后开始遇到起伏。
    "roughness": 0.018,       # 连续细小起伏的幅度。
    "feature_count": 120,     # 场地里散布的坑包数量。
    "seed": 7,
}

# height 正值是土包，负值是坑；radius_x/y 是椭圆的半轴。
# 左右轮会经过不同的坑，避免整条路只形成相同的上下运动。
FEATURES = [
    {"x": 1.3, "y": -0.25, "radius_x": 0.50, "radius_y": 0.40, "height": -0.075},
    {"x": 2.4, "y": 0.25, "radius_x": 0.65, "radius_y": 0.50, "height": 0.100},
    {"x": 4.0, "y": 0.0, "radius_x": 1.10, "radius_y": 1.30, "height": 0.160},
    {"x": 6.0, "y": 0.2, "radius_x": 0.85, "radius_y": 0.70, "height": -0.100},
    {"x": 8.0, "y": -0.3, "radius_x": 1.10, "radius_y": 0.85, "height": 0.130},
    # 两条纵向车辙；椭圆过渡使入口和出口没有竖直台阶。
    {"x": 11.0, "y": -0.25, "radius_x": 2.00, "radius_y": 0.25, "height": -0.065},
    {"x": 11.0, "y": 0.25, "radius_x": 2.00, "radius_y": 0.25, "height": -0.085},
    {"x": 16.0, "y": 0.0, "radius_x": 2.50, "radius_y": 2.00, "height": 0.180},
    {"x": 21.0, "y": 0.2, "radius_x": 1.50, "radius_y": 1.30, "height": -0.120},
]


def smooth_transition(value):
    """将 0~1 变成两端斜率为零的过渡，范围外钳制到 0/1。"""
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def generate():
    random_source = random.Random(TERRAIN["seed"])
    features = FEATURES.copy()
    # 在主行驶路线两侧和后方散布坑包；固定种子使每次生成结果相同。
    while len(features) < len(FEATURES) + TERRAIN["feature_count"]:
        x = random_source.uniform(TERRAIN["center_x"] - TERRAIN["half_x"] + 1.5,
                                  TERRAIN["center_x"] + TERRAIN["half_x"] - 1.5)
        y = random_source.uniform(-TERRAIN["half_y"] + 1.5, TERRAIN["half_y"] - 1.5)
        if x > 0.0 and abs(y) < 1.5:
            continue  # 主路线由 FEATURES 设置，避免随机坑包掩盖指定地形。
        features.append({
            "x": x,
            "y": y,
            "radius_x": random_source.uniform(0.50, 1.40),
            "radius_y": random_source.uniform(0.50, 1.40),
            "height": random_source.uniform(-0.120, 0.160),
        })

    ncol = round(2.0 * TERRAIN["half_x"] / TERRAIN["grid_spacing"]) + 1
    nrow = round(2.0 * TERRAIN["half_y"] / TERRAIN["grid_spacing"]) + 1
    heights = []
    # 二维高度矩阵按行存储：列对应 x，行对应 y（从负 y 到正 y）。
    for row in range(nrow):
        y = -TERRAIN["half_y"] + row * 2.0 * TERRAIN["half_y"] / (nrow - 1)
        for col in range(ncol):
            local_x = -TERRAIN["half_x"] + col * 2.0 * TERRAIN["half_x"] / (ncol - 1)
            x = TERRAIN["center_x"] + local_x
            height = TERRAIN["roughness"] * (
                0.6 * math.sin(6.1 * x + 1.3 * y) * math.sin(4.3 * y - 0.7 * x)
                + 0.4 * math.sin(2.7 * x - 3.2 * y)
            )
            for feature in features:
                radius = math.hypot((x - feature["x"]) / feature["radius_x"],
                                    (y - feature["y"]) / feature["radius_y"])
                if radius < 1.0:
                    height += feature["height"] * 0.5 * (1.0 + math.cos(math.pi * radius))

            spawn_blend = smooth_transition(
                (math.hypot(x, y) - TERRAIN["flat_radius"]) / TERRAIN["transition_width"]
            )
            edge_distance = min(TERRAIN["half_x"] - abs(local_x), TERRAIN["half_y"] - abs(y))
            edge_blend = smooth_transition(edge_distance / 0.5)
            heights.append(height * spawn_blend * edge_blend)

    minimum = min(heights)
    elevation_range = max(heights) - minimum
    # MuJoCo 将高度数据归一化到 [0, 1]，所以 geom 的 z 要放在实际最低点，
    # hfield.size[2] 要取最高与最低点之差，保证出生点仍然位于世界 z=0。
    normalized = array("f", [(height - minimum) / elevation_range for height in heights])
    directory = Path(__file__).resolve().parent
    with (directory / "terrain.bin").open("wb") as terrain_file:
        terrain_file.write(struct.pack("=ii", nrow, ncol))
        normalized.tofile(terrain_file)

    # 同时生成尺寸和位置，避免修改坑深后还要手动同步 XML 中的高度范围。
    xml = f'''<mujoco>
    <!-- 由 generate_terrain.py 生成；请在该脚本中调整路面参数。 -->
    <asset>
        <!-- size: x/y 半尺寸、高度范围、最低点以下的实体厚度。 -->
        <hfield name="rough_terrain" file="terrain.bin"
                size="{TERRAIN['half_x']} {TERRAIN['half_y']} {elevation_range:.9f} 0.1"/>
        <!-- 土色哑光地面；不用棋盘纹理，通过地形起伏和光照显示坑包。 -->
        <material name="terrain_material" rgba="0.46 0.37 0.25 1"
                  reflectance="0" specular="0" shininess="0"/>
    </asset>
    <worldbody>
        <geom name="floor" type="hfield" hfield="rough_terrain"
              pos="{TERRAIN['center_x']} 0 {minimum:.9f}" material="terrain_material" condim="3"/>
    </worldbody>
</mujoco>
'''
    (directory / "terrain.xml").write_text(xml, encoding="utf-8")
    print(f"已生成 {ncol}×{nrow} 路面，最低 {minimum:.4f} m，最高 {max(heights):.4f} m")


if __name__ == "__main__":
    generate()
