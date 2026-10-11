"""MuJoCo viewer 的可视化设置。"""

import mujoco


def show_center_of_mass(viewer):
    """开启整体质心小球；MuJoCo 会随机构构型变化自动更新位置。"""
    with viewer.lock():
        # 显示根 body 的 subtree_com，包含车身、所有连杆和轮子。
        # 这不是单独的车身质心，也不是排除轮子的倒立摆等效质心。
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True

