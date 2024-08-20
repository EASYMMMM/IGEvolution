'''
生成humanoid_srl的XML模型
'''

from RobotGraph import RobotGraph,RobotJoint,RobotLink
from mjcf import elements as e
import networkx as nx
import matplotlib.pyplot as plt
import queue
from RobotModelGen import ModelGenerator

def SRL_mode1(  name='srl_1',
            first_leg_lenth = 0.40,
            first_leg_size = 0.03,
            second_leg_lenth = 0.80,
            second_leg_size = 0.03,
            third_leg_lenth = 0.18,
            third_leg_size = 0.03,
            pretrain = False,
          ):
    if pretrain == False:
        density = 2500
    else:
        density = 10    

    R = RobotGraph(name=name)
    backpack_width = 0.19
    backpack_thick = 0.02

    root = RobotLink('root',link_type = 'box' , size=[backpack_thick,backpack_width/2,backpack_width/2],body_pos=[-.07, 0, 0.05],geom_pos=[0,0,0],euler=[0,0,180], density=density)
    R.add_node( node_type='link',node_info = root)

    # 添加左腿
    # 髋关节
    left_hipjoint_z = RobotJoint('left_hip_z',axis=[0,0,1],joint_range=[-40,0])
    R.add_node( node_type='joint', node_info=left_hipjoint_z)
    R.add_edge(started_node='root',ended_node='left_hip_z')
    left_hipjoint_y = RobotJoint('left_hip_y',axis=[0,1,0],joint_range=[-5,70])
    R.add_node( node_type='joint', node_info=left_hipjoint_y)
    R.add_edge(started_node='root',ended_node='left_hip_y')
    # 大腿
    left_leg1 = RobotLink('left_leg1',length=first_leg_lenth,  size=first_leg_size,body_pos=[backpack_thick,-backpack_width/4,0],euler=[0,0,0], density=density)    
    R.add_node( node_type='link', node_info=left_leg1)
    R.add_edge(started_node='left_hip_z',ended_node='left_leg1')
    R.add_edge(started_node='left_hip_y',ended_node='left_leg1')
    # 膝关节
    left_kneejoint = RobotJoint('left_kneejoint',axis=[0,1,0],joint_range=[-45,45])
    R.add_node( node_type='joint', node_info=left_kneejoint)
    R.add_edge(started_node='left_leg1',ended_node='left_kneejoint')
    # 小腿
    left_leg2 = RobotLink('left_leg2',length=second_leg_lenth, size=second_leg_size,body_pos=[first_leg_lenth,0,0],euler=[0,45,0], density=density)       
    R.add_node( node_type='link', node_info=left_leg2)
    R.add_edge(started_node='left_kneejoint',ended_node='left_leg2')
    # 踝关节
    left_ankle = RobotJoint('left_ankle',axis=[0,1,0],joint_range=[-5,5])
    R.add_node( node_type='joint', node_info=left_ankle)
    R.add_edge(started_node='left_leg2',ended_node='left_ankle')
    # 添加末端肢体
    left_end =  RobotLink('left_end',link_type='sphere',geom_pos=[0,0,0],size=third_leg_size*1.2,body_pos=[second_leg_lenth,0,0],euler=[0,-60,0],density=density)
    # shin2 = RobotLink('shin2',length=third_leg_lenth, size=third_leg_size,body_pos=[second_leg_lenth,0,0],euler=[0,-90,0])    
    R.add_node( node_type='link', node_info=left_end)
    R.add_edge(started_node='left_ankle',ended_node='left_end')

    # 添加第二条腿
    # 添加joint01, 髋关节
    right_hipjoint_z = RobotJoint('right_hipjoint_z',axis=[0,0,1],joint_range=[0,40])
    R.add_node( node_type='joint', node_info=right_hipjoint_z)
    R.add_edge(started_node='root',ended_node='right_hipjoint_z')
    right_hipjoint_y = RobotJoint('right_hipjoint_y',axis=[0,1,0],joint_range=[-5,70])
    R.add_node( node_type='joint', node_info=right_hipjoint_y)
    R.add_edge(started_node='root',ended_node='right_hipjoint_y')
    # 大腿
    right_leg1 = RobotLink('right_leg1',length=first_leg_lenth, size=first_leg_size,body_pos=[backpack_thick,backpack_width/4,0],euler=[0,0,0], density=density)    
    R.add_node( node_type='link', node_info=right_leg1)
    R.add_edge(started_node='right_hipjoint_z',ended_node='right_leg1')
    R.add_edge(started_node='right_hipjoint_y',ended_node='right_leg1')
    # 膝关节
    right_kneejoint = RobotJoint('right_kneejoint',axis=[0,1,0],joint_range=[-45,45])
    R.add_node( node_type='joint', node_info=right_kneejoint)
    R.add_edge(started_node='right_leg1',ended_node='right_kneejoint')
    # 小腿
    right_leg2 = RobotLink('right_leg2',length=second_leg_lenth, size=second_leg_size,body_pos=[first_leg_lenth,0,0],euler=[0,45,0], density=density)     
    R.add_node( node_type='link', node_info=right_leg2)
    R.add_edge( started_node='right_kneejoint',ended_node='right_leg2')
    # 踝关节
    right_ankle = RobotJoint('right_ankle',axis=[0,1,0],joint_range=[-5,5])
    R.add_node( node_type='joint', node_info=right_ankle)
    R.add_edge(started_node='right_leg2',ended_node='right_ankle')
    # 末端
    right_end =  RobotLink('right_end',link_type='sphere',geom_pos=[0,0,0],size=third_leg_size*1.2,body_pos=[second_leg_lenth,0,0],euler=[0,-60,0],density=density)
    R.add_node( node_type='link', node_info=right_end)
    R.add_edge(started_node='right_ankle',ended_node='right_end')

    return R


if __name__ == '__main__':
    R = SRL_mode1(name='humanoid_srl_mode1_pretrain',pretrain=True)
    M = ModelGenerator(R)
    M.gen_basic_humanoid_xml()
    M.get_SRL_dfs()
    M.generate()

    R = SRL_mode1(name='humanoid_srl_mode1')
    M = ModelGenerator(R)
    M.gen_basic_humanoid_xml()
    M.get_SRL_dfs(back_load=True)
    M.generate()

