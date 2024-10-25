'''
生成humanoid_srl的XML模型
'''

from .RobotGraph import RobotGraph,RobotJoint,RobotLink
from .mjcf import elements as e
import networkx as nx
import matplotlib.pyplot as plt
import queue
from .RobotModelGen import ModelGenerator

def SRL_mode1(  name='srl_1',
                first_leg_lenth = 0.40,
                first_leg_size = 0.03,
                second_leg_lenth = 0.80,
                second_leg_size = 0.03,
                third_leg_size = 0.05,
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
    left_kneejoint = RobotJoint('left_kneejoint',axis=[0,1,0],joint_range=[-50,50])
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
    left_end =  RobotLink('left_end',link_type='sphere',geom_pos=[0,0,-0.045],size=third_leg_size,body_pos=[second_leg_lenth,0,0],euler=[0,-90,0],density=density,friction=[5.0,0.05,0.05])    
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
    right_kneejoint = RobotJoint('right_kneejoint',axis=[0,1,0],joint_range=[-50, 50])
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
    right_end =  RobotLink('right_end',link_type='sphere',geom_pos=[0,0,-0.045],size=third_leg_size,body_pos=[second_leg_lenth,0,0],euler=[0,-90,0],density=density,friction=[5.0,0.05,0.05])
    R.add_node( node_type='link', node_info=right_end)
    R.add_edge(started_node='right_ankle',ended_node='right_end')

    return R

def SRL_mode2(  name='srl_2',
            first_leg_lenth = 0.35,
            first_leg_size = 0.03,
            second_leg_lenth = 0.80,
            second_leg_size = 0.03,
            third_leg_size = 0.03,
            pretrain = False, 
          ):

    if pretrain == False:
        density = 2500
    else:
        density = 10    
    R = RobotGraph(name=name)
    backpack_width = 0.19
    backpack_thick = 0.09
 

    root = RobotLink('root',link_type = 'box',  size=[backpack_thick,backpack_width/2,backpack_width/2],body_pos=[-.07, 0, 0.05],geom_pos=[0,0,0],euler=[0,0,180])
    R.add_node( node_type='link',node_info = root)

    # 添加一条腿
    # 添加joint01 02, 髋关节2个自由度
    joint01 = RobotJoint('joint01',axis=[0,0,1],)
    R.add_node( node_type='joint', node_info=joint01)
    R.add_edge(started_node='root',ended_node='joint01')
    joint02 = RobotJoint('joint02',axis=[0,1,0],)
    R.add_node( node_type='joint', node_info=joint02)
    R.add_edge(started_node='root',ended_node='joint02')
    # 添加大腿
    leg1 = RobotLink('leg1',length=first_leg_lenth,  size=first_leg_size,body_pos=[backpack_thick*3/4,-backpack_width/2,0],euler=[0,0,-90])    
    R.add_node( node_type='link', node_info=leg1)
    R.add_edge(started_node='joint01',ended_node='leg1')
    R.add_edge(started_node='joint02',ended_node='leg1')
    # 添加joint1
    joint1 = RobotJoint('joint1',axis=[0,1,0],)
    R.add_node( node_type='joint', node_info=joint1)
    R.add_edge( started_node='leg1',ended_node='joint1')
    # 添加小腿
    shin1 = RobotLink('shin1',length=second_leg_lenth, size=second_leg_size,body_pos=[first_leg_lenth,0,0],euler=[0,90,0])    
    R.add_node( node_type='link', node_info=shin1)
    R.add_edge(started_node='joint1',ended_node='shin1')
    # 添加joint2
    joint2 = RobotJoint('joint2',axis=[0,1,0],)
    R.add_node( node_type='joint', node_info=joint2)
    R.add_edge(started_node='shin1',ended_node='joint2')
    # 添加末端肢体
    shin2 = RobotLink('right_end',link_type='sphere',geom_pos=[0,0,0],size=third_leg_size*1.2,body_pos=[second_leg_lenth,0,0],euler=[0,-60,0])    
    # shin2 = RobotLink('SRL_right_end',length=third_leg_lenth ,size=third_leg_size,body_pos=[second_leg_lenth,0,0],euler=[0,-60,0])    
    R.add_node( node_type='link', node_info=shin2)
    R.add_edge(started_node='joint2',ended_node='right_end')

    # 添加第二条腿
    # 添加joint01, 髋关节
    joint11 = RobotJoint('joint11',axis=[0,0,1],)
    R.add_node( node_type='joint', node_info=joint11)
    R.add_edge(started_node='root',ended_node='joint11')
    joint12 = RobotJoint('joint12',axis=[0,1,0],)
    R.add_node( node_type='joint', node_info=joint12)
    R.add_edge(started_node='root',ended_node='joint12')
    # 添加大腿
    leg2 = RobotLink('leg2',length=first_leg_lenth, size=first_leg_size,body_pos=[backpack_thick*3/4,backpack_width/2,0],euler=[0,0,90])     
    R.add_node( node_type='link', node_info=leg2)
    R.add_edge(started_node='joint11',ended_node='leg2')
    R.add_edge(started_node='joint12',ended_node='leg2')
    # 添加joint1
    joint13 = RobotJoint('joint13',axis=[0,1,0],)
    R.add_node( node_type='joint', node_info=joint13)
    R.add_edge(started_node='leg2',ended_node='joint13')
    # 添加小腿
    shin11 = RobotLink('shin11',length=second_leg_lenth, size=second_leg_size,body_pos=[first_leg_lenth,0,0],euler=[0,90,0])    
    R.add_node( node_type='link', node_info=shin11)
    R.add_edge(started_node='joint13',ended_node='shin11')
    # 添加joint2
    joint14 = RobotJoint('joint14',axis=[0,1,0],)
    R.add_node( node_type='joint', node_info=joint14)
    R.add_edge(started_node='shin11',ended_node='joint14')
    # 添加末端肢体
    shin12 = RobotLink('left_end',link_type='sphere',geom_pos=[0,0,0],size=third_leg_size*1.2,body_pos=[second_leg_lenth,0,0],euler=[0,-60,0])    
    # shin12 = RobotLink('SRL_left_end',length=third_leg_lenth, size=third_leg_size,body_pos=[second_leg_lenth,0,0],euler=[0,-60,0])    
    R.add_node(node_type='link', node_info=shin12)
    R.add_edge(started_node='joint14',ended_node='left_end')


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

    R = SRL_mode2(name='humanoid_srl_mode2')
    M = ModelGenerator(R)
    M.gen_basic_humanoid_xml()
    M.get_SRL_dfs(back_load=True)
    M.generate()
