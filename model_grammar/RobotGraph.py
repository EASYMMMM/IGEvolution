import networkx as nx
from networkx import DiGraph
from typing import Union


class RobotLink():
    '''
    机器人的link
    body_pos: body坐标系的位置
    euler: body坐标系旋转的欧拉角
    geom_pos: 几何体位置
    '''
    def __init__(self,
                 name,
                 length = 0,
                 size = 0,
                 link_type = 'capsule',
                 start_point = [0,0,0], # 起点，capsule用
                 geom_pos = [0,0,0], # 几何体位置，sphere用
                 body_pos = [0,0,0],
                 euler = [0,0,0]  # body坐标系旋转的欧拉角
                ):
        self.name = name
        self.length = length
        self.size = size
        self.link_type = link_type
        self.start_point = start_point
        self.end_point = start_point + [self.length,0,0]
        self.body_pos = body_pos
        self.euler = euler
        self.geom_pos = geom_pos


class RobotJoint():
    '''
    机器人的关节（单自由度）
    '''
    def __init__(self,
                 name = 'abdomen_z',
                 joint_type = 'hinge',
                 axis = [0,0,1], # 默认Z轴
                 pos = [0,0,0],  # 关节位置，默认在body坐标系的原点
                 joint_range = [-45,45],
                 stiffness = None,
                 armature = None,  
                 ctrlrange = [-1,1],
                 ctrllimited = True):
        self.name = name
        self.joint_type = joint_type
        self.armature = armature
        self.axis = axis
        self.pos = pos
        self.joint_range = joint_range
        self.stiffness = stiffness
        self.ctrlrange = ctrlrange
        self.ctrllimited = ctrllimited


class RobotGraph(DiGraph):

    def __init__(self, name = 'RobotGraph'):
        super().__init__(self,name=name)
    
    def add_node(self,  node_type:str =None, node_info:Union[RobotJoint, RobotLink]=None):
        '''
        在有向图中构建一个节点
        Node:
        {   name : name                   节点名 
            type : joint / link           该节点为连杆节点或关节节点
            info : RobotJoint / RobotLink }  
        '''
        if not node_type in ['joint','link']:
            raise ValueError(f"Unknowen node type: {node_type}. It should be a 'joint' or 'link'.")

        super().add_node(node_info.name, type = node_type, info = node_info)
    
    def add_edge(self, started_node, ended_node, **attr):
        
        return super().add_edge(started_node, ended_node, **attr)

    def add_root(self, node_name:str = 'root', node_type:str = 'link',node_info:RobotJoint = None):
        # 根节点 有且只能有一个
        super().add_node(node_name, type = node_type, info = node_info)
    

if __name__ == '__main__':
    print('= = = = = = = = = = = =')
    R = RobotGraph(name='xmlrobot')
    root = RobotLink('root',length=5,size=10)
    R.add_node( node_type='link',node_info = root)
    # 添加part1，root子节点
    part1 = RobotLink('part1',length=10,size=3)    
    R.add_node( node_type='link', node_info=part1)
    R.add_edge(started_node='root',ended_node='part1')
    # 添加part2，root子节点
    part2 = RobotLink('part2',length=15,size=4)    
    R.add_node( node_type='link', node_info=part2)
    R.add_edge(started_node='root',ended_node='part2')

    # 添加joint11和12，part1的子节点，为关节
    joint11 = RobotJoint('joint11',axis=[0,0,1])
    R.add_node( node_type='joint', node_info=joint11)
    R.add_edge(started_node='part1',ended_node='joint11')
    joint12 = RobotJoint('joint12',axis=[0,1,0])
    R.add_node( node_type='joint', node_info=joint12)
    R.add_edge(started_node='part1',ended_node='joint12')

    # 添加part3，joint11和12的子节点
    part3 = RobotLink('part3',length=12,size=2)    
    R.add_node( node_type='link', node_info=part3)
    R.add_edge(started_node='joint11',ended_node='part3')
    R.add_edge(started_node='joint12',ended_node='part3')

    
    print('机器人图的名字：')
    print(R.graph['name'])
    print('root 的长度')
    print(R.nodes['root']['info'].length)
    print('part1 的长度')
    print(R.nodes['part1']['info'].length)

    print('root 的子节点：')
    for i in R.successors('root'):
        print(i)
        print(R.nodes[i]['info'].name)

    print('part1 的子节点：')
    print(list(R.successors('part1')))
    if len(list(R.successors('part1'))) == 0:
        print('part1 has no succ')
    for i in R.successors('part1'):
        print(i)
        print(R.nodes[i]['info'])  
    if 'r2' not in R.nodes:
        print('R has no node r2')

    print('part3的父节点：')
    for i in R.predecessors('part3'):
        print(i)
        print(R.nodes[i]['info'].axis)
