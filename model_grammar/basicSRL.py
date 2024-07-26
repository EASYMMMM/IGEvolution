from RobotModelGen import ModelGenerator
from RobotGraph import RobotGraph,RobotJoint,RobotLink

R = RobotGraph(name='srlrobot')

root = RobotLink('root',link_type = 'box',size=[0.05,0.35,0.35],body_pos=[0,0,2],geom_pos=[0,0,0])
R.add_node( node_type='link',node_info = root)

# 添加一条腿
# 添加joint01, 髋关节
joint01 = RobotJoint('joint01',axis=[0,0,1],)
R.add_node( node_type='joint', node_info=joint01)
R.add_edge(started_node='root',ended_node='joint01')
joint02 = RobotJoint('joint02',axis=[0,1,0],)
R.add_node( node_type='joint', node_info=joint02)
R.add_edge(started_node='root',ended_node='joint02')
# 添加大腿
leg1 = RobotLink('leg1',length=0.5,size=0.1,body_pos=[0.05,-0.15,0],euler=[0,0,0])    
R.add_node( node_type='link', node_info=leg1)
R.add_edge(started_node='joint01',ended_node='leg1')
R.add_edge(started_node='joint02',ended_node='leg1')
# 添加joint1
joint1 = RobotJoint('joint1',axis=[0,1,0],)
R.add_node( node_type='joint', node_info=joint1)
R.add_edge(started_node='leg1',ended_node='joint1')
# 添加小腿
shin1 = RobotLink('shin1',length=1.0,size=0.1,body_pos=[0.5,0,0],euler=[0,90,0])    
R.add_node( node_type='link', node_info=shin1)
R.add_edge(started_node='joint1',ended_node='shin1')
# 添加joint2
joint2 = RobotJoint('joint2',axis=[0,1,0],)
R.add_node( node_type='joint', node_info=joint2)
R.add_edge(started_node='shin1',ended_node='joint2')
# 添加末端肢体
shin2 = RobotLink('shin2',length=0.4,size=0.1,body_pos=[1.0,0,0],euler=[0,-90,0])    
R.add_node( node_type='link', node_info=shin2)
R.add_edge(started_node='joint2',ended_node='shin2')

# 添加第二条腿
# 添加joint01, 髋关节
joint11 = RobotJoint('joint11',axis=[0,0,1],)
R.add_node( node_type='joint', node_info=joint11)
R.add_edge(started_node='root',ended_node='joint11')
joint12 = RobotJoint('joint12',axis=[0,1,0],)
R.add_node( node_type='joint', node_info=joint12)
R.add_edge(started_node='root',ended_node='joint12')
# 添加大腿
leg2 = RobotLink('leg2',length=0.5,size=0.1,body_pos=[0.05,0.15,0],euler=[0,0,0])    
R.add_node( node_type='link', node_info=leg2)
R.add_edge(started_node='joint11',ended_node='leg2')
R.add_edge(started_node='joint12',ended_node='leg2')
# 添加joint1
joint13 = RobotJoint('joint13',axis=[0,1,0],)
R.add_node( node_type='joint', node_info=joint13)
R.add_edge(started_node='leg2',ended_node='joint13')
# 添加小腿
shin11 = RobotLink('shin11',length=1.0,size=0.1,body_pos=[0.5,0,0],euler=[0,90,0])    
R.add_node( node_type='link', node_info=shin11)
R.add_edge(started_node='joint13',ended_node='shin11')
# 添加joint2
joint14 = RobotJoint('joint14',axis=[0,1,0],)
R.add_node( node_type='joint', node_info=joint14)
R.add_edge(started_node='shin11',ended_node='joint14')
# 添加末端肢体
shin12 = RobotLink('shin12',length=0.4,size=0.1,body_pos=[1.0,0,0],euler=[0,-90,0])    
R.add_node( node_type='link', node_info=shin12)
R.add_edge(started_node='joint14',ended_node='shin12')


M = ModelGenerator(R)
M.set_compiler(angle='degree')
M.set_basic_assets()
M.set_size()
M.set_option(gravity=0)
M.set_default()
M.set_ground()
M.get_robot(R)

M.generate()
print(M.compiler.angle)