from RobotGraph import RobotGraph,RobotJoint,RobotLink
from mjcf import elements as e
import networkx as nx
import matplotlib.pyplot as plt
import queue

class ModelGenerator():
    '''
    基于图结构的模型生成器
    对于所有设定和参数,只会生成非默认值
    '''
    def __init__(self,
                 robot:RobotGraph , 
                 path:str = 'mjcf_model/'  , # 存储路径
                 ):
        self.robot = robot # 机器人图
        self.model = e.Mujoco(model=self.robot.graph['name']) # mjcf生成器
        self.path = path
        # 初始化xml文档结构
        self.compiler = e.Compiler()
        self.option = e.Option()
        self.size = e.Size()
        self.custom = e.Custom()
        self.default = e.Default()
        self.asset = e.Asset()
        self.worldbody = e.Worldbody()
        self.actuator = e.Actuator()
        self.model.add_children([self.compiler,
                                 self.option,
                                 self.size,
                                 self.custom,
                                 self.default,
                                 self.asset,
                                 self.worldbody,
                                 self.actuator])
    def set_basic_assets(self,):
        '''
        添加基本的素材：纹理，地面
        TODO 
        '''
        tex1 = e.Texture(
            builtin="gradient",
            height=100,
            rgb1=[1, 1, 1],
            rgb2=[0, 0, 0],
            type="skybox",
            width=100
        )
        tex2 = e.Texture(
            builtin="flat",
            height=1278,
            mark="cross",
            markrgb=[1, 1, 1],
            name="texgeom",
            random=0.01,
            rgb1=[0.8, 0.6, 0.4],
            rgb2=[0.8, 0.6, 0.4],
            type="cube",
            width=127
        )
        tex3 = e.Texture(
            builtin="checker",
            height=[100],
            name="texplane",
            rgb1=[0, 0, 0],
            rgb2=[0.8, 0.8, 0.8],
            type="2d",
            width=100
        )
        mat1 = e.Material(
            name="MatPlane",
            reflectance=0.5,
            shininess=1,
            specular=1,
            texrepeat=[60, 60],
            texture="texplane"
        )
        mat2 = e.Material(
            name="geom",
            texture="texgeom",
            texuniform=True
        )
        self.asset.add_children([
            tex1,
            tex2,
            tex3,
            mat1,
            mat2,
        ])
    
    def set_default(self,):
            # Default
        default_joint = e.Joint(
            armature=1,
            damping=1,
            limited=True
        )
        d_geom = e.Geom(
            conaffinity=0,
            condim=3,
            margin=0.01,
            rgba=[0.8, 0.6, 0.4, 1]
        )
        self.default.add_children([default_joint, d_geom])

    def set_compiler(self, angle = 'radian', eulerseq = 'xyz',**kwargs):
        self.compiler.angle = angle
        self.compiler.eulerseq = eulerseq

    def set_size(self, njmax = 1000, nconmax = 500 ):
        self.size.njmax = njmax
        self.size.nconmax = nconmax

    def set_option(self, gravity = -9.81):
        self.option.gravity = [0 ,0, gravity]

    def set_ground(self):
        '''
        添加灯光 场地
        '''
        light = e.Light(
            cutoff=100,
            diffuse=[1, 1, 1],
            dir=[-0, 0, -1.3],
            directional=True,
            exponent=1,
            pos=[0, 0, 1.3],
            specular=[.1, .1, .1]
        )
        floor_geom = e.Geom(
            conaffinity=1,
            condim=3,
            material="MatPlane",
            name="floor",
            pos=[0, 0, 0],
            rgba=[0.8, 0.9, 0.8, 1],
            size=[40, 40, 40],
            type="plane"
        )
        self.worldbody.add_children([
                    light,
                    floor_geom
                ])
    def get_link(self, robot_part : RobotLink):
        '''
        添加一个机器人部件的几何体
        return: 该部件的body和geom
        '''
        body =  e.Body(
                name=robot_part.name,
                pos=robot_part.body_pos,
                euler=robot_part.euler
                )
        start_point = list(robot_part.start_point)
        end_point = [start_point[0]+robot_part.length, start_point[1]+0,start_point[2]+0]
        start_point.extend(end_point)
        if robot_part.link_type == 'capsule': # 如果是胶囊形状
            geom =  e.Geom(
                    fromto = start_point,
                    name   = "geom"+robot_part.name,
                    size   = robot_part.size,
                    type   = robot_part.link_type
                        )
        if robot_part.link_type == 'sphere': # 如果是胶囊形状
            geom =  e.Geom(
                    pos = robot_part.geom_pos,
                    name   = "geom"+robot_part.name,
                    size   = robot_part.size,
                    type   = robot_part.link_type
                        )
        if robot_part.link_type == 'box': # 如果是box形状
            geom =  e.Geom(
                    pos = robot_part.geom_pos,
                    name   = "geom"+robot_part.name,
                    size   = robot_part.size,
                    type   = robot_part.link_type
                        )
        return body,geom
        
    def get_joint(self, robot_joint: RobotJoint) :
        '''
        添加一个机器人部件的关节
        return: 该部件的joint
        '''
        joint = e.Joint(
                        axis=robot_joint.axis,
                        name="joint_"+robot_joint.name,
                        pos=robot_joint.pos,
                        range=robot_joint.joint_range,
                        type=robot_joint.joint_type
                            )
        if robot_joint.armature != None:
            joint.armature = robot_joint.armature
        if robot_joint.stiffness != None:
            joint.stiffness = robot_joint.stiffness

        actuator  = e.Motor(
                        ctrllimited=robot_joint.ctrllimited,
                        ctrlrange= robot_joint.ctrlrange,
                        joint="joint_"+robot_joint.name,
                        
                    )
        return joint,actuator

    def get_robot_dfs(self,):
        robot_graph = self.robot
        node_stack = queue.LifoQueue() # 储存图节点 栈
        body_stack = queue.LifoQueue() # 储存xmlbody 栈
        joint_list = []
        node_list  = []
        # 先创建root body
        if 'root' not in robot_graph.nodes :
            raise ValueError('The robot graph does not have root node.') 
        rootbody, rootgeom = self.get_link(robot_graph.nodes['root']['info'])
        rootbody.add_child(rootgeom)
        for n in robot_graph.successors('root'):
            if robot_graph.nodes[n]['type'] == 'link':
                if n not in node_list:
                    node_stack.put(n) # 压栈
                    body_stack.put(rootbody) # 压栈   
                    node_list.append(n)
            if robot_graph.nodes[n]['type'] == 'joint':
                for p in robot_graph.successors(n):
                    if p not in node_list:
                        node_stack.put(p) # 压栈
                        body_stack.put(rootbody) # 压栈   
                        node_list.append(p) 

        while not node_stack.empty(): # 当栈不为空
            current_node = node_stack.get() # 栈顶元素
            current_father_body = body_stack.get() # 栈顶元素
            if robot_graph.nodes[current_node]['type'] == 'link':
                body,geom = self.get_link(robot_graph.nodes[current_node]['info'])
                body.add_child(geom)    
                # 添加该节点上方的joint节点
                pres = list(robot_graph.predecessors(current_node))[0]
                if robot_graph.nodes[pres]['type'] == 'joint':
                    for p in robot_graph.predecessors(current_node):
                        if p in joint_list: 
                            continue
                        joint,actuator = self.get_joint(robot_graph.nodes[p]['info'])
                        body.add_child(joint)
                        joint_list.append(p) # 将该关节结点记录，防止重复添加 
                        self.actuator.add_child(actuator) # 添加驱动
            current_father_body.add_child(body)
            # 继续下一个节点
            if len(list(robot_graph.successors(current_node))) == 0:
                # 若无子节点，继续循环
                continue
            else:
                for n in robot_graph.successors(current_node):
                    if robot_graph.nodes[n]['type'] == 'link':
                        if n not in node_list:
                            node_stack.put(n) # 压栈
                            body_stack.put(body) # 压栈   
                            node_list.append(n)
                    if robot_graph.nodes[n]['type'] == 'joint':
                        for p in robot_graph.successors(n):
                            if p not in node_list:
                                node_stack.put(p) # 压栈
                                body_stack.put(body) # 压栈   
                                node_list.append(p) 
        self.worldbody.add_child(rootbody)     
        
    def get_robot(self,robot_graph:RobotGraph):
        '''
        从机器人图生成一个机器人
        '''     
        if 'root' not in robot_graph.nodes :
            raise ValueError('The robot graph does not have root node.') 
        rootbody, rootgeom = self.get_link(robot_graph.nodes['root']['info'])
        rootbody.add_child(rootgeom)
        if len(list(robot_graph.successors('root'))) == 0:
            raise ValueError(f'The robot graph is empty.')
        joint_list = []

        for n in robot_graph.successors('root'):
            if n in joint_list:
                continue # 防止重复添加
            # 遍历root的每一个分支
            first_body = True # 该分支最上层的body
            current_node = n # 每个node为一个dict
            has_joint = False # 标记是否遍历到joint
            last_body = []
            while True:
                if robot_graph.nodes[current_node]['type'] == 'link':
                    body,geom = self.get_link(robot_graph.nodes[current_node]['info'])
                    body.add_child(geom)
                    if has_joint:
                        # 如果有关节需要添加,则遍历该节点的父节点，依次添加关节
                        for p in robot_graph.predecessors(current_node):
                            joint,actuator = self.get_joint(robot_graph.nodes[p]['info'])
                            body.add_child(joint)
                            joint_list.append(p) # 将该关节结点记录，防止重复添加 
                            self.actuator.add_child(actuator) # 添加驱动
                        has_joint = False
                                      
                    if first_body: # 如果当前分支最上层body为空，则该body为最上层
                        last_body = body
                        rootbody.add_child(last_body)
                        first_body = False
                    else:
                        last_body.add_child(body)
                        last_body = body
                    # 继续下一个节点
                    if len(list(robot_graph.successors(current_node))) == 0:
                        break # 当前分支无子节点，退出
                    next_node = list(robot_graph.successors(current_node))[0]
                    current_node = next_node 
                    continue
                if robot_graph.nodes[current_node]['type'] == 'joint':
                    # 如果当前节点是joint类型节点，继续到下一个节点
                    next_node = list(robot_graph.successors(current_node))[0]
                    has_joint = True
                    current_node = next_node 
                    continue
        self.worldbody.add_child(rootbody)

    def generate(self):
        '''
        输出xml文档
        '''
        model_xml = self.model.xml()
        save_path = self.path + self.robot.graph['name'] + '.xml'
        # Output
        with open(save_path, 'w') as fh:
            fh.write(model_xml)


if __name__ == '__main__':

    R = RobotGraph(name='antrobot_dfs')

    root = RobotLink('root',link_type = 'sphere',size=0.25,body_pos=[0,0,2],geom_pos=[0,0,0])
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
    leg1 = RobotLink('leg1',length=0.5,size=0.1,body_pos=[0.25,0,0],euler=[0,0,0])    
    R.add_node( node_type='link', node_info=leg1)
    R.add_edge(started_node='joint01',ended_node='leg1')
    R.add_edge(started_node='joint02',ended_node='leg1')
    # 添加joint11和12，leg1的子节点，为关节
    joint1 = RobotJoint('joint1',axis=[0,1,0],)
    R.add_node( node_type='joint', node_info=joint1)
    R.add_edge(started_node='leg1',ended_node='joint1')
    # 添加小腿
    shin1 = RobotLink('shin1',length=0.4,size=0.1,body_pos=[0.5,0,0],euler=[0,90,0])    
    R.add_node( node_type='link', node_info=shin1)
    R.add_edge(started_node='joint1',ended_node='shin1')

    # 添加第二条腿
    leg2 = RobotLink('leg2',length=0.5,size=0.1,body_pos=[0,0.25,0],euler=[0,0,90])    
    R.add_node( node_type='link', node_info=leg2)
    R.add_edge(started_node='root',ended_node='leg2')
    # 添加joint11和12，leg1的子节点，为关节
    joint2 = RobotJoint('joint2',axis=[0,1,0],)
    R.add_node( node_type='joint', node_info=joint2)
    R.add_edge(started_node='leg2',ended_node='joint2')
    # 添加小腿
    shin2 = RobotLink('shin2',length=0.4,size=0.1,body_pos=[0.5,0,0],euler=[0,90,0])    
    R.add_node( node_type='link', node_info=shin2)
    R.add_edge(started_node='joint2',ended_node='shin2')

    # 添加第三条腿
    leg3 = RobotLink('leg3',length=0.5,size=0.1,body_pos=[-0.25,0,0],euler=[0,0,180])    
    R.add_node( node_type='link', node_info=leg3)
    R.add_edge(started_node='root',ended_node='leg3')
    # 添加joint11和12，leg1的子节点，为关节
    joint3 = RobotJoint('joint3',axis=[0,1,0],)
    R.add_node( node_type='joint', node_info=joint3)
    R.add_edge(started_node='leg3',ended_node='joint3')
    # 添加小腿
    shin3 = RobotLink('shin3',length=0.4,size=0.1,body_pos=[0.5,0,0],euler=[0,90,0])    
    R.add_node( node_type='link', node_info=shin3)
    R.add_edge(started_node='joint3',ended_node='shin3')

    # 添加第四条腿
    leg4 = RobotLink('leg4',length=0.5,size=0.1,body_pos=[0,-0.25,0],euler=[0,0,270])    
    R.add_node( node_type='link', node_info=leg4)
    R.add_edge(started_node='root',ended_node='leg4')
    # 添加joint11和12，leg1的子节点，为关节
    joint4 = RobotJoint('joint4',axis=[0,1,0],)
    R.add_node( node_type='joint', node_info=joint4)
    R.add_edge(started_node='leg4',ended_node='joint4')
    # 添加小腿
    shin4 = RobotLink('shin4',length=0.4,size=0.1,body_pos=[0.5,0,0],euler=[0,90,0])    
    R.add_node( node_type='link', node_info=shin4)
    R.add_edge(started_node='joint4',ended_node='shin4')


    M = ModelGenerator(R)
    M.set_compiler(angle='degree')
    M.set_basic_assets()
    M.set_size()
    M.set_option(gravity=0)
    M.set_default()
    M.set_ground()
    #M.get_robot(R)
    M.get_robot_dfs()

    M.generate()
    print(M.compiler.angle)

    for layer, nodes in enumerate(nx.topological_generations(R)):
    # `multipartite_layout` expects the layer as a node attribute, so add the
    # numeric layer value as a node attribute
        for node in nodes:
            R.nodes[node]["layer"] = layer

    # Compute the multipartite_layout using the "layer" node attribute
    pos = nx.multipartite_layout(R, subset_key="layer")

    fig, ax = plt.subplots()
    nx.draw_networkx(R, ax=ax,pos=pos)
    ax.set_title("DAG layout in topological order")
    fig.tight_layout()
    plt.show()



