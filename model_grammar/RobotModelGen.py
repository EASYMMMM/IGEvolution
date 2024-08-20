from .RobotGraph import RobotGraph,RobotJoint,RobotLink
from .mjcf import elements as e
import networkx as nx
import matplotlib.pyplot as plt
import queue
from scipy.spatial.transform import Rotation 
import os
 
def euler2quaternion(euler):
    '''
    欧拉角转四元数
    '''
    r = Rotation.from_euler('xyz', euler, degrees=True)
    quaternion = r.as_quat()
    return quaternion
 

class ModelGenerator():
    '''
    基于图结构的模型生成器
    对于所有设定和参数,只会生成非默认值
    '''
    def __init__(self,
                 robot:RobotGraph , 
                 save_path:str = 'mjcf_model/'  , # 存储路径
                 model_name = 'XMLModel'
                 ):
        self.robot = robot # 机器人图
        self.model = e.Mujoco(model=model_name) # mjcf生成器
        self.save_path = save_path
        # 初始化xml文档结构
        # self.compiler = e.Compiler()
        self.statistic = e.Statistic()
        self.option = e.Option()
        # self.size = e.Size()
        # self.custom = e.Custom()
        self.default = e.Default()
        self.asset = e.Asset()
        self.worldbody = e.Worldbody()
        self.actuator = e.Actuator()
        self.sensor = e.Sensor()
        self.model.add_children([self.statistic,
                                 self.option,
                                 self.default,
                                 self.asset,
                                 self.worldbody,
                                 self.actuator,
                                 self.sensor])
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
        # 添加motor元素
        motor = e.Motor(ctrlrange=[-1, 1], ctrllimited=True)
        self.default.add_child(motor)

        # 创建并添加<body>的默认设置
        body_default = e.Default(class_="body")
        geom = e.Geom(type="capsule", condim=1, friction=[1.0, 0.05, 0.05], solimp=[0.9, 0.99, 0.003], solref=[0.015, 1])
        joint = e.Joint(type="hinge", damping=0.1, stiffness=5, armature=0.007, limited=True, solimplimit=[0, 0.99, 0.01])
        site = e.Site(size=0.04, group=3)
        body_default.add_children([geom, joint, site])

        # 添加force-torque的默认设置
        force_torque_default = e.Default(class_="force-torque")
        force_torque_site = e.Site(type="box", size=[0.01, 0.01, 0.02], rgba=[1, 0, 0, 1])
        force_torque_default.add_child(force_torque_site)
        body_default.add_child(force_torque_default)

        # 添加touch的默认设置
        touch_default = e.Default(class_="touch")
        touch_site = e.Site(type="capsule", rgba=[0, 0, 1, 0.3])
        touch_default.add_child(touch_site)
        body_default.add_child(touch_default)

        # 将body_default添加到default_element中
        self.default.add_child(body_default)

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
        quat_np = euler2quaternion(robot_part.euler)
        quat = [0.00,0.00,0.00,0.00]
        i = 1
        for q in quat_np:
            quat[i] = round(q,5)
            i = i+1
            if i > 3:
                i=0
        body =  e.Body(
                name="SRL_"+robot_part.name,
                pos=robot_part.body_pos,
                quat=quat
                )
        if robot_part.link_type == 'capsule': # 如果是胶囊形状
            start_point = list(robot_part.start_point)
            end_point = [start_point[0]+robot_part.length, start_point[1]+0,start_point[2]+0]
            start_point.extend(end_point)
            geom =  e.Geom(
                    fromto = start_point,
                    name   = "SRL_geom_"+robot_part.name,
                    size   = robot_part.size,
                    type   = robot_part.link_type
                        )
        if robot_part.link_type == 'sphere': # 如果是胶囊形状
            geom =  e.Geom(
                    pos = robot_part.geom_pos,
                    name   = "SRL_geom_"+robot_part.name,
                    size   = robot_part.size,
                    type   = robot_part.link_type
                        )
        if robot_part.link_type == 'box': # 如果是box形状
            geom =  e.Geom(
                    pos = robot_part.geom_pos,
                    name   = "SRL_geom_"+robot_part.name,
                    size   = robot_part.size,
                    type   = robot_part.link_type
                        )
        if robot_part.material != None:   # 添加几何体材料
            geom.material = robot_part.material
        if robot_part.density != None:
            geom.density = robot_part.density
        return body,geom
        
    def get_joint(self, robot_joint: RobotJoint) :
        '''
        添加一个机器人部件的关节
        return: 该部件的joint
        '''
        joint = e.Joint(
                        axis=robot_joint.axis,
                        name="SRL_joint_"+robot_joint.name,
                        pos=robot_joint.pos,
                        range=robot_joint.joint_range,
                        type=robot_joint.joint_type
                            )
        if robot_joint.armature != None:
            joint.armature = robot_joint.armature
        if robot_joint.stiffness != None:
            joint.stiffness = robot_joint.stiffness

        actuator  = e.Motor(name = "SRL_joint_"+robot_joint.name,
                            joint="SRL_joint_"+robot_joint.name,
                            gear=125)
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
            print(robot_graph.nodes[current_node])
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

    def get_humanoid_model(self):
        # 创建<worldbody>元素
        #worldbody = e.Worldbody()

        # 添加floor geom
        floor_geom = e.Geom(name="floor", type="plane", conaffinity=1, size=[100, 100, 0.2] )
        self.worldbody.add_child(floor_geom)

        # 创建pelvis body及其子元素
        self.pelvis_body = e.Body(name="pelvis", pos=[0, 0, 1], childclass="body")
        self.pelvis_body.add_children([
            e.Freejoint(name="root"),
            e.Site(name="root", class_="force-torque"),
            e.Geom(name="pelvis", type="sphere", pos=[0, 0, 0.07], size=[0.09], density=2226),
            e.Geom(name="upper_waist", type="sphere", pos=[0, 0, 0.205], size=[0.07], density=2226),
            e.Site(name="pelvis", class_="touch", type="sphere", pos=[0, 0, 0.07], size=[0.091]),
            e.Site(name="upper_waist", class_="touch", type="sphere", pos=[0, 0, 0.205], size=[0.071])
        ])

        # 创建torso body及其子元素
        torso_body = e.Body(name="torso", pos=[0, 0, 0.236151])
        torso_body.add_children([
            e.Light(name="top", pos=[0, 0, 2], mode="trackcom"),
            e.Camera(name="back", pos=[-3, 0, 1], xyaxes=[0, -1, 0, 1, 0, 2], mode="trackcom"),
            e.Camera(name="side", pos=[0, -3, 1], xyaxes=[1, 0, 0, 0, 1, 2], mode="trackcom"),
            e.Joint(name="abdomen_x", pos=[0, 0, 0], axis=[1, 0, 0], range=[-60, 60], stiffness=600, damping=60, armature=0.025),
            e.Joint(name="abdomen_y", pos=[0, 0, 0], axis=[0, 1, 0], range=[-60, 90], stiffness=600, damping=60, armature=0.025),
            e.Joint(name="abdomen_z", pos=[0, 0, 0], axis=[0, 0, 1], range=[-50, 50], stiffness=600, damping=60, armature=0.025),
            e.Geom(name="torso", type="sphere", pos=[0, 0, 0.12], size=[0.11], density=1794),
            e.Site(name="torso", class_="touch", type="sphere", pos=[0, 0, 0.12], size=[0.111]),
            e.Geom(name="right_clavicle", fromto=[-0.0060125, -0.0457775, 0.2287955, -0.016835, -0.128177, 0.2376182], size=[0.045], density=1100),
            e.Geom(name="left_clavicle", fromto=[-0.0060125, 0.0457775, 0.2287955, -0.016835, 0.128177, 0.2376182], size=[0.045], density=1100)
        ])

        # 创建head body及其子元素
        head_body = e.Body(name="head", pos=[0, 0, 0.223894])
        head_body.add_children([
            e.Joint(name="neck_x", axis=[1, 0, 0], range=[-50, 50], stiffness=50, damping=5, armature=0.017),
            e.Joint(name="neck_y", axis=[0, 1, 0], range=[-40, 60], stiffness=50, damping=5, armature=0.017),
            e.Joint(name="neck_z", axis=[0, 0, 1], range=[-45, 45], stiffness=50, damping=5, armature=0.017),
            e.Geom(name="head", type="sphere", pos=[0, 0, 0.175], size=[0.095], density=1081),
            e.Site(name="head", class_="touch", pos=[0, 0, 0.175], type="sphere", size=[0.103]),
            e.Camera(name="egocentric", pos=[0.103, 0, 0.175], xyaxes=[0, -1, 0, 0.1, 0, 1], fovy=80)
        ])

        # 创建right_upper_arm body及其子元素
        right_upper_arm_body = e.Body(name="right_upper_arm", pos=[-0.02405, -0.18311, 0.24350])
        right_upper_arm_body.add_children([
            e.Joint(name="right_shoulder_x", axis=[1, 0, 0], range=[-180, 45], stiffness=200, damping=20, armature=0.02),
            e.Joint(name="right_shoulder_y", axis=[0, 1, 0], range=[-180, 60], stiffness=200, damping=20, armature=0.02),
            e.Joint(name="right_shoulder_z", axis=[0, 0, 1], range=[-90, 90], stiffness=200, damping=20, armature=0.02),
            e.Geom(name="right_upper_arm", fromto=[0, 0, -0.05, 0, 0, -0.23], size=[0.045], density=982),
            e.Site(name="right_upper_arm", class_="touch", pos=[0, 0, -0.14], size=[0.046, 0.1], zaxis=[0, 0, 1])
        ])

        # 创建right_lower_arm body及其子元素
        right_lower_arm_body = e.Body(name="right_lower_arm", pos=[0, 0, -0.274788])
        right_lower_arm_body.add_children([
            e.Joint(name="right_elbow", axis=[0, 1, 0], range=[-160, 0], stiffness=150, damping=15, armature=0.015),
            e.Geom(name="right_lower_arm", fromto=[0, 0, -0.0525, 0, 0, -0.1875], size=[0.04], density=1056),
            e.Site(name="right_lower_arm", class_="touch", pos=[0, 0, -0.12], size=[0.041, 0.0685], zaxis=[0, 1, 0])
        ])

        # 创建right_hand body及其子元素
        right_hand_body = e.Body(name="right_hand", pos=[0, 0, -0.258947])
        right_hand_body.add_children([
            e.Geom(name="right_hand", type="sphere", size=[0.04], density=1865),
            e.Site(name="right_hand", class_="touch", type="sphere", size=[0.041])
        ])

        # 创建left_upper_arm body及其子元素
        left_upper_arm_body = e.Body(name="left_upper_arm", pos=[-0.02405, 0.18311, 0.24350])
        left_upper_arm_body.add_children([
            e.Joint(name="left_shoulder_x", axis=[1, 0, 0], range=[-45, 180], stiffness=200, damping=20, armature=0.02),
            e.Joint(name="left_shoulder_y", axis=[0, 1, 0], range=[-180, 60], stiffness=200, damping=20, armature=0.02),
            e.Joint(name="left_shoulder_z", axis=[0, 0, 1], range=[-90, 90], stiffness=200, damping=20, armature=0.02),
            e.Geom(name="left_upper_arm", fromto=[0, 0, -0.05, 0, 0, -0.23], size=[0.045], density=982),
            e.Site(name="left_upper_arm", class_="touch", pos=[0, 0, -0.14], size=[0.046, 0.1], zaxis=[0, 0, 1])
        ])

        # 创建left_lower_arm body及其子元素
        left_lower_arm_body = e.Body(name="left_lower_arm", pos=[0, 0, -0.274788])
        left_lower_arm_body.add_children([
            e.Joint(name="left_elbow", axis=[0, 1, 0], range=[-160, 0], stiffness=150, damping=15, armature=0.015),
            e.Geom(name="left_lower_arm", fromto=[0, 0, -0.0525, 0, 0, -0.1875], size=[0.04], density=1056),
            e.Site(name="left_lower_arm", class_="touch", pos=[0, 0, -0.1], size=[0.041, 0.0685], zaxis=[0, 0, 1])
        ])

        # 创建left_hand body及其子元素
        left_hand_body = e.Body(name="left_hand", pos=[0, 0, -0.258947])
        left_hand_body.add_children([
            e.Geom(name="left_hand", type="sphere", size=[0.04], density=1865),
            e.Site(name="left_hand", class_="touch", type="sphere", size=[0.041])
        ])

        # 将left_lower_arm和left_hand body添加到left_upper_arm body中
        left_upper_arm_body.add_child(left_lower_arm_body)
        left_lower_arm_body.add_child(left_hand_body)



        # 创建right_thigh body及其子元素
        right_thigh_body = e.Body(name="right_thigh", pos=[0, -0.084887, 0])
        right_thigh_body.add_children([
            e.Site(name="right_hip", class_="force-torque"),
            e.Joint(name="right_hip_x", axis=[1, 0, 0], range=[-60, 15], stiffness=300, damping=30, armature=0.02),
            e.Joint(name="right_hip_y", axis=[0, 1, 0], range=[-140, 60], stiffness=300, damping=30, armature=0.02),
            e.Joint(name="right_hip_z", axis=[0, 0, 1], range=[-60, 35], stiffness=300, damping=30, armature=0.02),
            e.Geom(name="right_thigh", fromto=[0, 0, -0.06, 0, 0, -0.36], size=[0.055], density=1269),
            e.Site(name="right_thigh", class_="touch", pos=[0, 0, -0.21], size=[0.056, 0.301], zaxis=[0, 0, -1])
        ])

        # 创建right_shin body及其子元素
        right_shin_body = e.Body(name="right_shin", pos=[0, 0, -0.421546])
        right_shin_body.add_children([
            e.Site(name="right_knee", class_="force-torque", pos=[0, 0, 0]),
            e.Joint(name="right_knee", pos=[0, 0, 0], axis=[0, 1, 0], range=[0, 160], stiffness=300, damping=30, armature=0.02),
            e.Geom(name="right_shin", fromto=[0, 0, -0.045, 0, 0, -0.355], size=[0.05], density=1014),
            e.Site(name="right_shin", class_="touch", pos=[0, 0, -0.2], size=[0.051, 0.156], zaxis=[0, 0, -1])
        ])

        # 创建right_foot body及其子元素
        right_foot_body = e.Body(name="right_foot", pos=[0, 0, -0.409870])
        right_foot_body.add_children([
            e.Site(name="right_ankle", class_="force-torque"),
            e.Joint(name="right_ankle_x", pos=[0, 0, 0], axis=[1, 0, 0], range=[-30, 30], stiffness=200, damping=20, armature=0.01),
            e.Joint(name="right_ankle_y", pos=[0, 0, 0], axis=[0, 1, 0], range=[-55, 55], stiffness=200, damping=20, armature=0.01),
            e.Joint(name="right_ankle_z", pos=[0, 0, 0], axis=[0, 0, 1], range=[-40, 40], stiffness=200, damping=20, armature=0.01),
            e.Geom(name="right_foot", type="box", pos=[0.045, 0, -0.0225], size=[0.0885, 0.045, 0.0275], density=1141),
            e.Site(name="right_foot", class_="touch", type="box", pos=[0.045, 0, -0.0225], size=[0.0895, 0.055, 0.0285])
        ])

        # 将right_shin和right_foot body添加到right_thigh body中
        right_thigh_body.add_child(right_shin_body)
        right_shin_body.add_child(right_foot_body)



        # 创建left_thigh body及其子元素
        left_thigh_body = e.Body(name="left_thigh", pos=[0, 0.084887, 0])
        left_thigh_body.add_children([
            e.Site(name="left_hip", class_="force-torque"),
            e.Joint(name="left_hip_x", axis=[1, 0, 0], range=[-15, 60], stiffness=300, damping=30, armature=0.02),
            e.Joint(name="left_hip_y", axis=[0, 1, 0], range=[-140, 60], stiffness=300, damping=30, armature=0.02),
            e.Joint(name="left_hip_z", axis=[0, 0, 1], range=[-35, 60], stiffness=300, damping=30, armature=0.02),
            e.Geom(name="left_thigh", fromto=[0, 0, -0.06, 0, 0, -0.36], size=[0.055], density=1269),
            e.Site(name="left_thigh", class_="touch", pos=[0, 0, -0.21], size=[0.056, 0.301], zaxis=[0, 0, -1])
        ])

        # 创建left_shin body及其子元素
        left_shin_body = e.Body(name="left_shin", pos=[0, 0, -0.421546])
        left_shin_body.add_children([
            e.Site(name="left_knee", class_="force-torque", pos=[0, 0, 0.02]),
            e.Joint(name="left_knee", pos=[0, 0, 0], axis=[0, 1, 0], range=[0, 160], stiffness=300, damping=30, armature=0.02),
            e.Geom(name="left_shin", fromto=[0, 0, -0.045, 0, 0, -0.355], size=[0.05], density=1014),
            e.Site(name="left_shin", class_="touch", pos=[0, 0, -0.2], size=[0.051, 0.156], zaxis=[0, 0, -1])
        ])

        # 创建left_foot body及其子元素
        left_foot_body = e.Body(name="left_foot", pos=[0, 0, -0.409870])
        left_foot_body.add_children([
            e.Site(name="left_ankle", class_="force-torque"),
            e.Joint(name="left_ankle_x", pos=[0, 0, 0], axis=[1, 0, 0], range=[-30, 30], stiffness=200, damping=20, armature=0.01),
            e.Joint(name="left_ankle_y", pos=[0, 0, 0], axis=[0, 1, 0], range=[-55, 55], stiffness=200, damping=20, armature=0.01),
            e.Joint(name="left_ankle_z", pos=[0, 0, 0], axis=[0, 0, 1], range=[-40, 40], stiffness=200, damping=20, armature=0.01),
            e.Geom(name="left_foot", type="box", pos=[0.045, 0, -0.0225], size=[0.0885, 0.045, 0.0275], density=1141),
            e.Site(name="left_foot", class_="touch", type="box", pos=[0.045, 0, -0.0225], size=[0.0895, 0.055, 0.0285])
        ])

        # 将left_shin和left_foot body添加到left_thigh body中
        left_thigh_body.add_child(left_shin_body)
        left_shin_body.add_child(left_foot_body)

        # 将所有子元素添加到对应的父元素中
        right_upper_arm_body.add_child(right_lower_arm_body)
        right_lower_arm_body.add_child(right_hand_body)
        torso_body.add_child(head_body)
        torso_body.add_child(right_upper_arm_body)
        torso_body.add_child(left_upper_arm_body)

        self.pelvis_body.add_child(torso_body)
        self.pelvis_body.add_child(right_thigh_body)
        self.pelvis_body.add_child(left_thigh_body)

        self.worldbody.add_child(self.pelvis_body)

        # 添加motor元素
        self.motor_list = [
            e.Motor(name='abdomen_x', gear=125, joint='abdomen_x'),
            e.Motor(name='abdomen_y', gear=125, joint='abdomen_y'),
            e.Motor(name='abdomen_z', gear=125, joint='abdomen_z'),
            e.Motor(name='neck_x', gear=20, joint='neck_x'),
            e.Motor(name='neck_y', gear=20, joint='neck_y'),
            e.Motor(name='neck_z', gear=20, joint='neck_z'),
            e.Motor(name='right_shoulder_x', gear=70, joint='right_shoulder_x'),
            e.Motor(name='right_shoulder_y', gear=70, joint='right_shoulder_y'),
            e.Motor(name='right_shoulder_z', gear=70, joint='right_shoulder_z'),
            e.Motor(name='right_elbow', gear=60, joint='right_elbow'),
            e.Motor(name='left_shoulder_x', gear=70, joint='left_shoulder_x'),
            e.Motor(name='left_shoulder_y', gear=70, joint='left_shoulder_y'),
            e.Motor(name='left_shoulder_z', gear=70, joint='left_shoulder_z'),
            e.Motor(name='left_elbow', gear=60, joint='left_elbow'),
            e.Motor(name='right_hip_x', gear=125, joint='right_hip_x'),
            e.Motor(name='right_hip_z', gear=125, joint='right_hip_z'),
            e.Motor(name='right_hip_y', gear=125, joint='right_hip_y'),
            e.Motor(name='right_knee', gear=100, joint='right_knee'),
            e.Motor(name='right_ankle_x', gear=50, joint='right_ankle_x'),
            e.Motor(name='right_ankle_y', gear=50, joint='right_ankle_y'),
            e.Motor(name='right_ankle_z', gear=50, joint='right_ankle_z'),
            e.Motor(name='left_hip_x', gear=125, joint='left_hip_x'),
            e.Motor(name='left_hip_z', gear=125, joint='left_hip_z'),
            e.Motor(name='left_hip_y', gear=125, joint='left_hip_y'),
            e.Motor(name='left_knee', gear=100, joint='left_knee'),
            e.Motor(name='left_ankle_x', gear=50, joint='left_ankle_x'),
            e.Motor(name='left_ankle_y', gear=50, joint='left_ankle_y'),
            e.Motor(name='left_ankle_z', gear=50, joint='left_ankle_z')
        ]
        self.actuator.add_children(self.motor_list)

    def get_SRL_dfs(self, back_load=False):
        '''
        根据图结构递归生成SRL机器人
        SRL附着在pelvis下
        '''
        robot_graph = self.robot
        node_stack  = queue.LifoQueue() # 储存图节点 栈
        body_stack  = queue.LifoQueue() # 储存xmlbody 栈
        joint_list  = []
        node_list   = []
        # 先创建root body
        if 'root' not in robot_graph.nodes :
            raise ValueError('The robot graph does not have root node.') 
        rootbody, rootgeom = self.get_link(robot_graph.nodes['root']['info'])
        rootbody.add_child(rootgeom)
        if back_load: # 背部负重
            load_geom =  e.Geom(
                    pos = [0.20, 0, 0.35],
                    name   = "SRL_geom_load" ,
                    size   = [0.13, 0.13, 0.22],
                    type   = "box",
                    density= 6000
                        )
            rootbody.add_child(load_geom)

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
            print(robot_graph.nodes[current_node]['info'].name)
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
        self.pelvis_body.add_child(rootbody)     

    def set_sensors(self):
        # 创建<sensor>元素
        sensor = e.Sensor()

        # 添加各类传感器元素
        sensors = [
            e.sensor.Subtreelinvel(name="pelvis_subtreelinvel", body="pelvis"),
            e.sensor.Accelerometer(name="root_accel", site="root"),
            e.sensor.Velocimeter(name="root_vel", site="root"),
            e.sensor.Gyro(name="root_gyro", site="root"),

            e.sensor.Force(name="left_ankle_force", site="left_ankle"),
            e.sensor.Force(name="right_ankle_force", site="right_ankle"),
            e.sensor.Force(name="left_knee_force", site="left_knee"),
            e.sensor.Force(name="right_knee_force", site="right_knee"),
            e.sensor.Force(name="left_hip_force", site="left_hip"),
            e.sensor.Force(name="right_hip_force", site="right_hip"),

            e.sensor.Torque(name="left_ankle_torque", site="left_ankle"),
            e.sensor.Torque(name="right_ankle_torque", site="right_ankle"),
            e.sensor.Torque(name="left_knee_torque", site="left_knee"),
            e.sensor.Torque(name="right_knee_torque", site="right_knee"),
            e.sensor.Torque(name="left_hip_torque", site="left_hip"),
            e.sensor.Torque(name="right_hip_torque", site="right_hip"),

            e.sensor.Touch(name="pelvis_touch", site="pelvis"),
            e.sensor.Touch(name="upper_waist_touch", site="upper_waist"),
            e.sensor.Touch(name="torso_touch", site="torso"),
            e.sensor.Touch(name="head_touch", site="head"),
            e.sensor.Touch(name="right_upper_arm_touch", site="right_upper_arm"),
            e.sensor.Touch(name="right_lower_arm_touch", site="right_lower_arm"),
            e.sensor.Touch(name="right_hand_touch", site="right_hand"),
            e.sensor.Touch(name="left_upper_arm_touch", site="left_upper_arm"),
            e.sensor.Touch(name="left_lower_arm_touch", site="left_lower_arm"),
            e.sensor.Touch(name="left_hand_touch", site="left_hand"),
            e.sensor.Touch(name="right_thigh_touch", site="right_thigh"),
            e.sensor.Touch(name="right_shin_touch", site="right_shin"),
            e.sensor.Touch(name="right_foot_touch", site="right_foot"),
            e.sensor.Touch(name="left_thigh_touch", site="left_thigh"),
            e.sensor.Touch(name="left_shin_touch", site="left_shin"),
            e.sensor.Touch(name="left_foot_touch", site="left_foot")
        ]

        # 将所有传感器元素添加到<sensor>中
        self.sensor.add_children(sensors)


    def gen_basic_humanoid_xml(self):
        
        # statistic
        self.statistic.extent=2
        self.statistic.center=[0,0,1]
        # option
        self.option.timestep=0.00555
        # default
        self.set_default()
        self.get_humanoid_model()
        self.set_sensors()



    def generate(self):
        '''
        输出xml文档
        '''
        model_xml = self.model.xml()
        save_path = self.save_path + self.robot.graph['name'] + '.xml'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Output
        with open(save_path, 'w') as fh:
            fh.write(model_xml)
        print('Model save to:',save_path)


if __name__ == '__main__':

    R = RobotGraph(name='humanoid_test')

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


    M.gen_basic_humanoid_xml()
    M.get_robot_dfs()
    #M.get_robot(R)
    

    M.generate()

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



