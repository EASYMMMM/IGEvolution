"""Code-grounded architecture diagrams and literature discussion for the report.

Only documentation is generated here; proposed adaptation modules are not installed
in the policy. SVGs are standalone and are also embedded directly in the HTML.
"""
from html import escape


EXTRA_CSS = '''
.architecture-figure{margin:24px 0;border:1px solid #dce7ed;border-radius:12px;background:#fcfdfe;overflow:hidden}
.diagram-scroll{overflow-x:auto;padding:12px}.diagram-scroll svg{display:block;width:100%;min-width:820px;height:auto}
figcaption{padding:14px 20px;background:#edf4f7;color:#435d68;font-size:14px}
.equation-card{padding:18px 22px;margin:16px 0;border:1px solid #dce7ed;border-radius:10px;background:#f8fbfc}
.equation-card h4{margin:0 0 10px;font-size:16px;color:#175963}
.equation{font:18px/1.95 "Cambria Math","STIX Two Math","Noto Sans CJK SC",serif;overflow-x:auto;padding:5px 0}
.equation .line{display:block;white-space:nowrap}.equation-card p{font-size:14px;color:#435d68;margin:9px 0 0}
.decision{border-left:4px solid #b07814;background:#fffaed;padding:16px 20px;margin:22px 0}
.source-list{font-size:14px}.source-list li{margin:12px 0}.architecture-table td:first-child{font-weight:650}
@media(max-width:700px){.diagram-scroll{padding:0}.equation-card{padding:14px}.equation{font-size:16px}}
'''


class Diagram:
    def __init__(self, name, title, height):
        self.name = name
        self.parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 {height}" role="img" aria-labelledby="{name}-title">
<title id="{name}-title">{escape(title)}</title>
<defs><marker id="{name}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/></marker></defs>
<rect width="1100" height="{height}" fill="#fcfdfe"/>
<g font-family="Noto Sans CJK SC,Microsoft YaHei,Arial,sans-serif" fill="#142d3c">''']

    def text(self, x, y, text, size=17, weight=400, color='#142d3c'):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(text)}</text>')

    def box(self, x, y, w, h, title, lines, kind='raw'):
        fill, stroke = {'raw': ('#edf4ff', '#7394c0'), 'encoder': ('#eaf8f5', '#3f9c90'),
                        'training': ('#f0f2f5', '#a0aab7'), 'proposal': ('#fff6de', '#bc8c34')}[kind]
        dash = ' stroke-dasharray="7 5"' if kind == 'training' else ''
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.7"{dash}/>')
        self.text(x + 17, y + 32, title, 20, 700)
        for i, line in enumerate(lines):
            self.text(x + 17, y + 63 + i * 25, line, 16)

    def arrow(self, points, color='#4e739e', dashed=False):
        dash = ' stroke-dasharray="6 5"' if dashed else ''
        self.parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round"{dash} marker-end="url(#{self.name}-arrow)"/>')

    def finish(self):
        return ''.join(self.parts) + '</g></svg>'


def current_diagram():
    d = Diagram('current-control', '当前外肢体控制：原始观测与人体信息编码双通路', 1080)
    d.text(40, 37, '当前实现 · 第五阶段 HMI Encoder checkpoint', 24, 700)
    d.text(40, 65, '蓝色：原有控制主干　绿色：新增编码通路　灰色虚线：训练 / 仿真人体支撑', 16, color='#597080')
    d.box(40, 100, 270, 175, '仿真状态 / 虚拟传感器', [
        '人体–SRL 相对姿态、腿段姿态', '安装界面交互力、SRL 姿态', '关节 q / q̇、上一动作、相位', '当前还不是实机传感器输入'])
    d.box(370, 100, 300, 175, '原始观测支路', [
        '39 维单帧 × 5 帧 + 3 命令', 'o ∈ ℝ¹⁹⁸ → 旧观测归一化', 'HMI 模式：末 3 维命令屏蔽', '人体 / HMI 原始量仍然保留'])
    d.box(760, 100, 300, 175, '预留槽位注入', [
        '15 维 z 写入 5 × 3 个槽位', '替换原置零力矩通道的位置', '训练有 α 渐入；当前评估 α=1', '输入维数不变：198'])
    d.box(40, 340, 270, 160, '编码特征支路', [
        '角度 sin/cos + 人体/HMI', '机器人本体 + 有效标记', '每帧 41 维，保留 64 帧', '独立归一化；只用过去与当前'], 'encoder')
    d.box(370, 340, 300, 160, '因果 TCN → 15 维 z', [
        '41 → 32 通道', 'dilation = 1, 2, 4, 8, 16, 32', '取末时刻 → 32 → 15 → tanh', 'PPO 与运动预测共同训练'], 'encoder')
    d.box(760, 340, 300, 160, 'SRL Actor', [
        '198 → 1024 → 512 → 6', '隐藏层 ELU；高斯策略', '本次评估取均值并裁剪', '输出为归一化关节位置动作'])
    d.box(760, 565, 300, 135, '关节目标 → 底层驱动', [
        '分段映射至 6 关节目标角', '二阶 Butterworth，10 Hz', 'PhysX 位置 / PD 驱动'])
    d.box(40, 565, 630, 135, '人体–安装界面–外肢体耦合物理系统', [
        'SRL 六关节作用于系统；人体分支在仿真中产生人的运动',
        '反馈姿态、关节状态与交互力；控制周期 15 ms（约 66.7 Hz）',
        '实机时由真实人体与机器人替代这里的仿真对象'])
    d.arrow('310,183 370,183')
    d.arrow('670,183 760,183')
    d.arrow('175,275 175,340', '#288b7d')
    d.arrow('310,420 370,420', '#288b7d')
    d.arrow('670,420 710,420 710,235 760,235', '#288b7d')
    d.text(690, 318, 'z', 21, 700, '#288b7d')
    d.arrow('910,275 910,340')
    d.arrow('910,500 910,565')
    d.arrow('760,630 670,630')
    d.arrow('40,630 18,630 18,183 40,183')
    d.text(40, 750, '训练与仿真支撑（下列模块不构成实机 SRL Actor 的额外输入）', 20, 700, '#596679')
    d.box(40, 785, 260, 175, '仿真人体分支', [
        'Humanoid Actor → 58 激活量', '肌肉模型 → 人体关节力矩', 'AMP 模仿约束 / 人体奖励', '实机由真实穿戴者替代'], 'training')
    d.box(335, 785, 225, 175, '特权运动标签', [
        '人体 vₓ、vᵧ、yaw rate', '当前 / +120 / +240 ms', '仅仿真训练监督可见', '不作为 Actor 输入'], 'training')
    d.box(595, 785, 225, 175, '预测头 P(z)', [
        '15 → 64 → 9', '3 时距 × 3 目标', 'Smooth L1 辅助损失', '预测值不直接控制关节'], 'training')
    d.box(855, 785, 205, 175, 'PPO / Critic', [
        '特权状态 → 价值', 'PPO → Actor → E', '辅助损失 → P 与 E', 'Critic 不注入 z'], 'training')
    d.arrow('560,877 595,877', '#8a95a4', True)
    d.text(40, 1007, '关键结构：人体信息可经原始观测直达 Actor，也可经 64 帧 Encoder → z 影响 Actor。', 20, 700, '#175963')
    d.text(40, 1041, '因此 z 是附加控制通路；当前结构不是所有人体信息都必须通过的单一瓶颈。', 18, color='#597080')
    return d.finish()


def adaptation_diagram():
    d = Diagram('adaptation-proposal', '建议域自适应结构：在两条观测支路前增加共享适配', 900)
    d.text(40, 38, '建议结构 · 保留现有 checkpoint 的第一步方案', 24, 700)
    d.text(40, 66, '金色：建议新增，尚未实现　蓝 / 绿：复用现有模块　灰色虚线：适配训练阶段', 16, color='#597080')
    d.box(40, 105, 260, 160, '真实传感器 + 控制记录', [
        '人体姿态、腿段、交互力', '机器人 q / q̇ / 姿态等', '时间戳、前一动作、有效性', '无标签真实序列 Xᴿ'])
    d.box(360, 105, 330, 160, '共享、因果的物理信号适配 G', [
        '单位 / 坐标系 / 时间语义一致', '真实域 → checkpoint 参考域', '只改需适配的观测信号', '控制量与有效标记保持确定语义'], 'proposal')
    d.box(760, 105, 300, 160, '统一的适配后信号', [
        '从同一物理量重建两条输入', '角度修正后再计算 sin / cos', '避免两支路姿态 / 力互相矛盾', '适配不是仅修改 z'], 'proposal')
    d.box(40, 345, 300, 155, '冻结的原始观测主干', [
        '39 × 5 + 3 → 旧归一化', '末 3 维命令屏蔽', '机器人已知量按原规则重建'], 'raw')
    d.box(410, 345, 280, 155, '冻结的 Encoder', [
        '41 × 64 → 旧特征归一化', '因果 TCN → 15 维 z', '预测头保留作诊断 / 约束'], 'encoder')
    d.box(760, 345, 300, 155, '冻结的 Actor 与原驱动', [
        '原始观测 + 槽位注入 z', 'Actor → 位置目标 → 滤波', 'PD → 机器人 → 真实反馈'])
    d.arrow('300,185 360,185', '#b1832d')
    d.arrow('690,185 760,185', '#b1832d')
    d.arrow('910,265 910,300 190,300 190,345', '#b1832d')
    d.arrow('550,300 550,345', '#b1832d')
    d.arrow('340,425 375,425 375,530 730,530 730,460 760,460')
    d.arrow('690,420 760,420', '#288b7d')
    d.text(40, 580, '训练时：利用源域标签 + 无标签真实序列；下方模块不加入部署主链', 20, 700, '#596679')
    d.box(40, 620, 300, 165, '两域数据', [
        '仿真源域 Xˢ 与运动标签 yˢ', '实机目标域 Xᴿ 无昂贵标签', '覆盖不同人、动作、安装状态', '训练 / 测试按人和序列分开'], 'training')
    d.box(410, 620, 280, 165, 'CycleGAN 类训练组件', [
        '可增加反向映射 F：S → R', '域判别器 + 循环一致性', '具体网络 / 权重待验证', '不是本文方法的逐项复现'], 'training')
    d.box(760, 620, 300, 165, '保持控制相关语义', [
        '源域运动标签保持', '源域动作 / 策略一致性', '角度、时序、信号幅值约束', '最终以独立真实闭环检验'], 'training')
    d.arrow('340,703 410,703', '#8a95a4', True)
    d.arrow('690,703 760,703', '#8a95a4', True)
    d.text(40, 837, '最低限度的结构增量：共享传感器适配器；TCN、15 维瓶颈与 198 维 Actor 可先保留。', 20, 700, '#8a5c0b')
    d.text(40, 871, '本图是面向本项目的工程建议；论文采用数据域迁移训练下游估计器，二者不是同一实现。', 17, color='#597080')
    return d.finish()


def figure(svg, filename, caption):
    return f'<figure class="architecture-figure"><div class="diagram-scroll">{svg}</div><figcaption>{caption} · <a href="figures/{filename}.svg">独立 SVG</a>（窄屏可横向滚动）</figcaption></figure>'


def equation(title, lines, note):
    return '<div class="equation-card"><h4>' + title + '</h4><div class="equation">' + ''.join(
        '<span class="line">' + line + '</span>' for line in lines) + '</div><p>' + note + '</p></div>'


def sections(out, conclusion, table):
    current, proposed = current_diagram(), adaptation_diagram()
    (out / 'figures' / '09_control_architecture.svg').write_text(current)
    (out / 'figures' / '10_domain_adaptation_proposal.svg').write_text(proposed)
    current_section = '''<section id="architecture"><h2>当前架构 · 外肢体控制与人体信息编码如何连接</h2>'''
    current_section += conclusion(
        '当前是“原始观测 + 编码器”双通路控制；人体信息并未全部压缩后再进入 Actor。',
        '原始 <mark>198 维</mark>观测仍含人体/HMI 信息；另一路把 <mark>64 × 41</mark>历史编码为 <mark>15 维 z</mark>，写入旧观测的预留槽位。Actor 输出六维关节位置动作。',
        '下图对应当前第五阶段 checkpoint 与代码；灰色模块服务于仿真或训练。未来实机传感器接口和域适配模块尚未实现。', status='实现已核对')
    current_section += figure(current, '09_control_architecture', '图 9 · 当前闭环与训练支撑。蓝色主干包含已有原始人体信息；绿色编码通路与其并行。')
    current_section += '<h3>输入结论：编码器读取的是人体–机器人交互历史，而非纯人体信息</h3>'
    current_section += '<div class="architecture-table">' + table(
        ['部分', '当前数据与维度', '控制含义'], [
            ['原始 Actor 观测', '39 维 × 5 帧 + 3 命令 = 198 维', '含 SRL 角速度、姿态误差、q/q̇/前一动作、相位、相对姿态、交互力、零力矩槽和四个腿段俯仰角'],
            ['人体 / HMI 特征', '6 个角度的 sin/cos（12）+ 交互力（3）+ 腿段姿态（4）= 19', '6 角度来自人–SRL 相对姿态和 SRL 姿态误差；HMI 航向误差参照人体骨盆航向'],
            ['机器人与有效性特征', 'SRL 角速度（3）+ q/q̇/前一动作（18）+ 有效位（1）= 22', '与前 19 维拼成 41 维；部分量已经按环境定义缩放'],
            ['时间历史', 'Actor 5 帧；Encoder 64 帧；控制间隔 15 ms', '最新到最老的时间跨度分别为 60 ms 和 945 ms；Encoder 感受野为 64 帧'],
            ['训练标签与预测', '当前、+8、+16 步，各 3 目标 → 共 9 输出', '人体 vx、vy、航向角速度；来自仿真特权状态，标签不作为策略输入'],
        ]) + '</div>'
    current_section += '<p class="note">Encoder 归一化在训练最初 10 个 rollout 估计并保存，之后固定；原 Actor 的观测归一化沿用旧策略并冻结。这是仿真特征统计估计，不能作为已经完成实机适配的证据。每次 reset，Encoder 的不足历史以零帧和有效位 0 表达。</p>'
    current_section += '<h3>公式结论：z 通过“归一化之后的槽位替换”影响动作</h3>'
    current_section += equation('① 两路输入与因果编码', [
        'o<sub>t</sub> = [r<sub>t</sub>, r<sub>t−1</sub>, …, r<sub>t−4</sub>, c<sub>t</sub>] ∈ ℝ<sup>198</sup>，r<sub>t</sub> ∈ ℝ<sup>39</sup>',
        'x<sub>t</sub> = [sin θ<sub>t</sub>, cos θ<sub>t</sub>, F<sub>t</sub>, ℓ<sub>t</sub>, ω<sub>t</sub>, q<sub>t</sub>, q̇<sub>t</sub>, a<sub>t−1</sub>, m<sub>t</sub>] ∈ ℝ<sup>41</sup>',
        'H<sub>t</sub> = [x<sub>t−63</sub>, …, x<sub>t</sub>]；z<sub>t</sub> = E<sub>φ</sub>(H<sub>t</sub>) ∈ [−1, 1]<sup>15</sup>',
    ], 'θ 为 6 个周期角，F 为三轴交互力，ℓ 为四个腿段俯仰角，m 为有效位。x 中 q/q̇/前一动作等按现有观测尺度取值。E 内含独立归一化、因果 TCN 和 tanh 瓶颈；64 帧按最老到最新排列。')
    current_section += equation('② 保持 Actor 维度不变的融合', [
        'b<sub>t</sub> = M<sub>c</sub>N<sub>old</sub>(o<sub>t</sub>)；u<sub>t</sub> = b<sub>t</sub> + αS(z<sub>t</sub> − S<sup>⊤</sup>b<sub>t</sub>)',
        '𝓘 = {39k + j | k = 0,…,4；j ∈ {32,33,34}}；S = [e<sub>i</sub>]<sub>i∈𝓘</sub> ∈ ℝ<sup>198×15</sup>',
        'μ<sub>t</sub> = π<sub>θ</sub>(u<sub>t</sub>)；a<sub>t</sub> = clip(μ<sub>t</sub>, −1, 1) ∈ ℝ<sup>6</sup>',
    ], 'N<sub>old</sub> 为旧 Actor 归一化，M<sub>c</sub> 将末 3 维任务命令置零；𝓘 使用零基索引。15 个 z 分别填入 15 个槽位，不是把同一个 15 维 z 重复 5 次。α 在训练中渐入，本次评估为 1；上式动作是确定性评估形式，训练使用高斯采样。')
    current_section += equation('③ 动作输出到真实控制量的对应', [
        'q<sup>raw</sup><sub>t,i</sub> = q<sup>0</sup><sub>i</sub> + a<sub>t,i</sub>·(q<sup>max</sup><sub>i</sub> − q<sup>0</sup><sub>i</sub>)，a<sub>t,i</sub> ≥ 0',
        'q<sup>raw</sup><sub>t,i</sub> = q<sup>0</sup><sub>i</sub> + a<sub>t,i</sub>·(q<sup>0</sup><sub>i</sub> − q<sup>min</sup><sub>i</sub>)，a<sub>t,i</sub> &lt; 0',
        'q<sup>ref</sup><sub>t</sub> = clip<sub>joint limits</sub>(BW<sub>2,10Hz</sub>(q<sup>raw</sup><sub>t</sub>))',
        'τ<sub>PD</sub> ≈ K<sub>p</sub>(q<sup>ref</sup> − q) − K<sub>d</sub>q̇　（位置驱动的概念表达）',
    ], 'q<sup>0</sup> 是配置的中立姿态，q<sup>min/max</sup> 是动作映射上下界；当前软限位边距为 0，action_scale=1。BW 为二阶因果 Butterworth 滤波。实际代码把 q<sup>ref</sup> 送入 PhysX 位置驱动，并受驱动器/关节约束；最后一式解释 PD 机制，不代表代码在该分支显式计算并直接下发力矩。')
    current_section += equation('④ 运动监督如何约束编码', [
        'ŷ<sub>t,h</sub> = P<sub>ψ,h</sub>(z<sub>t</sub>)；h ∈ {0, 8, 16}，Δt = 0.015 s',
        'y<sub>t,h</sub> = [R(−ψ<sup>SRL</sup><sub>t</sub>)v<sup>human,world</sup><sub>t+h</sub>；wrap(ψ<sup>human</sup><sub>t+h</sub> − ψ<sup>human</sup><sub>t+h−1</sub>)/Δt]',
        'L<sub>aux</sub> = Σ<sub>h</sub> w<sub>h</sub> · {Σ<sub>t,j</sub> ν<sub>t,h</sub> SmoothL1(ŷ<sub>t,h,j</sub> − y<sub>t,h,j</sub>)} / {3 max(1, Σ<sub>t</sub> ν<sub>t,h</sub>)}',
        'L = L<sub>PPO package</sub> + λ<sub>aux</sub>L<sub>aux</sub>；(w<sub>0</sub>, w<sub>8</sub>, w<sub>16</sub>) = (1, 0.5, 0.25)，λ<sub>aux</sub> = 1',
    ], 'ν 排除越过 rollout 末尾、跨回合或无有效角速度的目标；当前 target_scale=(1,1,1)。速度始终投影到锚点 t 的 SRL 航向系。实际训练还对镜像辅助损失取平均；L<sub>PPO package</sub> 包含策略、价值、熵、动作边界和对称约束。策略梯度经 Actor 回传 Encoder，预测损失经 P 回传 Encoder；预测值不直接作为关节目标。')
    current_section += '''<details><summary>展开：框图与公式对应的代码依据</summary><div class="details-body"><ul class="source-list">
<li><code>isaacgymenvs/tasks/SRLEvo/srl_real_hri_base.py</code>：<code>_compute_observations</code>、<code>_set_hmi_policy_heading</code>、<code>pre_physics_step</code>、<code>_action_to_pd_targets</code>、<code>_apply_srl_pd_target_filter</code>。</li>
<li><code>isaacgymenvs/learning/SRLEvo/hmi_encoder/layout.py</code>：41 维特征、15 个槽位、特权标签；<code>model.py</code>：归一化 / TCN / 瓶颈 / 预测头。</li>
<li>同目录 <code>runtime.py</code>：历史、reset、时距标签与 α；<code>training.py</code>：PPO 与辅助损失联合优化。</li>
<li><code>isaacgymenvs/learning/SRLEvo/srl_network_builder.py</code>：归一化后的命令屏蔽和注入；<code>srl_players.py</code>：人体与 SRL 动作汇合、确定性推理。</li>
</ul><p class="note">结构与超参数按本报告 checkpoint 所在运行目录的 config.yaml 及当前实现核对。代码路径为项目内定位信息；本报告可脱离源码独立阅读。</p></div></details></section>'''

    domain_section = '<section id="domain-adaptation"><h2>6 · 域自适应：能否匹配论文思路，需要改哪些结构？</h2>'
    domain_section += conclusion(
        '可以作为域自适应的起点；现有网络骨干可先保留，但当前系统还没有实现域自适应。',
        '已有时序编码与仿真运动标签，可提供任务约束；缺少真实域数据、跨域映射与适配目标。由于存在原始人体信息旁路，<mark>仅对齐 15 维 z 不足以覆盖 Actor 的全部域偏移</mark>。',
        '下面是依据论文思路和本项目代码作出的设计判断，不是已经完成的实机结果。是否保留当前骨干最终取决于真实数据可观测性、时延与闭环验证。', status='兼容，但须新增适配环节', tone='qualified')
    domain_section += '''<h3>文献结论：省去昂贵的设备专用标注，不等于无需真实数据或物理接口定义</h3>
<p><em>Deep domain adaptation eliminates costly data required for task-agnostic wearable robotic control</em> 是 Scherpereel 等发表于 <strong>Science Robotics（2025）</strong>的研究，DOI：<a href="https://doi.org/10.1126/scirobotics.ads8652">10.1126/scirobotics.ads8652</a>。它用生物力学模型生成的模拟传感器域作为桥梁，结合公开数据与无标签外骨骼数据，降低训练髋/膝生物关节力矩估计器所需的设备专用标签成本。<a href="https://pubmed.ncbi.nlm.nih.gov/41259448/">论文摘要</a></p>
<p>研究机构对方法的说明明确提到 <strong>CycleGAN</strong>：把易获得的数据映射到设备相关的数据域，再服务于助力估计。这里不能进一步推断其与本项目的 15 维瓶颈、PPO 或位置控制器相同。<a href="https://coe.gatech.edu/news/2025/11/real-world-helper-exoskeletons-just-got-closer-reality">Georgia Tech 方法说明</a></p>
<p class="note">你提供的 Nature Machine Intelligence 文章 <a href="https://www.nature.com/articles/s42256-026-01297-7#ref-CR34">A roadmap for end-to-end task-agnostic exoskeleton control 的参考文献 34</a>引用了这篇工作，因而可能是你看到该思路的入口；这不能确认导师的具体出处。此处依据可访问的摘要、参考文献与研究机构说明讨论，不声称复现了付费正文中未核实的网络细节。</p>
<p class="finding"><strong>对“不需要标定”的准确理解：</strong>可以尝试避免逐人、逐设备收集大量带真实运动/力矩标签的监督标定集；仍须有训练用目标域无标签序列，并保证单位、坐标系、时间同步、安装方向和控制接口的语义可对应。适配能否替代某一项具体标定，应单独验证。传感器零偏与安装误差可以成为适配目标，但论文标题不能证明任何硬件条件下都可以省略这些工作。</p>
<h3>兼容性结论：已有时序骨干和监督锚点，缺口在两域数据与完整观测链</h3>'''
    domain_section += '<div class="architecture-table">' + table(['要求 / 前提', '当前状态', '判断与所需工作'], [
        ['有标签源域 + 无标签目标域', '有仿真状态与运动标签；本次实验没有真实域数据', '具备源域条件；还需采集实机传感器与控制历史，并留出独立人物/序列测试'],
        ['两域信号语义可对应', '仿真可获得相对姿态、腿段俯仰、力和本体状态', '必须确认实机能构造相同信息；缺失传感器不是简单分布对齐可保证解决的'],
        ['时间上下文可处理', '因果 TCN、64 帧、15 ms，已含 reset 有效位', '骨干可复用；需对齐采样、延迟和丢帧，并约束新增适配器为因果'],
        ['域迁移与任务语义保持', '现有训练仅 PPO + 仿真运动预测', '尚不具备；需要适配模块、跨域目标与运动/动作语义约束'],
        ['覆盖所有受域偏移影响的 Actor 输入', 'z 与原始人体/HMI 信息并行进入 Actor', '只改 Encoder 会遗漏原始支路；两路必须共享一致的信号适配'],
        ['下游控制任务对应', '论文估计生物关节力矩；本项目输出 SRL 关节位置动作', '可借鉴迁移原则，不能直接等同方法；无需为了名字一致强行增加人体力矩头'],
        ['真实闭环效果', '现有结果只支持仿真中编码被使用', '未验证；观测域适配也不能单独消除驱动、接触、人体响应的动力学差异'],
    ]) + '</div>'
    domain_section += '<h3>结构建议结论：优先在分流之前加共享适配器，先冻结现有控制网络</h3>'
    domain_section += figure(proposed, '10_domain_adaptation_proposal', '图 10 · 本项目建议的最小结构增量。部署用真实域 → 仿真参考域的前端，是借鉴域迁移原则的工程方案，并非论文实现原图。')
    domain_section += '''<p>建议先把需要适配的<strong>物理观测信号</strong>通过共享因果映射 G，再从同一结果构造原始 Actor 观测和 Encoder 历史。先冻结 E、P、Actor 与原归一化，检验能否恢复它们熟悉的输入语义。不要让两个独立生成器分别任意修改 39 维和 41 维输入：同一姿态和交互力会出现不一致。角度应在一致的姿态表示中修正后再产生 sin/cos；上一动作、相位生成规则、命令和有效标记应按控制系统规则重建。机器人本体信号也要检查域偏移，不能先验认定完全一致。</p>
<p class="note">论文的公开说明侧重将可获得的数据迁移到设备域、训练下游估计器；本建议选择“真实输入 → 冻结策略参考域”以尽量复用当前 checkpoint。若这种方式不够，也可用适配后的数据重新训练下游网络；这属于第二步实验选择，并非当前必须更换 TCN 或增加隐变量维数。</p>'''
    domain_section += equation('建议目标：对齐数据，同时约束运动与动作语义（不是论文原损失的逐项复现）', [
        'X̃<sup>R</sup> = G<sub>R→S</sub>(X<sup>R</sup>)；a<sup>R</sup> = Π<sub>frozen</sub>(X̃<sup>R</sup>)',
        'L<sub>adapt</sub> = L<sub>adv</sub> + λ<sub>cyc</sub>L<sub>cyc</sub> + λ<sub>task</sub>L<sub>task</sub> + λ<sub>act</sub>L<sub>act</sub> + λ<sub>phys</sub>L<sub>phys</sub>',
        'L<sub>cyc</sub> = 𝔼<sub>R</sub>‖F(G(X<sup>R</sup>)) − X<sup>R</sup>‖<sub>1</sub> + 𝔼<sub>S</sub>‖G(F(X<sup>S</sup>)) − X<sup>S</sup>‖<sub>1</sub>',
        'L<sub>task</sub> = 𝔼<sub>S</sub>ℓ(Q<sub>frozen</sub>(G(F(X<sup>S</sup>))), y<sup>S</sup>)',
        'L<sub>act</sub> = 𝔼<sub>S</sub>‖Π<sub>frozen</sub>(G(F(X<sup>S</sup>))) − Π<sub>frozen</sub>(X<sup>S</sup>)‖<sup>2</sup>',
    ], 'X 为物理信号序列；Π 包括两路特征构造、归一化、编码注入与 Actor，Q 为特征构造后的冻结 E+P。F 是可选的反向 S→R 映射，配合判别器用于 CycleGAN 类训练。L<sub>adv</sub> 对齐两域分布；L<sub>phys</sub> 约束物理表示、幅值和时序。任务与动作项只在有标签源域的往返数据上计算，不假设有配对实机标签；这些约束仍不能证明真实域语义一定正确。权重需用独立验证选择。')
    domain_section += '''<p class="finding"><strong>为什么不能只让两域 z 的分布相似？</strong>边缘分布相似并不保证对应运动具有相同语义。域不变表征在条件分布变化时可能无法保证目标域效果，这在理论上已有反例。<a href="https://proceedings.mlr.press/v97/zhao19a.html">Zhao 等，ICML 2019</a> 对本项目而言，预测头只约束可读出的运动目标，未固定全部 15 维中的控制相关方向；因此还要查动作变化与闭环行为。循环一致性和源域往返约束也是训练约束，不是目标域正确性的证明。</p>
<h3>修改范围结论：需要增添适配模块与训练约束，暂不必重做主干</h3>'''
    domain_section += '<div class="architecture-table">' + table(['模型部分', '本轮建议', '是否必需 / 原因'], [
        ['TCN、64 帧历史、15 维 z', '先保留，冻结作参考', '没有证据表明必须更换；真实信息不足或时延不匹配时再比较结构'],
        ['198→1024→512→6 Actor', '先保留原权重与输入接口', '当前检查的目标是迁移现有策略；直接改维数会失去可比性和 checkpoint 兼容'],
        ['共享因果传感器适配器 G', '新增于两条支路之前', '本建议路径的核心结构改动；覆盖原始人体信息旁路'],
        ['反向 F 与域判别器', '若采用 CycleGAN 类训练则新增', '训练时使用；其他适配方法可不使用同样组件'],
        ['运动预测头与动作约束', '保留 P，新增跨域训练目标', '利用已有源域标签和冻结策略约束语义；不需实机昂贵标签参与该训练项'],
        ['人体信息统一经过瓶颈', '可作后续结构对照：π(robot state, E(human history, robot context))', '不是域适配的硬性要求；若移除原始人体旁路，必须重训/蒸馏并重新验证'],
        ['槽位注入改为显式拼接', '仅在后续重构时考虑', '更易解释，但不是适配必需；需要明确权重迁移或重新训练'],
        ['动力学 / 执行器差异', '另做驱动、接触、时延及人体响应评估', '属于控制闭环迁移问题，不应全部交给观测生成器吸收'],
    ]) + '</div>'
    domain_section += '''<h3>验证结论：用真实闭环收益判断是否适配成功，不能只看域分类或潜空间重叠</h3>
<p>后续至少比较<strong>无适配、仅 Encoder 适配、共享观测适配</strong>三组，按未见人物与运动序列评估。离线检查输入物理一致性、运动预测和动作幅值/连续性；闭环沿用本报告的位置跟踪、交互力、终止率与功率指标，并检查控制时延。若有独立的少量真实运动标签，可只用于验收，明确与无监督适配训练分开。当前仿真中固定均值编码仍可行走，因此“能走”不足以证明人体信息已正确迁移。</p>
<div class="decision"><strong>架构决策：</strong>保留现有 TCN 与 Actor，先补“共享因果观测适配 + 无标签真实域数据 + 运动/动作语义约束”。只有在这条路径不能覆盖实机信息缺失、传感延迟或控制依赖时，再考虑拆分人体/本体输入并重训。<strong>这能匹配论文的核心迁移思想，但尚不能称为已经实现或验证了该论文方法。</strong></div>
</section>'''
    return current_section, domain_section
