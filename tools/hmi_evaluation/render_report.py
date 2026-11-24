"""Build a self-contained Chinese HTML report from measured experiment results."""
import argparse
import base64
import html
import json
from pathlib import Path
from report_architecture import EXTRA_CSS, sections as architecture_sections


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output.resolve()
    d=json.loads((out/'results.json').read_text())
    labels={'intact':'完整编码','zero':'隐变量置零','mean':'固定均值编码','shuffle':'跨任务替换编码',
            'delay16':'编码延迟 240 ms','human_swap':'仅替换编码器人体/HMI输入','short5':'仅保留最近 5 帧',
            'intact_repeat':'完整编码重复对照','pose_swap':'替换姿态信息','force_swap':'替换交互力',
            'legs_swap':'替换人体腿段姿态','robot_swap':'替换机器人本体信息','reverse':'反转历史顺序'}
    closed={r['condition']:r for r in d['closed_loop']}
    actions={r['intervention']:r for r in d['actions']}
    predictions=[r for r in d['prediction'] if r['model']=='Frozen prediction head']
    def image(name):
        b64=base64.b64encode((out/'figures'/f'{name}.png').read_bytes()).decode()
        return f'<img alt="{name}" src="data:image/png;base64,{b64}">'
    def table(headers,rows):
        return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{html.escape(str(h))}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(c))}</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    def conclusion(title, evidence, boundary, status='证据支持', tone='supported'):
        return (f'<aside class="conclusion {tone}" aria-label="本节结论">'
                f'<div class="conclusion-label">本节结论 <span class="badge">{html.escape(status)}</span></div>'
                f'<p class="conclusion-title">{html.escape(title)}</p>'
                f'<p class="evidence"><b>关键证据</b>　{evidence}</p>'
                f'<p class="boundary"><b>解释边界</b>　{boundary}</p></aside>')
    def fmtci(row,key,scale=1,decimals=3):
        return f"{row[key]*scale:.{decimals}f} [{row[key+'_delta_ci_low']*scale:+.{decimals}f}, {row[key+'_delta_ci_high']*scale:+.{decimals}f}]"
    architecture, domain_adaptation = architecture_sections(out, conclusion, table)
    force_delta=(closed['zero']['force_mean']/closed['intact']['force_mean']-1)*100
    short_delta=(closed['short5']['position_rmse']/closed['intact']['position_rmse']-1)*100
    mean_force_delta=(closed['mean']['force_mean']/closed['intact']['force_mean']-1)*100
    current_probe={(r['probe'],r['target']):r['r2'] for r in d['probes'] if r['horizon_steps']==0}
    probe_names=list(dict.fromkeys(r['probe'] for r in d['probes']))
    probe_table=table(['线性探针输入','当前 vx R²','当前 vy R²','当前航向角速度 R²'],
                      [[name]+[f"{r['r2']:.4f}" for r in d['probes'] if r['probe']==name and r['horizon_steps']==0]
                       for name in probe_names])
    history_extra=''
    if 'history_validity_controls' in d:
        hc={r['condition']:r for r in d['history_validity_controls']}
        history_extra=f'''<h3>复核结论：历史顺序本身影响编码和动作</h3><p class="finding">保持最近 5 帧和全部有效标记不变，只打乱更早 59 帧，当前 vx 的 RMSE 从 <mark>{hc['intact']['rmse_h0_0']:.3f} 升至 {hc['shuffle_old59']['rmse_h0_0']:.3f} m/s</mark>，动作 RMS 改变 <strong>{hc['shuffle_old59']['action_rms']:.4f}</strong>。长历史效应不能完全归因于补零或有效帧数量。</p>{image('08_history_controls')}<p class="note">复核使用独立抽取的 4,000 个时刻。打乱、保持旧帧、重复最近 5 帧会产生不自然的历史；结论限定为当前网络的时间依赖。</p>'''
    content=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>HMI 人体信息编码器系统评估 · 2026-11-10</title><style>

:root{{--ink:#142d3c;--muted:#597080;--accent:#087f8c;--line:#dce7ed;--bg:#f3f7fa}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.75 system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}}
main{{max-width:1180px;margin:auto;padding:42px 28px 70px}}header{{background:#142d3c;color:white;border-radius:20px;padding:38px;margin-bottom:24px}}header .eyebrow{{color:#83d6d8;font-size:13px;letter-spacing:.1em}}h1{{font-size:32px;margin:8px 0 12px;line-height:1.35}}h2{{font-size:25px;margin:0 0 20px;line-height:1.5}}h3{{font-size:19px;margin:28px 0 12px}}p{{margin:12px 0}}
section{{padding:30px;background:white;border:1px solid var(--line);border-radius:16px;margin:24px 0;scroll-margin-top:24px}}.lead{{font-size:18px}}.note{{color:var(--muted);font-size:14px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:15px}}.card{{background:white;border:1px solid var(--line);border-radius:14px;padding:20px}}.card b{{display:block;font-size:26px;color:var(--accent)}}.card span{{font-size:13px;color:var(--muted)}}
nav{{display:flex;flex-wrap:wrap;gap:10px;margin:22px 0}}nav a{{padding:7px 13px;border:1px solid var(--line);border-radius:8px;background:white;text-decoration:none;font-size:14px}}nav a:hover{{border-color:var(--accent)}}
.conclusion{{border:1px solid #bee0df;border-left:5px solid var(--accent);background:#eef8f7;border-radius:10px;padding:21px 24px;margin:0 0 24px}}.conclusion-label{{color:#12696e;font-size:13px;font-weight:700;letter-spacing:.08em}}.badge{{display:inline-block;margin-left:10px;padding:2px 9px;border-radius:5px;background:#d3ece9;letter-spacing:0;font-size:12px}}.conclusion-title{{font-size:22px;font-weight:750;line-height:1.6;margin:10px 0 14px}}.evidence{{font-size:16px}}.boundary{{border-top:1px solid #cfe3e2;padding-top:12px;margin-bottom:0;color:#435d68;font-size:14px}}.conclusion.qualified{{background:#fffaed;border-color:#eddbb3;border-left-color:#b07814}}.qualified .conclusion-label{{color:#8a5c0b}}.qualified .badge{{background:#f5e8c7}}.qualified .boundary{{border-top-color:#ebdec0}}
mark{{background:#d5efec;color:#124f57;border-radius:3px;padding:1px 4px;font-weight:700}}.qualified mark{{background:#f5e8c7;color:#6d4a0e}}.finding{{padding-left:16px;border-left:3px solid #cbdde5}}.reading-guide{{margin:0 0 16px;color:var(--muted);font-size:14px}}
img{{display:block;width:100%;height:auto;margin:20px 0}}table{{width:100%;border-collapse:collapse;font-size:14px}}th{{background:#edf4f7;color:#24495e;text-align:left;white-space:nowrap}}th,td{{padding:12px 14px;border-bottom:1px solid var(--line);vertical-align:top}}td{{min-width:90px}}tbody tr:nth-child(even){{background:#fafcfd}}.scroll{{overflow:auto}}.overview-table td:nth-child(2){{font-weight:650;color:#175963}}.key-table tbody tr:first-child{{background:#eaf6f5;font-weight:700}}
details{{margin:20px 0;border:1px solid var(--line);border-radius:9px;background:#fbfdfe}}summary{{padding:13px 17px;cursor:pointer;font-weight:650;color:#375669}}summary:hover{{color:var(--accent)}}.details-body{{padding:2px 18px 18px}}details[open] summary{{border-bottom:1px solid var(--line);margin-bottom:14px}}
code{{background:#edf3f6;border-radius:4px;padding:2px 5px;font-size:13px;overflow-wrap:anywhere}}pre{{overflow:auto;background:#edf3f6;padding:18px;border-radius:10px;font-size:13px}}a{{color:#087f8c}}ul,ol{{padding-left:24px}}li{{margin:8px 0}}footer{{color:var(--muted);font-size:13px}}
@media(max-width:700px){{main{{padding:15px}}header,section{{padding:20px}}.cards{{grid-template-columns:1fr 1fr}}h1{{font-size:26px}}h2{{font-size:22px}}.conclusion{{padding:16px}}.conclusion-title{{font-size:20px}}th,td{{padding:10px}}}}
@media print{{body{{background:white}}main{{padding:0}}nav{{display:none}}.conclusion{{break-inside:avoid;print-color-adjust:exact}}h2,h3{{break-after:avoid}}}}
{EXTRA_CSS}
</style></head><body><main><header><div class="eyebrow">控制架构 · 编码评估 · 域自适应分析</div><h1>人体信息编码器：控制架构、系统评估与迁移分析</h1><p class="lead">固定第五阶段第 3541 轮最佳模型，先说明现有控制与编码通路，再评估隐变量的信息和作用，最后讨论面向实机的域自适应方案。</p><p>实验与架构核对日期：2026-11-10 · 平台：Isaac Gym / RTX 5090 · 控制周期：15 ms</p></header>
<div class="cards"><div class="card"><b>1,536</b><span>闭环评估回合：8 条件 × 3 种子 × 64</span></div><div class="card"><b>75 秒</b><span>每回合上限；只统计首次回合</span></div><div class="card"><b>{sum(int(d['sample_counts'][str(s)]) for s in (202,303)):,}</b><span>未见评估轨迹上的解码样本</span></div><div class="card"><b>12,000</b><span>固定观测的隐变量干预样本</span></div></div>
<nav aria-label="报告目录"><a href="#overview">结论总览</a><a href="#architecture">当前架构与公式</a><a href="#information">1 信息内容</a><a href="#actions">2 动作作用</a><a href="#history">3 时间依赖</a><a href="#closed-loop">4 闭环效果</a><a href="#geometry">5 隐空间形态</a><a href="#reproduce">复现与范围</a><a href="#domain-adaptation">6 域自适应</a></nav>
<section id="overview"><h2>结论总览</h2>
{conclusion('隐变量已被控制器使用，现有证据支持其承担辅助调节作用。',
    '人体运动可准确读出；仅改变编码通路即可改变动作；闭环交互力和跟踪误差也随干预变化。',
    '尚不能认定它是维持基本行走的必要通路，也不能认定完整编码在所有控制指标上更优。',status='有明确作用，收益有边界',tone='qualified')}
<p class="reading-guide">每节依次给出：本节结论、关键证据、解释边界。绿色表示证据支持的判断；金色表示需要保留条件的判断。完整方法与明细表可展开查看。</p>
<div class="overview-table">{table(['评估问题','本次结论','最直接的证据'],[
['1 · 编码中有什么？','人体运动信息可读出，且具有短时预测信息',f"当前 vx / vy / 航向角速度 R²：{predictions[0]['r2']:.3f} / {predictions[1]['r2']:.3f} / {predictions[2]['r2']:.3f}"],
['2 · 控制器是否使用它？','使用；人体信息经编码通路影响动作',f"固定原始观测，z 换成均值后动作 RMS 改变 {actions['mean']['mean_action_rms']:.4f}"],
['3 · 较早历史是否有作用？','有；历史顺序本身也影响编码与动作','最近 5 帧不变，打乱更早历史仍改变输出'],
['4 · 闭环控制是否受影响？','受影响，但没有全面占优或必要性的证据',f"置零后交互力 {force_delta:+.1f}%；仅保留 5 帧后位置 RMSE {short_delta:+.1f}%"],
['5 · 隐空间是否有结构？','有变化且方差集中，不能据此认定语义解耦',f"15 维协方差参与率 {d['geometry']['participation_ratio']:.2f}"]])}</div>
</section>
{architecture}
<section id="information"><h2>1 · 信息内容：人体运动状态可以准确读出</h2>
{conclusion('隐变量包含当前人体运动信息，也包含未来 240 ms 的可预测信息。',
    f"当前 vx、vy、航向角速度的 R² 为 <mark>{predictions[0]['r2']:.3f} / {predictions[1]['r2']:.3f} / {predictions[2]['r2']:.3f}</mark>；未来 240 ms 为 <strong>{predictions[6]['r2']:.3f} / {predictions[7]['r2']:.3f} / {predictions[8]['r2']:.3f}</strong>，均优于保持当前估计的参考。",
    '这证明运动状态可读出，不等于识别了主观意图，也不直接证明控制性能收益。')}
<h3>关键证据：独立轨迹上的解码与预测</h3><p>预测目标为当前、未来 120 ms 和 240 ms 的人体 vx、vy、航向角速度。vx/vy 均在<strong>当前锚点的外肢体航向坐标系</strong>中表达。未来标签跨回合、历史不足 64 帧的样本被排除。</p>
{image('01_information')}
<details><summary>展开：各时距的完整预测指标</summary><div class="details-body">{table(['时距','目标','RMSE','MAE','R²'],[[str(r['horizon_steps']*15)+' ms',r['target'],f"{r['rmse']:.4f}",f"{r['mae']:.4f}",f"{r['r2']:.4f}"] for r in predictions])}</div></details>
<p class="note">R² 是样本方差解释比例，负值表示比预测测试集均值还差；它不是控制成功率。虚线 persistence 使用当前预测直接充当未来预测。另保存“真实当前状态保持不变”的特权参考，帮助区分运动平滑性和真正的预测能力。</p>
<h3>补充结论：速度信息更易线性读出，但并非所有目标都优于短历史</h3><p class="finding">隐变量的当前 vx 线性读出 R² 为 <strong>{current_probe[("z (15D)","vx (m/s)")]:.3f}</strong>，最近 5 帧原始特征为 <strong>{current_probe[("Recent 5 raw frames (200D)","vx (m/s)")]:.3f}</strong>；但当前航向角速度的短历史读出已达到 <strong>{current_probe[("Recent 5 raw frames (200D)","yaw rate (rad/s)")]:.3f}</strong>。编码收益具有目标差异。</p><p>线性探针只用种子 101 的轨迹训练，环境 0–47 用于候选拟合、48–63 用于正则化选择，随后在 101 全部轨迹上重拟合；种子 202、303 完全用于测试。原始输入对照为编码器当前帧的 40 个非有效标记特征。</p>
{probe_table}<p class="note">另加入最近 5 帧的 200 维原始特征及其与 z 的拼接，避免只与单帧比较。线性探针差异反映信息的线性可读出程度，不证明原有非线性 Actor 无法从短历史提取相同信息。</p>
{image('02_prediction_scatter')}</section>
<section id="actions"><h2>2 · 动作作用：控制器实际读取了隐变量</h2>
{conclusion('在原始观测不变时，仅改变编码通路就会改变控制动作。',
    f"z 固定为均值，动作 RMS 改变 <mark>{actions['mean']['mean_action_rms']:.4f}</mark>，95% 区间为 {actions['mean']['action_rms_ci_low']:.4f}–{actions['mean']['action_rms_ci_high']:.4f}；仅替换编码器人体/HMI 输入，改变 <mark>{actions['human_swap']['mean_action_rms']:.4f}</mark>。",
    '这建立了当前网络中编码通路对动作的因果作用；动作发生变化，本身不表示变化方向更好。')}
<h3>关键证据：固定原始观测的编码干预</h3><p>对相同原始观测重新计算动作，只替换编码通路。统计的是送入后续动作滤波器前、裁剪到 [-1,1] 的 6 维控制输出。每个样本的变化量为六维动作差的 RMS；不是关节角误差或力矩误差。</p>
{image('03_action_causality')}
<details><summary>展开：11 种动作干预的完整指标与区间</summary><div class="details-body">{table(['干预','动作 RMS 变化','95% 回合重采样区间','策略 KL（裁剪前）'],[[labels[r['intervention']],f"{r['mean_action_rms']:.5f}",f"[{r['action_rms_ci_low']:.5f}, {r['action_rms_ci_high']:.5f}]",f"{r['mean_policy_kl']:.5f}"] for r in d['actions']])}</div></details>
<h3>补充结论：当前通路对人体腿段姿态更敏感</h3><p class="finding">替换腿段姿态的动作 RMS 变化为 <mark>{actions['legs_swap']['mean_action_rms']:.4f}</mark>，替换交互力为 <strong>{actions['force_swap']['mean_action_rms']:.4f}</strong>。这反映本次干预下的依赖强度，不是跨模态信息价值的统一排名。</p>
<p class="note">每个测试种子随机抽取 6,000 个合法时刻。区间以环境回合为块重采样，避免将高度相关的帧当成独立实验。姿态/HMI 替换只修改编码器输入，原始 Actor 中的人体信息仍保留。</p></section>
<section id="history"><h2>3 · 时间依赖：较早历史及其顺序参与控制</h2>
{conclusion('编码器利用了最近 5 帧之外的历史，历史顺序也会影响控制输出。',
    f"局部历史→动作敏感度中，最近 5 帧占 <mark>{d['temporal']['newest_5_fraction']*100:.1f}%</mark>，更早帧合计占 <mark>{(1-d['temporal']['newest_5_fraction'])*100:.1f}%</mark>；保留最新帧、仅打乱更早帧的复核也改变了预测与动作。",
    '敏感度不是性能贡献比例；人工历史干预会改变输入分布，不能据此认定 64 帧是最优或必要窗口。')}
<h3>关键证据：截断历史与局部敏感度</h3>{image('07_temporal_information')}
<p>局部的“历史 → 编码器 → 动作”敏感度中，最近 5 帧占 <strong>{d['temporal']['newest_5_fraction']*100:.1f}%</strong>，最近 16 帧占 <strong>{d['temporal']['newest_16_fraction']*100:.1f}%</strong>，最近 32 帧占 <strong>{d['temporal']['newest_32_fraction']*100:.1f}%</strong>。这里按各输入特征的经验标准差加权。</p>
<p class="note">截断历史改变输入分布，梯度只描述采样点附近的局部依赖。两者互为补充，不能单独证明更长窗口在所有任务上必要。</p>
{history_extra}</section>
<section id="closed-loop"><h2>4 · 闭环效果：存在调节作用，尚无全面占优证据</h2>
{conclusion('编码改变交互力与跟踪行为，但完整动态编码并非所有指标上的最佳条件。',
    f"置零后交互力 <mark>{force_delta:+.1f}%</mark>；固定均值后 <strong>{mean_force_delta:+.1f}%</strong>；仅保留 5 帧后位置 RMSE <mark>{short_delta:+.1f}%</mark>。固定均值仍能完成本次全部回合。",
    '置零和延迟条件的跟踪误差略低；固定均值的跟踪差值区间包含零。不能据此认定动态编码全面改善性能，或是维持行走的必要输入。',status='作用明确，收益取决于指标',tone='qualified')}
<h3>关键证据：192 个配对回合／条件</h3>
<p class="finding"><strong>判读要点：</strong>置零同时移除了编码的平均偏置与动态变化；固定均值保留典型偏置，只移除时变信息。两种对照结合，才能区分编码偏置与动态信息的作用。交互力变小本身不等于助力更好。</p>
<p>每个条件 192 个首次回合（3 个种子 × 64 环境）。冻结 humanoid、SRL、编码器权重，使用确定性均值动作和原动作滤波器。各条件随机分配到相互隔离的物理环境编号，所有初态一次性提交给 PhysX；初始状态与完整目标轨迹已通过哈希核对，重复对照的首步动作一致性也通过检查。</p>
<details><summary>展开：8 个闭环条件的具体操作</summary><div class="details-body">{table(['条件','具体操作'],[
['完整编码','使用当前完整 64 帧历史计算 z，α=1'],
['隐变量置零','15 维 z 全部置零；其余 Actor 观测不变'],
['固定均值编码','使用独立种子 17 的校准轨迹均值，去掉时变信息'],
['跨任务替换编码','替换为同一时刻、另一条完整编码参考轨迹的 z；固定循环错配'],
['延迟 240 ms','使用本回合 16 步前的 z；起始不足 16 步时置零'],
['替换人体/HMI输入','将编码器历史前 19 个姿态、交互力、人体腿段通道换成另一条参考轨迹；本体历史和原始 Actor 观测保留'],
['最近 5 帧','编码器较早 59 帧置为无效零帧；保持 64 帧张量尺寸'],
['完整编码重复对照','相同策略和轨迹，放在另一组物理环境，检查数值/采样变异']])}</div></details>
{image('05_closed_loop')}
<p class="note">提前终止率使用 95% Wilson 区间；其余图示区间使用回合 bootstrap。每组只有 0–1 次提前终止：0/192 的 Wilson 上限仍约为 1.96%，不能把未观察到失败解释为零风险，也不能据此认定各条件风险等价。</p>
<div class="key-table">{table(['条件','位置 RMSE/m','交互力/N','提前终止回合'],[[labels[r['condition']],f"{r['position_rmse']:.4f}",f"{r['force_mean']:.2f}",f"{round(r['failed']*r['n_episodes'])}/{r['n_episodes']}"] for r in d['closed_loop']])}</div>
<details><summary>展开：时长、速度、功率等完整闭环指标</summary><div class="details-body">{table(['条件','提前终止率','平均时长/s','位置 RMSE/m','速度 RMSE/(m/s)','交互力/N','SRL机械绝对功率/W'],[[labels[r['condition']],f"{r['failed']*100:.2f}%",f"{r['duration']:.2f}",f"{r['position_rmse']:.4f}",f"{r['velocity_rmse']:.4f}",f"{r['force_mean']:.2f}",f"{r['power_mean']:.2f}"] for r in d['closed_loop']])}</div></details>
<p class="note">位置 RMSE 衡量人体骨盆水平位置相对目标轨迹的误差；速度 RMSE 使用任务中由未来 0.5 秒轨迹得到的目标水平速度。交互力取仿真安装界面虚拟传感器的滤波后三维合力模长。功率仅统计 SRL 六个关节。</p>
<h3>统计结论：历史截断的跟踪变化明确，失败风险尚难区分</h3>
<p class="finding">仅保留 5 帧的位置 RMSE 配对差值 95% 区间为 <mark>+{closed['short5']['position_rmse_delta_ci_low']*100:.2f}～+{closed['short5']['position_rmse_delta_ci_high']*100:.2f} cm</mark>；固定均值的区间跨零。各组仅 0–1 次提前终止，不足以判断失败风险相同或不同。</p>
<details><summary>展开：配对差值的完整置信区间</summary><div class="details-body"><p class="note">下表方括号为“该条件 − 完整编码”的 95% 配对回合 bootstrap 区间（4,000 次）；方括号之前为该条件均值。仅 3 个轨迹生成种子；这些区间是探索性估计，未做多重比较校正。</p>
{table(['条件','位置 RMSE：均值 [差值区间]/m','交互力：均值 [差值区间]/N','提前终止：均值 [差值区间]/百分点'],[[labels[r['condition']],fmtci(r,'position_rmse',decimals=4),fmtci(r,'force_mean',decimals=2),fmtci(r,'failed',scale=100,decimals=2)] for r in d['closed_loop'] if r['condition']!='intact'])}</div></details>
{image('06_survival')}
<p class="note">“提前终止”包括摔倒/接触/高度终止及偏离轨迹超过 3 m，不等同于纯摔倒率。位置、速度、力、功率按每个回合终止前的有效步计算，再对回合平均；因此存在存活条件化，不能脱离终止率解释。绝对机械功率为 Σ|τ·q̇|，不等于电功耗。完整编码重复组用于估计不同 GPU 物理实例的数值变异。</p></section>
<section id="geometry"><h2>5 · 隐空间形态：表征有变化，但不能认定语义解耦</h2>
{conclusion('15 个隐变量维度均有变化，方差集中在较少方向；这不足以赋予各维明确语义。',
    f"协方差参与率为 <mark>{d['geometry']['participation_ratio']:.2f} / 15</mark>；各维标准差均非零，在本次样本中未见 |z| &gt; 0.95 的输出。",
    '参与率衡量方差集中程度，不等于只需 4 个变量。相关性和 PCA 不能证明“意图维度”或语义解耦；该表征同时混合人体/HMI 与机器人历史。',status='描述性证据',tone='qualified')}
<h3>关键证据：方差、相关性与主成分分布</h3>{image('04_latent_geometry')}
<p>15 维隐变量的协方差参与率为 <strong>{d['geometry']['participation_ratio']:.2f}</strong>。相关图和 PCA 仅描述信息分布，不能据此给单个维度命名为“人体意图”或证明语义解耦。</p>
<p>该编码器同时使用人体/HMI与机器人本体历史，因此是混合交互表征。人体运动可读出，仍可能部分来自物理耦合下的机器人自身信号。固定原始观测的编码器人体通道替换，专门检查了人体信息经这条额外通路对动作的影响。</p></section>
<section id="reproduce"><h2>复现与适用范围</h2>
{conclusion('本报告评估的是当前已训练模型如何使用编码，结论限定于本次仿真任务。',
    '模型权重、归一化参数与推理方式固定；正式数据来自 3 个种子、8 个条件和 1,536 个首次回合，初态与目标轨迹配对核对通过。',
    '未比较重新训练后的编码结构，也未验证噪声、体型变化、随机策略及实机泛化。先导实验数据已从正式统计中排除。',status='范围已界定',tone='qualified')}
<h3>复现依据：模型、数据与命令</h3><p>模型：<code>{html.escape(d['checkpoint'])}</code></p><p>SHA-256：<code>{d['checkpoint_sha256']}</code></p>
<p>校准种子 17：64 个环境、1,500 步，仅用于固定均值编码。正式种子：101、202、303；每组 64 个环境、5,000 控制步，原环境 4,999 步触发时限终止。所有原策略与归一化参数保持冻结。</p>
<p class="note">质量审查：先导实验发现分组连续 reset 后物理初态没有统一提交。已在评估脚本中修正，并重新采集主实验。本报告所有分析均使用修正后的 validated 目录；先导目录仅保留审计，不混入最终指标。没有修改项目原有环境或训练代码。</p>
<pre>bash tools/hmi_evaluation/run_suite.sh reports/hmi_encoder_eval_20261110/validated
# 在 srlgym 环境中执行分析与报告生成
python tools/hmi_evaluation/analyze.py --output reports/hmi_encoder_eval_20261110/validated
python tools/hmi_evaluation/history_controls.py --output reports/hmi_encoder_eval_20261110/validated
python tools/hmi_evaluation/summary_intervals.py --output reports/hmi_encoder_eval_20261110/validated
python tools/hmi_evaluation/render_report.py --output reports/hmi_encoder_eval_20261110/validated</pre>
<p>机器可读结果：<a href="results.json">results.json</a> · <a href="verification.json">数据质量检查</a> · <a href="closed_loop_summary.csv">闭环汇总 CSV</a> · <a href="closed_loop_episodes.csv">逐回合 CSV</a> · <a href="prediction_metrics.csv">预测指标 CSV</a> · <a href="linear_probe_metrics.csv">线性探针 CSV</a> · <a href="action_interventions.csv">动作干预 CSV</a> · <a href="history_length_metrics.csv">历史长度 CSV</a></p>
<p>本报告的范围：单一已训练 checkpoint、原 S4 任务分布、确定性推理。未覆盖传感器噪声、时延泛化、人体体型变化、随机策略动作与实机部署；也没有以消融数据重新训练策略。它回答现有网络如何使用编码，不能替代编码结构之间的重新训练对比。</p></section>
{domain_adaptation}
<footer>实验指标来自实际仿真数据与冻结模型计算；当前架构依据代码核对，域适配部分为有文献依据的设计建议。图像内嵌于 HTML，可独立打开；SVG、PNG 和原始数值同时保存在实验目录。</footer></main></body></html>'''
    (out/'report.html').write_text(content)
    print(out/'report.html')


if __name__=='__main__':main()
