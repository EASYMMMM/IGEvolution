import torch

class TensorizedMuscleGroup:
    """
    全张量化 Hill-Type 多关节耦合肌肉模型 (Jacobian-based Routing)
    总肌肉数 58: 44 根普通单关节拮抗肌 + 14 根基于 Geyer 2010 解剖学的下肢耦合肌肉
    """
    def __init__(self, num_envs, num_dofs, device, dt, control_freq_inv,
                 dof_lower, dof_upper, dof_damping, dof_effort, q_init, sagittal_dofs,
                 use_biarticular=False): # 🔥 新增开关，默认关闭
        self.num_envs = num_envs
        self.num_dofs = num_dofs
        self.device = device
        self.sim_dt = dt
        self.control_steps = control_freq_inv
        self.dt_a = 0.01

        # 根据开关决定肌肉总数
        self.use_biarticular = use_biarticular
        self.num_muscles = 58 if use_biarticular else num_dofs * 2

        self.a = torch.zeros((num_envs, self.num_muscles), dtype=torch.float, device=device)

        range_q = torch.clamp(dof_upper - dof_lower, min=0.01)
        safe_damping = torch.clamp(dof_damping, min=2.0)

        self.M = torch.zeros((self.num_muscles, self.num_dofs), dtype=torch.float, device=device)
        self.f_max = torch.zeros(self.num_muscles, dtype=torch.float, device=device)
        self.beta = torch.zeros(self.num_muscles, dtype=torch.float, device=device)

        muscle_idx = 0

        if self.use_biarticular:
            # ================== 58维: 多关节耦合模式 ==================
            sagittal_dofs_flat = sagittal_dofs['right'] + sagittal_dofs['left']
            # 1. 非矢状面关节 (44根)
            for dof in range(self.num_dofs):
                if dof not in sagittal_dofs_flat:
                    r_val = 1.0 / range_q[dof]
                    self.M[muscle_idx, dof] = -r_val
                    self.f_max[muscle_idx] = torch.clamp(dof_effort[dof] * 1.5, 150.0, 1000.0)
                    self.beta[muscle_idx] = torch.clamp(safe_damping[dof] / (4.0 * abs(r_val) * self.f_max[muscle_idx]), 0.01, 1.0)
                    muscle_idx += 1

                    self.M[muscle_idx, dof] = r_val
                    self.f_max[muscle_idx] = torch.clamp(dof_effort[dof] * 1.5, 150.0, 1000.0)
                    self.beta[muscle_idx] = torch.clamp(safe_damping[dof] / (4.0 * abs(r_val) * self.f_max[muscle_idx]), 0.01, 1.0)
                    muscle_idx += 1

            # 2. 矢状面关节 (14根)
            def add_sagittal_muscles(hip, knee, ankle):
                nonlocal muscle_idx
                def add_m(dofs, rs):

                    nonlocal muscle_idx
                    scale_r = 6.0
                    scaled_rs = [r * scale_r for r in rs]
                    for d, r in zip(dofs, scaled_rs): self.M[muscle_idx, d] = r
                    eff = max([dof_effort[d] for d in dofs])
                    dmp = sum([safe_damping[d] for d in dofs]) / len(dofs)
                    avg_r = sum([abs(r) for r in scaled_rs]) / len(scaled_rs)
                    self.f_max[muscle_idx] = torch.clamp(eff * 1.5, 150.0, 1000.0)
                    self.beta[muscle_idx] = torch.clamp(dmp / (4.0 * avg_r * self.f_max[muscle_idx]), 0.01, 1.0)

                    muscle_idx += 1

                add_m([hip],[-0.10])
                add_m([hip],[0.10])
                add_m([hip, knee],[-0.10, -0.05])
                add_m([knee], [0.16])
                add_m([knee, ankle],[-0.05, -0.05])
                add_m([ankle],[-0.05])
                add_m([ankle],[0.04])

            add_sagittal_muscles(*sagittal_dofs['right'])
            add_sagittal_muscles(*sagittal_dofs['left'])
        else:
            # ================== 56维: 纯单关节退回模式 ==================
            for dof in range(self.num_dofs):
                r_val = 1.0 / range_q[dof]
                # Agonist
                self.M[muscle_idx, dof] = -r_val
                self.f_max[muscle_idx] = torch.clamp(dof_effort[dof] * 1.5, 150.0, 1000.0)
                self.beta[muscle_idx] = torch.clamp(safe_damping[dof] / (4.0 * abs(r_val) * self.f_max[muscle_idx]), 0.01, 1.0)
                muscle_idx += 1
                # Antagonist
                self.M[muscle_idx, dof] = r_val
                self.f_max[muscle_idx] = torch.clamp(dof_effort[dof] * 1.5, 150.0, 1000.0)
                self.beta[muscle_idx] = torch.clamp(safe_damping[dof] / (4.0 * abs(r_val) * self.f_max[muscle_idx]), 0.01, 1.0)
                muscle_idx += 1

        self.l_ref = 1.0 - torch.matmul(self.M, q_init)

    def _force_length_curve(self, l):
        return torch.exp(-4.0 * (l - 1.0)**2)

    def _force_velocity_curve(self, v):
        v_max = 1.0
        fv = torch.where(v < 0, (v_max + v) / v_max, (v_max + 1.5 * v) / (v_max + v))
        return torch.clamp(fv, min=0.0, max=1.5)

    def _passive_force_curve(self, l):
        strain = torch.clamp(l - 1.05, min=0.0)
        fp_multiplier = 1.5 * (strain ** 2)
        return torch.clamp(fp_multiplier, max=0.5)

    def forward(self, q, dq, target_a, return_details=False, disable_passive=True):
        # 1. 激活度更新
        sub_dt = self.sim_dt
        for _ in range(self.control_steps):
            da = (target_a - self.a) / self.dt_a
            self.a = torch.clamp(self.a + da * sub_dt, 0.0, 1.0)

        # 2. 运动学映射 (两行代码搞定所有复杂耦合！)
        # L = q @ M^T + L_ref
        L = torch.matmul(q, self.M.T) + self.l_ref.unsqueeze(0)
        V = self.beta.unsqueeze(0) * torch.matmul(dq, self.M.T)

        # 3. 主动力计算
        F_active = self.f_max.unsqueeze(0) * (self.a * self._force_length_curve(L) * self._force_velocity_curve(V))

        # 4. 被动力与双向平滑粘滞阻尼计算
        if disable_passive:
            F_passive = 0.0
        else:
            tissue_viscosity = 0.5
            viscous_F = tissue_viscosity * V
            F_passive = self.f_max.unsqueeze(0) * self._passive_force_curve(L) + self.f_max.unsqueeze(0) * viscous_F
            F_passive = torch.clamp(F_passive, min=0.0)

        F = F_active + F_passive

        # 5. 动力学逆映射: 转换为关节力矩
        # tau = - F @ M
        tau = -torch.matmul(F, self.M)

        if return_details:
            return tau, {'a': self.a, 'L': L, 'V': V, 'F': F}

        return tau
