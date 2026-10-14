"""World-XY admittance and horizontal force scheduling; no simulator dependency."""
import math

import torch


def _positive(config, key, default, allow_zero=False):
    value = float(config.get(key, default))
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{key} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def _range(config, key, default, allow_zero=False):
    bounds = tuple(float(x) for x in config.get(key, default))
    if (len(bounds) != 2 or not all(math.isfinite(x) for x in bounds)
            or bounds[0] < 0 or bounds[1] < bounds[0]
            or (bounds[0] == 0 and not allow_zero)):
        raise ValueError(f"Invalid {key}: {bounds}")
    return bounds


def free_response_coefficients(mass, damping, stiffness, horizon):
    """Exact homogeneous oscillator transition, including critical/over damping."""
    a = damping / (2.0 * mass)
    discriminant = a * a - stiffness / mass
    decay = math.exp(-a * horizon)
    if abs(discriminant) < 1e-12:
        c, s = 1.0, horizon
    elif discriminant < 0:
        w = math.sqrt(-discriminant)
        c, s = math.cos(w * horizon), math.sin(w * horizon) / w
    else:
        w = math.sqrt(discriminant)
        # Combine exponents first to avoid cosh overflow for strong damping.
        e1, e2 = math.exp((-a + w) * horizon), math.exp((-a - w) * horizon)
        c, s = (e1 + e2) * 0.5, (e1 - e2) / (2.0 * w)
        return c + a * s, s, -stiffness / mass * s, c - a * s
    return decay * (c + a * s), decay * s, -decay * stiffness / mass * s, decay * (c - a * s)


class ComplianceController:
    def __init__(self, num_envs, device, compliance, perturbation, physics_dt):
        self.num_envs, self.device = num_envs, device
        self.enabled = bool(compliance.get("enabled", False))
        self.force_enabled = bool(perturbation.get("enabled", False))
        self.active = self.enabled or self.force_enabled
        self.randomize = bool(perturbation.get("randomize", False))
        self.test_mode = bool(perturbation.get("test_mode", False))
        self.random_allowed = self.randomize and (not self.test_mode or bool(perturbation.get("randomize_in_test", False)))
        self.mass = _positive(compliance, "mass", 70)
        self.damping = _positive(compliance, "damping", 35)
        self.stiffness = _positive(compliance, "stiffness", 45)
        self.filter_tau = _positive(compliance, "force_filter_tau", 0.03, True)
        self.max_displacement = _positive(compliance, "max_displacement", 0.75)
        self.force_reference = _positive(compliance, "force_reference", 220)
        self.alpha_range = _range(compliance, "alpha_range", (0.04, 0.16), True)
        self.force_range = _range(perturbation, "force_range", (80, 120), True)
        self.duration_range = _range(perturbation, "duration_range", (0.2, 0.5))
        self.interval_range = _range(perturbation, "interval_range", (3, 6), True)
        if self.duration_range[0] < physics_dt:
            raise ValueError("duration_range minimum must be at least one physics step")
        self.quiet_time = _positive(perturbation, "reset_quiet_time", 1, True)
        self.clean_probability = float(perturbation.get("clean_episode_probability", 0.25))
        if not 0 <= self.clean_probability <= 1:
            raise ValueError("clean_episode_probability must be in [0, 1]")
        self.fixed = dict(perturbation.get("fixed", {}))
        self.fixed_time = _positive(self.fixed, "start_time", 2, True)
        self.fixed_magnitude = _positive(self.fixed, "magnitude", 100, True)
        self.fixed_duration = _positive(self.fixed, "duration", 0.5)
        self.fixed_angle = float(self.fixed.get("angle_deg", 90))
        if not math.isfinite(self.fixed_angle) or self.fixed_duration < physics_dt:
            raise ValueError("Invalid fixed pulse angle/duration")
        self.delta = torch.zeros((num_envs, 2), device=device)
        self.velocity = torch.zeros_like(self.delta)
        self.filtered_force = torch.zeros_like(self.delta)
        self.pulse_force = torch.zeros_like(self.delta)
        self.force = torch.zeros((num_envs, 3), device=device)
        self.remaining = torch.zeros(num_envs, device=device)
        self.cooldown = torch.zeros_like(self.remaining)
        self.age = torch.zeros_like(self.remaining)
        self.clean_episode = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.fixed_fired = torch.zeros_like(self.clean_episode)
        self.event_count = torch.zeros_like(self.remaining)
        self.impulse = torch.zeros_like(self.delta)
        self._forecast_cache = {}

    def _sample(self, bounds, n):
        return bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n, device=self.device)

    def reset(self, env_ids):
        for buffer in (self.delta, self.velocity, self.filtered_force, self.pulse_force,
                       self.force, self.remaining, self.age, self.event_count, self.impulse):
            buffer[env_ids] = 0
        self.fixed_fired[env_ids] = False
        if self.force_enabled and self.random_allowed:
            self.cooldown[env_ids] = self.quiet_time + self._sample(self.interval_range, len(env_ids))
            self.clean_episode[env_ids] = torch.rand(len(env_ids), device=self.device) < self.clean_probability
        else:
            # No extra RNG consumption when new functionality is disabled.
            self.cooldown[env_ids] = self.quiet_time
            self.clean_episode[env_ids] = False

    def trigger(self, env_ids, magnitude=None, angle_deg=None, duration=None):
        """Manual world-direction pulse; it remains gated by the force master switch."""
        if not self.force_enabled:
            return
        magnitude = self.fixed_magnitude if magnitude is None else float(magnitude)
        duration = self.fixed_duration if duration is None else float(duration)
        angle = self.fixed_angle if angle_deg is None else float(angle_deg)
        if not all(math.isfinite(v) for v in (magnitude, duration, angle)) or magnitude < 0 or duration <= 0:
            raise ValueError("Invalid manual pulse")
        radians = math.radians(angle)
        self.pulse_force[env_ids, 0] = magnitude * math.cos(radians)
        self.pulse_force[env_ids, 1] = magnitude * math.sin(radians)
        self.remaining[env_ids] = duration
        self.event_count[env_ids] += 1

    def advance(self, dt):
        """Return interval-average force, preserving pulse impulse at partial steps."""
        idle = self.remaining <= 1e-7
        self.cooldown = torch.where(idle, self.cooldown - dt, self.cooldown)
        if self.force_enabled and self.random_allowed:
            due = idle & (self.cooldown <= 0) & ~self.clean_episode
            angle = torch.rand(self.num_envs, device=self.device) * (2 * math.pi)
            magnitude = self._sample(self.force_range, self.num_envs)
            sampled_force = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1) * magnitude[:, None]
            self.pulse_force = torch.where(due[:, None], sampled_force, self.pulse_force)
            self.remaining = torch.where(due, self._sample(self.duration_range, self.num_envs), self.remaining)
            self.event_count += due.to(self.event_count.dtype)
        if self.force_enabled and bool(self.fixed.get("enabled", False)):
            due = (self.age + 1e-7 >= self.fixed_time) & ~self.fixed_fired
            # Fixed tests take precedence over a coincident random event.
            angle = math.radians(self.fixed_angle)
            fixed_force = self.pulse_force.new_tensor([math.cos(angle), math.sin(angle)]) * self.fixed_magnitude
            self.pulse_force = torch.where(due[:, None], fixed_force, self.pulse_force)
            self.remaining = torch.where(due, torch.full_like(self.remaining, self.fixed_duration), self.remaining)
            self.event_count += due.to(self.event_count.dtype)
            self.fixed_fired |= due
        fraction = torch.clamp(self.remaining / dt, 0, 1)
        self.force.zero_()
        if self.force_enabled:
            self.force[:, :2] = self.pulse_force * fraction[:, None]
        ended = (self.remaining > 1e-7) & (self.remaining <= dt + 1e-7)
        self.remaining = torch.clamp(self.remaining - dt, min=0)
        if self.force_enabled and self.random_allowed:
            self.cooldown = torch.where(ended, self._sample(self.interval_range, self.num_envs), self.cooldown)
        self.impulse += self.force[:, :2] * dt
        self.age += dt
        if self.enabled:
            blend = 1.0 if self.filter_tau == 0 else -math.expm1(-dt / self.filter_tau)
            self.filtered_force += blend * (self.force[:, :2] - self.filtered_force)
            ratio = torch.clamp(torch.linalg.vector_norm(self.filtered_force, dim=-1) / self.force_reference, 0, 1)
            alpha = self.alpha_range[0] + (self.alpha_range[1] - self.alpha_range[0]) * ratio
            acceleration = (alpha[:, None] * self.filtered_force - self.damping * self.velocity - self.stiffness * self.delta) / self.mass
            self.velocity += dt * acceleration
            self.delta += dt * self.velocity
            norm = torch.linalg.vector_norm(self.delta, dim=-1, keepdim=True).clamp_min(1e-9)
            limited = norm > self.max_displacement
            self.delta *= torch.clamp(self.max_displacement / norm, max=1)
            normal = self.delta / torch.linalg.vector_norm(self.delta, dim=-1, keepdim=True).clamp_min(1e-9)
            outward = torch.clamp(torch.sum(self.velocity * normal, dim=-1, keepdim=True), min=0)
            self.velocity -= torch.where(limited, outward * normal, torch.zeros_like(self.velocity))
        return self.force

    def forecast(self, env_ids, horizons):
        """No future force events: free return from current delta/velocity only."""
        key = tuple(float(h) for h in horizons)
        if key not in self._forecast_cache:
            coeffs = [free_response_coefficients(self.mass, self.damping, self.stiffness, h)[:2] for h in key]
            self._forecast_cache[key] = self.delta.new_tensor(coeffs)
        coeffs = self._forecast_cache[key]
        predicted = self.delta[env_ids, None, :] * coeffs[None, :, 0:1] + self.velocity[env_ids, None, :] * coeffs[None, :, 1:2]
        norm = torch.linalg.vector_norm(predicted, dim=-1, keepdim=True).clamp_min(1e-9)
        return predicted * torch.clamp(self.max_displacement / norm, max=1)
