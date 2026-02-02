"""
MeanCache Wrapper for Z-Image Flow Matching Models

Based on UnicomAI MeanCache paper:
"From Instantaneous to Average Velocity for Accelerating Flow Matching Inference"
Reference: https://unicomai.github.io/MeanCache/

Key formulas:
- JVP approximation: JVP_{r→t} ≈ (v_t - v_r) / (t - r)
- Average velocity: û(z_t, t, s) = v(z_t, t) + (s - t) · JVP_{r→t}
- Stability deviation: L_K measures retrospective JVP accuracy
"""

import torch
from typing import Optional, Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Preset configurations matching original repo
MEANCACHE_PRESETS = {
    "quality": {
        "rel_l1_thresh": 0.15,
        "skip_budget": 0.25,
        "start_step": 3,
        "gamma": 2.0,
        "peak_threshold": 0.20,
    },
    "balanced": {
        "rel_l1_thresh": 0.12,
        "skip_budget": 0.35,
        "start_step": 2,
        "gamma": 2.0,
        "peak_threshold": 0.15,
    },
    "speed": {
        "rel_l1_thresh": 0.10,
        "skip_budget": 0.42,
        "start_step": 2,
        "gamma": 2.0,
        "peak_threshold": 0.12,
    },
    "turbo": {
        "rel_l1_thresh": 0.08,
        "skip_budget": 0.50,
        "start_step": 1,
        "gamma": 2.0,
        "peak_threshold": 0.10,
    },
}


class TrajectoryScheduler:
    """
    Implements Peak-Suppressed Shortest Path (PSSP) algorithm.
    Pre-computes a fixed skip schedule before sampling begins.
    """
    
    def __init__(
        self,
        total_steps: int,
        skip_budget: float = 0.3,
        gamma: float = 2.0,
        peak_threshold: float = 0.15,
        min_compute_steps: int = 4,
        critical_start_ratio: float = 0.20,
        critical_end_ratio: float = 0.85
    ):
        self.total_steps = max(1, total_steps)
        self.skip_budget = max(0.0, min(0.5, skip_budget))
        self.gamma = gamma
        self.peak_threshold = peak_threshold
        self.min_compute_steps = min_compute_steps
        self.critical_start_ratio = critical_start_ratio
        self.critical_end_ratio = critical_end_ratio
        
        # Pre-computed skip mask (True = skip this step)
        self.skip_mask: List[bool] = [False] * total_steps
        
        # Compute heuristic schedule at initialization
        self._compute_heuristic_schedule()
    
    def _compute_heuristic_schedule(self) -> None:
        """
        Compute initial heuristic schedule.
        Protects critical early/late steps and distributes skips evenly
        in the middle zone with minimum spacing.
        """
        n = self.total_steps
        max_skips = max(0, int(n * self.skip_budget))
        
        # Compute critical zone boundaries
        critical_start = max(1, int(n * self.critical_start_ratio))
        critical_end = min(n - 1, int(n * self.critical_end_ratio))
        
        # Initialize all as compute (False = don't skip)
        self.skip_mask = [False] * n
        
        # No skips if budget is 0 or too few steps
        if max_skips == 0 or n <= self.min_compute_steps:
            return
        
        # Identify skip candidates in middle zone
        skip_candidates = list(range(critical_start, critical_end))
        
        if len(skip_candidates) == 0:
            return
        
        # Calculate minimum spacing between skips
        min_spacing = max(2, len(skip_candidates) // (max_skips + 1))
        
        # Assign skips with even spacing
        skips_assigned = 0
        for i in range(0, len(skip_candidates), min_spacing):
            if skips_assigned >= max_skips:
                break
            idx = skip_candidates[i]
            self.skip_mask[idx] = True
            skips_assigned += 1
    
    def get_skip_decision(
        self,
        step_index: int,
        velocity_similarity: float,
        accumulated_error: float
    ) -> Tuple[bool, float]:
        """
        Get skip decision for current step using PSSP schedule.
        
        Uses pre-computed schedule with runtime velocity checks for safety.
        """
        max_accumulated = 0.5
        
        # Out of bounds check
        if step_index < 0 or step_index >= len(self.skip_mask):
            return False, 0.0
        
        scheduled_skip = self.skip_mask[step_index]
        
        # Override: don't skip if velocity changed too much (peak suppression)
        if scheduled_skip and velocity_similarity > self.peak_threshold:
            return False, 0.0
        
        # Override: don't skip if too much error accumulated
        if scheduled_skip and accumulated_error > max_accumulated * 0.8:
            return False, 0.0
        
        # Follow schedule
        if scheduled_skip:
            new_error = accumulated_error + velocity_similarity
            return True, new_error
        
        # Compute step - reset accumulated error
        return False, 0.0
    
    def get_schedule_summary(self) -> dict:
        """Get summary of the skip schedule."""
        skip_indices = [i for i, skip in enumerate(self.skip_mask) if skip]
        return {
            'total_steps': self.total_steps,
            'scheduled_skips': len(skip_indices),
            'skip_ratio': len(skip_indices) / max(1, self.total_steps),
            'skip_indices': skip_indices,
        }


class MeanCacheState:
    """Manages state across sampling steps for MeanCache algorithm."""
    
    def __init__(self, cache_device: str = 'cpu', max_cache_span: int = 3):
        self.cache_device = cache_device
        self.max_cache_span = max_cache_span
        self.states: Dict[int, Dict[str, Any]] = {}
        
    def get_or_create(self, pred_id: int) -> Dict[str, Any]:
        """Get existing state or create new one."""
        if pred_id not in self.states:
            self.states[pred_id] = {
                'v_cache': None,           # Cached velocity for skip reuse
                'jvp_cache': None,         # Cached JVP for velocity correction
                'sigma_cache': None,       # Sigma at which v_cache was computed
                'v_history': [],           # Recent K velocities
                't_history': [],           # Corresponding timesteps
                'accumulated_error': 0.0,  # Accumulated error for adaptive thresholding
                'skipped_steps': [],       # List of skipped step indices
                'step_index': 0,           # Current step counter
            }
        return self.states[pred_id]
    
    def update_history(self, pred_id: int, velocity: torch.Tensor, timestep: float):
        """Update velocity and timestep history for JVP_K computation."""
        state = self.get_or_create(pred_id)
        v_cached = velocity.detach().clone().to(self.cache_device)
        state['v_history'].append(v_cached)
        state['t_history'].append(timestep)
        
        # Maintain sliding window
        max_len = self.max_cache_span + 1
        if len(state['v_history']) > max_len:
            old_v = state['v_history'].pop(0)
            del old_v
            state['t_history'].pop(0)
    
    def get_jvp_k(self, pred_id: int, k: int, eps: float = 1e-8) -> Optional[torch.Tensor]:
        """Compute JVP with cache span K from stored history."""
        state = self.get_or_create(pred_id)
        v_history = state['v_history']
        t_history = state['t_history']
        
        if len(v_history) < k + 1:
            return None
        
        v_now = v_history[-1]
        v_k_ago = v_history[-(k + 1)]
        t_now = t_history[-1]
        t_k_ago = t_history[-(k + 1)]
        
        # Z-Image: normalized_timestep INCREASES (0→1), so t_now > t_k_ago
        dt = t_now - t_k_ago
        if abs(dt) < eps:
            return None
        
        return (v_now - v_k_ago.to(v_now.device)) / dt
    
    def clear_all(self):
        """Reset all states between sampling runs."""
        for state in self.states.values():
            for v in state.get('v_history', []):
                del v
        self.states = {}


def get_optimal_k(sigma: float, max_k: int = 3) -> int:
    """
    Select optimal JVP lookback K based on sigma (normalized timestep).
    For Z-Image: sigma goes from ~0 (start) to ~1 (end).
    """
    if sigma < 0.3:
        return 1  # Early: fast changes, short lookback
    elif sigma < 0.6:
        return min(max_k, 2)
    elif sigma < 0.8:
        return min(max_k, 3)
    else:
        return max_k  # Late: stable, max smoothing


def compute_online_L_K(
    v_new: torch.Tensor,
    v_cached: torch.Tensor,
    jvp_cached: torch.Tensor,
    dt_elapsed: float,
    eps: float = 1e-8
) -> float:
    """
    Compute online approximation of paper's L_K stability deviation.
    L_K ≈ ||v_new - (v_cached + dt * JVP_cached)|| / ||v_new||
    """
    predicted_v = v_cached.to(v_new.device) + dt_elapsed * jvp_cached.to(v_new.device)
    prediction_error = torch.abs(v_new - predicted_v).mean()
    normalizer = torch.abs(v_new).mean() + eps
    return (prediction_error / normalizer).item()


def compute_velocity_similarity(v_current: torch.Tensor, v_cache: torch.Tensor) -> float:
    """Compute relative L1 distance for quick similarity check."""
    v_cache = v_cache.to(v_current.device)
    l1_distance = torch.abs(v_current - v_cache).mean()
    norm = torch.abs(v_cache).mean() + 1e-8
    return (l1_distance / norm).item()


class ZImageMeanCacheWrapper:
    """
    A wrapper for Z-Image transformer that enables MeanCache acceleration.
    
    Uses PSSP scheduler to pre-compute a FIXED skip schedule,
    then follows that schedule during sampling.
    """
    
    def __init__(
        self,
        transformer: torch.nn.Module,
        preset: str = "balanced",
        enabled: bool = True,
        total_steps: int = 30,
    ):
        self.transformer = transformer
        self.enabled = enabled
        self.preset = preset
        
        # Get config from preset
        self.config = MEANCACHE_PRESETS.get(preset, MEANCACHE_PRESETS["balanced"])
        
        # Initialize PSSP scheduler with fixed schedule
        self.scheduler = TrajectoryScheduler(
            total_steps=total_steps,
            skip_budget=self.config['skip_budget'],
            gamma=self.config['gamma'],
            peak_threshold=self.config['peak_threshold'],
        )
        
        # Log the pre-computed schedule
        summary = self.scheduler.get_schedule_summary()
        logger.info(f"[MeanCache] Pre-computed schedule: {summary['scheduled_skips']}/{summary['total_steps']} steps will be skipped")
        logger.info(f"[MeanCache] Skip indices: {summary['skip_indices']}")
        
        # Initialize state
        self.state = MeanCacheState(cache_device='cpu', max_cache_span=3)
        
        # Statistics
        self.total_steps = 0
        self.skipped_steps = 0
        self.computed_steps = 0
        
    def reset_state(self):
        """Reset internal state for a new sampling run."""
        self.state.clear_all()
        self.total_steps = 0
        self.skipped_steps = 0
        self.computed_steps = 0
        
    def __call__(
        self,
        latent_list: list[torch.Tensor],
        timestep: torch.Tensor,
        prompt_embedding: torch.Tensor,
        **kwargs
    ) -> Any:
        """
        Forward pass with MeanCache acceleration for Z-Image.
        Uses pre-computed PSSP schedule for consistent skip decisions.
        """
        if not self.enabled:
            return self.transformer(latent_list, timestep, prompt_embedding, **kwargs)
        
        # Get current sigma (normalized timestep 0-1)
        current_sigma = float(timestep.flatten()[0])
        
        batch_size = len(latent_list)
        
        # Get step index from state (use pred_id=0 as reference)
        pred_state = self.state.get_or_create(0)
        step_index = pred_state['step_index']
        
        # Check if in active range (skip first few steps)
        in_active_range = step_index >= self.config['start_step']
        
        # Get skip decision from pre-computed schedule
        should_skip = False
        if in_active_range and pred_state['v_cache'] is not None:
            accumulated_error = pred_state.get('accumulated_error', 0.0)
            # Velocity similarity from last step (use cached deviation)
            velocity_similarity = accumulated_error
            
            should_skip, new_error = self.scheduler.get_skip_decision(
                step_index=step_index,
                velocity_similarity=velocity_similarity,
                accumulated_error=accumulated_error
            )
            
            # Update accumulated error for all predictions
            for pid in range(batch_size):
                self.state.get_or_create(pid)['accumulated_error'] = new_error
        
        if should_skip:
            # SKIP: Apply JVP correction to cached velocity
            outputs = []
            for pred_id in range(batch_size):
                pred_state = self.state.get_or_create(pred_id)
                v_cache = pred_state['v_cache'].to(latent_list[0].device)
                jvp_cache = pred_state.get('jvp_cache')
                sigma_cache = pred_state.get('sigma_cache', current_sigma)
                
                if jvp_cache is not None:
                    # Apply JVP correction: v_corrected = v_cache + dt * JVP
                    # Z-Image: normalized_timestep INCREASES, so current > cache
                    dt = current_sigma - sigma_cache
                    jvp_device = jvp_cache.to(latent_list[0].device)
                    v_corrected = v_cache + dt * jvp_device
                    outputs.append(v_corrected)
                else:
                    outputs.append(v_cache)
                
                pred_state['skipped_steps'].append(step_index)
                pred_state['step_index'] += 1
            
            self.total_steps += 1
            self.skipped_steps += 1
            
            # Return fake output structure matching transformer output
            class FakeOutput:
                def __init__(self, sample):
                    self.sample = sample
            return FakeOutput(sample=outputs)
        
        # COMPUTE: Run full transformer
        self.total_steps += 1
        self.computed_steps += 1
        
        output = self.transformer(latent_list, timestep, prompt_embedding, **kwargs)
        
        # Update state for each prediction
        for pred_id in range(batch_size):
            pred_state = self.state.get_or_create(pred_id)
            
            # Get the velocity output for this prediction
            v_current = output.sample[pred_id]
            
            # Update velocity history for JVP_K computation
            self.state.update_history(pred_id, v_current, current_sigma)
            
            # Try to compute JVP using multi-step history
            jvp = None
            optimal_k = get_optimal_k(current_sigma, self.state.max_cache_span)
            jvp = self.state.get_jvp_k(pred_id, optimal_k)
            
            # Fallback to smaller K if optimal not available
            if jvp is None:
                for k in range(optimal_k - 1, 0, -1):
                    jvp_k = self.state.get_jvp_k(pred_id, k)
                    if jvp_k is not None:
                        jvp = jvp_k
                        break
            
            # Compute deviation for next step's skip decision
            v_cache_old = pred_state.get('v_cache')
            jvp_cache_old = pred_state.get('jvp_cache')
            sigma_cache_old = pred_state.get('sigma_cache')
            
            if v_cache_old is not None and jvp_cache_old is not None and sigma_cache_old is not None:
                # Z-Image: normalized_timestep INCREASES, so current > cache
                dt_elapsed = current_sigma - sigma_cache_old
                deviation = compute_online_L_K(v_current, v_cache_old, jvp_cache_old, dt_elapsed)
            elif v_cache_old is not None:
                deviation = compute_velocity_similarity(v_current, v_cache_old)
            else:
                deviation = 0.0  # First step
            
            pred_state['accumulated_error'] = deviation
            
            # Cache current velocity, JVP, and sigma
            pred_state['v_cache'] = v_current.detach().clone().to(self.state.cache_device)
            pred_state['sigma_cache'] = current_sigma
            if jvp is not None:
                pred_state['jvp_cache'] = jvp.detach().clone().to(self.state.cache_device)
            
            pred_state['step_index'] += 1
        
        return output
    
    def get_stats(self) -> dict:
        """Get sampling statistics."""
        skip_rate = self.skipped_steps / max(self.total_steps, 1)
        speedup = 1.0 / (1.0 - skip_rate) if skip_rate < 1.0 else 1.0
        
        return {
            "total_steps": self.total_steps,
            "skipped_steps": self.skipped_steps,
            "computed_steps": self.computed_steps,
            "skip_rate": skip_rate,
            "speedup": speedup,
            "scheduled_skip_indices": self.scheduler.get_schedule_summary()['skip_indices'],
        }
    
    def print_summary(self):
        """Print sampling summary to console."""
        stats = self.get_stats()
        print(
            f"[MeanCache] Sampling complete ({self.preset}): "
            f"{stats['total_steps']} steps, "
            f"{stats['skipped_steps']} skipped, "
            f"{stats['computed_steps']} computed "
            f"({stats['skip_rate']*100:.1f}% skip rate, ~{stats['speedup']:.2f}x speedup)"
        )
