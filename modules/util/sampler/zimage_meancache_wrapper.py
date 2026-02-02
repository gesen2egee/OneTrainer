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
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# Preset configurations
MEANCACHE_PRESETS = {
    "quality": {"rel_l1_thresh": 0.20, "skip_budget": 0.25, "start_step": 3},   # ~1.3x
    "balanced": {"rel_l1_thresh": 0.15, "skip_budget": 0.35, "start_step": 2},  # ~1.5x  
    "speed": {"rel_l1_thresh": 0.12, "skip_budget": 0.42, "start_step": 2},     # ~1.7x
    "turbo": {"rel_l1_thresh": 0.10, "skip_budget": 0.50, "start_step": 1},     # ~2.0x
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
        
        dt = t_k_ago - t_now  # sigma decreases, so t_k_ago > t_now
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
    """Select optimal JVP lookback K based on sigma (noise level)."""
    if sigma > 0.5:
        return 1  # Early: fast changes, short lookback
    elif sigma > 0.2:
        return min(max_k, 2)
    elif sigma > 0.1:
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
    
    Measures retrospective JVP accuracy: how accurate was the cached
    JVP extrapolation at predicting the new velocity?
    
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


def should_skip_step(
    stability_deviation: float,
    threshold: float,
    accumulated_error: float,
    max_accumulated: float = 0.5
) -> tuple[bool, float]:
    """Decide whether to skip current step based on stability deviation."""
    # Adaptive threshold with peak suppression
    accumulation_factor = 1.0 - (accumulated_error / max_accumulated)
    effective_threshold = threshold * max(0.1, accumulation_factor)
    
    if stability_deviation < effective_threshold:
        new_accumulated = accumulated_error + stability_deviation
        if new_accumulated < max_accumulated:
            return True, new_accumulated
    
    # Reset accumulated error on compute
    return False, 0.0


class ZImageMeanCacheWrapper:
    """
    A wrapper for Z-Image transformer that enables MeanCache acceleration.
    
    This wrapper intercepts forward() calls and decides whether to:
    1. Compute the actual velocity (first few steps, or when needed)
    2. Use cached JVP-corrected velocity (when stable)
    """
    
    def __init__(
        self,
        transformer: torch.nn.Module,
        preset: str = "balanced",
        enabled: bool = True,
    ):
        self.transformer = transformer
        self.enabled = enabled
        self.preset = preset
        
        # Get config from preset
        self.config = MEANCACHE_PRESETS.get(preset, MEANCACHE_PRESETS["balanced"])
        
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
        
        The transformer returns output with .sample attribute containing a list of tensors.
        """
        if not self.enabled:
            return self.transformer(latent_list, timestep, prompt_embedding, **kwargs)
        
        # Get current sigma (normalized timestep 0-1, decreasing)
        current_sigma = float(timestep.flatten()[0])
        
        # Check step index for each prediction in batch
        batch_size = len(latent_list)
        
        # Process each prediction in the batch
        outputs = []
        can_skip_all = True
        
        for pred_id in range(batch_size):
            pred_state = self.state.get_or_create(pred_id)
            step_index = pred_state['step_index']
            
            # Check if in active range (skip first few steps)
            in_active_range = step_index >= self.config['start_step']
            
            if not in_active_range or pred_state['v_cache'] is None:
                can_skip_all = False
                break
            
            # Check skip decision based on accumulated error
            accumulated_error = pred_state.get('accumulated_error', 1.0)
            should_skip, _ = should_skip_step(
                stability_deviation=accumulated_error,
                threshold=self.config['rel_l1_thresh'],
                accumulated_error=accumulated_error
            )
            
            if not should_skip:
                can_skip_all = False
                break
        
        if can_skip_all:
            # All predictions can be skipped - apply JVP correction
            for pred_id in range(batch_size):
                pred_state = self.state.get_or_create(pred_id)
                v_cache = pred_state['v_cache'].to(latent_list[0].device)
                jvp_cache = pred_state.get('jvp_cache')
                sigma_cache = pred_state.get('sigma_cache', current_sigma)
                
                if jvp_cache is not None:
                    # Apply JVP correction: v_corrected = v_cache + dt * JVP
                    dt = sigma_cache - current_sigma  # sigma decreases
                    jvp_device = jvp_cache.to(latent_list[0].device)
                    v_corrected = v_cache + dt * jvp_device
                    outputs.append(v_corrected)
                else:
                    outputs.append(v_cache)
                
                pred_state['skipped_steps'].append(pred_state['step_index'])
                pred_state['step_index'] += 1
            
            self.total_steps += 1
            self.skipped_steps += 1
            
            # Return fake output structure
            class FakeOutput:
                def __init__(self, sample):
                    self.sample = sample
            return FakeOutput(sample=outputs)
        
        # Need to compute - run full transformer
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
            
            # Compute deviation for adaptive skip decision using paper's L_K metric
            v_cache_old = pred_state.get('v_cache')
            jvp_cache_old = pred_state.get('jvp_cache')
            sigma_cache_old = pred_state.get('sigma_cache')
            
            if v_cache_old is not None and jvp_cache_old is not None and sigma_cache_old is not None:
                dt_elapsed = sigma_cache_old - current_sigma
                deviation = compute_online_L_K(v_current, v_cache_old, jvp_cache_old, dt_elapsed)
            elif v_cache_old is not None:
                deviation = compute_velocity_similarity(v_current, v_cache_old)
            else:
                deviation = 1.0  # Force compute on first step
            
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
