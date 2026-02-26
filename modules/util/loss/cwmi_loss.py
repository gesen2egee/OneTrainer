import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class ComplexSteerablePyramid(nn.Module):
    def __init__(self, *, levels: int = 4, orientations: int = 4):
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be >= 1")
        if orientations < 1:
            raise ValueError("orientations must be >= 1")

        self.levels = levels
        self.orientations = orientations
        self._mask_cache: Dict[Tuple[int, int, str, torch.dtype], Dict[str, List[Tensor]]] = {}

    @staticmethod
    def _down_sample(fourier_domain_image: Tensor) -> Tensor:
        _, _, height, width = fourier_domain_image.shape
        target_h = max(height // 2, 1)
        target_w = max(width // 2, 1)
        start_h = (height - target_h) // 2
        start_w = (width - target_w) // 2
        return fourier_domain_image[:, :, start_h:start_h + target_h, start_w:start_w + target_w]

    @staticmethod
    def _get_grid(height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        fy = torch.fft.fftfreq(height, d=1.0, device=device, dtype=dtype) * (2.0 * math.pi)
        fx = torch.fft.fftfreq(width, d=1.0, device=device, dtype=dtype) * (2.0 * math.pi)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(xx ** 2 + yy ** 2)
        theta = torch.atan2(yy, xx).remainder(2.0 * math.pi)
        radius = torch.fft.fftshift(radius)
        theta = torch.fft.fftshift(theta)
        return radius, theta

    def _get_masks(self, height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> Dict[str, List[Tensor]]:
        cache_key = (height, width, str(device), dtype)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        high_pass_filters: List[Tensor] = []
        low_pass_filters: List[Tensor] = []
        band_filters: List[Tensor] = []

        current_h = height
        current_w = width
        for level in range(self.levels + 1):
            radius, theta = self._get_grid(current_h, current_w, device=device, dtype=dtype)
            if level == 0:
                high_filter = torch.zeros_like(radius)
                transition = (radius > math.pi / 2.0) & (radius < math.pi)
                high_filter[transition] = torch.cos((math.pi / 2.0) * torch.log2(radius[transition] / math.pi))
                high_filter[radius >= math.pi] = 1.0

                low_filter = torch.zeros_like(radius)
                low_filter[transition] = torch.cos((math.pi / 2.0) * torch.log2((2.0 * radius[transition]) / math.pi))
                low_filter[radius <= math.pi / 2.0] = 1.0

                high_pass_filters.append(high_filter)
                low_pass_filters.append(low_filter)
                band_filters.append(torch.empty(0, device=device, dtype=dtype))
            else:
                high_filter = torch.zeros_like(radius)
                transition = (radius > math.pi / 4.0) & (radius < math.pi / 2.0)
                high_filter[transition] = torch.cos((math.pi / 2.0) * torch.log2((2.0 * radius[transition]) / math.pi))
                high_filter[radius >= math.pi / 2.0] = 1.0

                low_filter = torch.zeros_like(radius)
                low_filter[transition] = torch.cos((math.pi / 2.0) * torch.log2((4.0 * radius[transition]) / math.pi))
                low_filter[radius <= math.pi / 4.0] = 1.0

                alpha_k = (
                    (2 ** (self.orientations - 1)) * math.factorial(self.orientations - 1)
                ) / math.sqrt(self.orientations * math.factorial(2 * (self.orientations - 1)))
                band = torch.zeros((self.orientations, current_h, current_w), device=device, dtype=dtype)
                for k in range(self.orientations):
                    angle = theta - (math.pi * k / self.orientations)
                    directional = 2.0 * torch.abs(alpha_k * torch.relu(torch.cos(angle)) ** (self.orientations - 1))
                    band[k] = directional
                    band[k, current_h // 2, current_w // 2] = 0.0

                high_pass_filters.append(high_filter)
                low_pass_filters.append(low_filter)
                band_filters.append(band)

                current_h = max(current_h // 2, 1)
                current_w = max(current_w // 2, 1)

        masks = {"high": high_pass_filters, "low": low_pass_filters, "band": band_filters}
        self._mask_cache[cache_key] = masks
        return masks

    def forward(self, images: Tensor) -> List[Tensor]:
        if images.ndim != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], got {tuple(images.shape)}")

        _, _, height, width = images.shape
        masks = self._get_masks(height, width, device=images.device, dtype=images.dtype)

        fourier_domain = torch.fft.fftshift(torch.fft.fft2(images), dim=(-2, -1))
        output: List[Tensor] = []

        high_residue = fourier_domain * masks["high"][0].view(1, 1, height, width)
        output.append(torch.fft.ifft2(torch.fft.ifftshift(high_residue, dim=(-2, -1))))

        current = fourier_domain * masks["low"][0].view(1, 1, height, width)
        for level in range(1, self.levels + 1):
            _, _, h_level, w_level = current.shape
            high_mask = masks["high"][level].view(1, 1, 1, h_level, w_level)
            band_mask = masks["band"][level].view(1, 1, self.orientations, h_level, w_level)
            band_signal = current.unsqueeze(2) * high_mask * band_mask
            output.append(torch.fft.ifft2(torch.fft.ifftshift(band_signal, dim=(-2, -1))))
            current = self._down_sample(current * masks["low"][level].view(1, 1, h_level, w_level))

        return output


class CWMILoss(nn.Module):
    def __init__(self, *, levels: int = 4, orientations: int = 4, eps: float = 5e-4):
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be >= 1")
        if orientations < 1:
            raise ValueError("orientations must be >= 1")
        if eps <= 0:
            raise ValueError("eps must be > 0")

        self.levels = levels
        self.orientations = orientations
        self.eps = eps
        self._pyramid_cache: Dict[int, ComplexSteerablePyramid] = {}

    @staticmethod
    def _max_supported_levels(height: int, width: int) -> int:
        min_hw = min(height, width)
        if min_hw < 2:
            return 0
        return int(math.floor(math.log2(min_hw)))

    @staticmethod
    def _to_real_representation(x: Tensor) -> Tensor:
        # [B, C, K, HW] complex -> [B, C, 2K, 2HW]
        real = x.real
        imag = x.imag
        upper = torch.cat([real, imag], dim=2)
        lower = torch.cat([-imag, real], dim=2)
        return torch.cat([upper, lower], dim=3)

    def _safe_cholesky(self, matrix: Tensor) -> Tensor:
        dim = matrix.shape[-1]
        identity = torch.eye(dim, device=matrix.device, dtype=matrix.dtype).view(
            *([1] * (matrix.ndim - 2)), dim, dim
        )
        jitter = self.eps
        for _ in range(7):
            chol, info = torch.linalg.cholesky_ex(matrix + identity * jitter)
            if torch.all(info == 0):
                return chol
            jitter *= 10.0
        return torch.linalg.cholesky(matrix + identity * jitter)

    def _complex_negative_mi(self, target_band: Tensor, pred_band: Tensor) -> Tensor:
        # [B, C, K, H, W] complex
        batch_size, channels, _, height, width = target_band.shape

        target_flat = target_band.reshape(batch_size, channels, target_band.shape[2], height * width)
        pred_flat = pred_band.reshape(batch_size, channels, pred_band.shape[2], height * width)

        target_real = self._to_real_representation(target_flat)
        pred_real = self._to_real_representation(pred_flat)

        target_centered = target_real - target_real.mean(dim=-1, keepdim=True)
        pred_centered = pred_real - pred_real.mean(dim=-1, keepdim=True)

        cov_target = torch.matmul(target_centered, target_centered.transpose(-1, -2))
        cov_pred = torch.matmul(pred_centered, pred_centered.transpose(-1, -2))
        cov_target_pred = torch.matmul(target_centered, pred_centered.transpose(-1, -2))

        dim = cov_pred.shape[-1]
        identity = torch.eye(dim, device=cov_pred.device, dtype=cov_pred.dtype).view(
            *([1] * (cov_pred.ndim - 2)), dim, dim
        )
        inv_cov_pred = torch.linalg.inv(cov_pred + identity * self.eps)

        cond_cov = cov_target - torch.matmul(
            torch.matmul(cov_target_pred, inv_cov_pred), cov_target_pred.transpose(-1, -2)
        )
        cond_cov = 0.5 * (cond_cov + cond_cov.transpose(-1, -2))

        chol_cond = self._safe_cholesky(cond_cov)
        diag = torch.diagonal(chol_cond, dim1=-2, dim2=-1).clamp_min(1e-12)
        log_det = 2.0 * torch.sum(torch.log(diag), dim=-1)
        negative_mi = 0.5 * log_det
        return negative_mi.mean(dim=1)

    def _get_pyramid(self, levels: int) -> ComplexSteerablePyramid:
        pyramid = self._pyramid_cache.get(levels)
        if pyramid is None:
            pyramid = ComplexSteerablePyramid(levels=levels, orientations=self.orientations)
            self._pyramid_cache[levels] = pyramid
        return pyramid

    @staticmethod
    def _prepare_4d(x: Tensor) -> Tuple[Tensor, int, int]:
        if x.ndim == 4:
            return x, x.shape[0], 1
        if x.ndim == 5:
            # [B, C, F, H, W] -> [B*F, C, H, W]
            b, c, f, h, w = x.shape
            x4 = x.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
            return x4, b, f
        raise ValueError(f"Expected 4D or 5D tensor, got shape={tuple(x.shape)}")

    def forward(self, target: Tensor, pred: Tensor) -> Tensor:
        if target.shape != pred.shape:
            raise ValueError(f"target and pred must have same shape, got {tuple(target.shape)} vs {tuple(pred.shape)}")

        target4, batch_size, frames = self._prepare_4d(target.to(dtype=torch.float32))
        pred4, _, _ = self._prepare_4d(pred.to(dtype=torch.float32))

        _, _, height, width = target4.shape
        effective_levels = min(self.levels, self._max_supported_levels(height, width))
        if effective_levels <= 0:
            return torch.zeros(batch_size, device=target.device, dtype=target.dtype)

        pyramid = self._get_pyramid(effective_levels)
        target_bands = pyramid(target4)
        pred_bands = pyramid(pred4)

        frame_losses = torch.zeros(target4.shape[0], device=target4.device, dtype=torch.float32)
        for level in range(effective_levels):
            frame_losses += self._complex_negative_mi(target_bands[level + 1], pred_bands[level + 1])

        if target.ndim == 4:
            return frame_losses.to(dtype=target.dtype)

        return frame_losses.reshape(batch_size, frames).mean(dim=1).to(dtype=target.dtype)
