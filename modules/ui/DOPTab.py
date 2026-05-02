import customtkinter as ctk

from modules.util import dop_util
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DOPPolicy import DOPPolicy
from modules.util.ui import components
from modules.util.ui.UIState import UIState
from modules.util.ui.validation_helpers import check_non_negative_or_minus_one, check_range


class DOPTab:
    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        self.master = master
        self.train_config = train_config
        self.ui_state = ui_state
        self.scroll_frame = None
        self.warning_label = None
        self._trace_hooks = []
        self.refresh_ui()

    def refresh_ui(self):
        if self.scroll_frame:
            self.scroll_frame.destroy()
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        self.scroll_frame = ctk.CTkScrollableFrame(self.master, fg_color="transparent")
        self.scroll_frame.grid(row=0, column=0, sticky="nsew")
        frame = self.scroll_frame
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)

        row = 0
        self.warning_label = ctk.CTkLabel(
            frame,
            text="",
            justify="left",
            wraplength=900,
            text_color="#f1c40f",
        )
        self.warning_label.grid(row=row, column=0, columnspan=2, padx=10, pady=(10, 0), sticky="w")
        row += 1
        row = self._build_enable_mode_group(frame, row)
        row = self._build_schedule_group(frame, row)
        row = self._build_prompt_group(frame, row)
        row = self._build_safety_group(frame, row)
        self._build_notes_group(frame, row)
        self._install_warning_traces()
        self._update_text_encoder_warning()

    def _build_enable_mode_group(self, frame, row: int) -> int:
        components.label(frame, row, 0, "Enable / Mode", tooltip="Core DOP toggles and high-level behavior.")
        row += 1
        components.label(
            frame, row, 0, "Enable DOP",
            tooltip="Turns Differential Output Preservation on/off. When enabled, OneTrainer adds an extra preservation loss."
        )
        components.switch(frame, row, 1, self.ui_state, "dop_enabled")
        row += 1
        components.label(
            frame, row, 0, "Preset",
            tooltip="quality: strongest preservation (highest cost); balanced: periodic preservation; fast: adaptive sparse preservation; manual: keep your custom policy values."
        )
        components.options(frame, row, 1, ["quality", "balanced", "fast", "manual"], self.ui_state, "dop_preset", command=self._apply_dop_preset)
        row += 1
        components.label(
            frame, row, 0, "Policy",
            tooltip="always_on: every step; periodic: every N steps; adaptive: denser early and sparser later; manual: follow manual interval/start/end fields."
        )
        components.options_kv(frame, row, 1, [
            ("always_on", DOPPolicy.ALWAYS_ON),
            ("periodic", DOPPolicy.PERIODIC),
            ("adaptive", DOPPolicy.ADAPTIVE),
            ("manual", DOPPolicy.MANUAL),
        ], self.ui_state, "dop_policy")
        row += 1
        components.label(
            frame, row, 0, "Multiplier",
            tooltip="Non-negative float scaling for DOP loss contribution. Typical starting range: 0.2 to 2.0. Higher values preserve base behavior more aggressively."
        )
        components.entry(frame, row, 1, self.ui_state, "dop_multiplier", extra_validate=check_range(lower=0))
        row += 1
        components.label(
            frame, row, 0, "Max DOP / base (0 = off)",
            tooltip="If > 0, caps weighted DOP loss per step at this multiple of base loss. Prevents preservation from overwhelming concept learning (trigger samples staying like the base model). Try 0.25–0.5 when DOP feels too strong; 0 disables capping."
        )
        components.entry(frame, row, 1, self.ui_state, "dop_max_weighted_to_base_ratio", extra_validate=check_range(lower=0))
        return row + 2

    def _build_schedule_group(self, frame, row: int) -> int:
        components.label(frame, row, 0, "Scheduling", tooltip="When DOP should run during training.")
        row += 1
        components.label(
            frame, row, 0, "Interval steps",
            tooltip="Integer >= 1. Used by periodic/manual policies. Example: 5 means run DOP every 5 global steps."
        )
        components.entry(frame, row, 1, self.ui_state, "dop_interval_steps", extra_validate=check_range(lower=1))
        row += 1
        components.label(
            frame, row, 0, "Start step",
            tooltip="Integer >= 0. DOP is disabled before this global step."
        )
        components.entry(frame, row, 1, self.ui_state, "dop_start_step", extra_validate=check_range(lower=0))
        row += 1
        components.label(
            frame, row, 0, "End step",
            tooltip="Use -1 for no end. Otherwise use integer >= 0 to stop DOP after that step."
        )
        components.entry(
            frame, row, 1, self.ui_state, "dop_end_step",
            extra_validate=check_non_negative_or_minus_one(),
        )
        row += 1
        components.label(
            frame, row, 0, "Adaptive strength",
            tooltip="Positive float controlling adaptive policy density. Higher values keep DOP active more often."
        )
        components.entry(frame, row, 1, self.ui_state, "dop_adaptive_strength", extra_validate=check_range(lower=0.1))
        return row + 2

    def _build_prompt_group(self, frame, row: int) -> int:
        components.label(frame, row, 0, "Prompt Replacement", tooltip="How trigger prompts are transformed for preservation pass.")
        row += 1
        components.label(
            frame, row, 0, "Trigger token",
            tooltip="The token/word to replace during DOP pass, e.g. your concept trigger."
        )
        components.entry(frame, row, 1, self.ui_state, "dop_trigger_token")
        row += 1
        components.label(
            frame, row, 0, "Class replacement",
            tooltip="Replacement text used when trigger is removed, e.g. 'person', 'style', or another broad class descriptor."
        )
        components.entry(frame, row, 1, self.ui_state, "dop_class_replacement")
        return row + 2

    def _build_safety_group(self, frame, row: int) -> int:
        components.label(frame, row, 0, "Safety / Matching", tooltip="Matching precision and fallback behavior.")
        row += 1
        components.label(
            frame, row, 0, "Word-boundary only",
            tooltip="If enabled, replace only whole-word trigger matches. Helps avoid accidental partial replacements."
        )
        components.switch(frame, row, 1, self.ui_state, "dop_word_boundary_only")
        row += 1
        components.label(
            frame, row, 0, "Case sensitive",
            tooltip="If enabled, replacement matches exact case only. If disabled, matching is case-insensitive."
        )
        components.switch(frame, row, 1, self.ui_state, "dop_case_sensitive")
        row += 1
        components.label(
            frame, row, 0, "Allow missing trigger",
            tooltip="If disabled, DOP pass is skipped when trigger is not found in the batch prompts. If enabled, DOP still runs."
        )
        components.switch(frame, row, 1, self.ui_state, "dop_allow_missing_trigger")
        return row + 2

    def _build_notes_group(self, frame, row: int):
        components.label(
            frame, row, 0, "Overhead Guide",
            tooltip="Estimated runtime overhead: quality is highest, balanced is medium, fast is lowest."
        )
        components.label(frame, row, 1, "Quality: high overhead | Balanced: medium | Fast: low")

    def _apply_dop_preset(self, preset: str):
        if preset == "manual":
            return
        dop_util.apply_preset(self.train_config, preset)
        self.ui_state.get_var("dop_enabled").set(self.train_config.dop_enabled)
        self.ui_state.get_var("dop_policy").set(self.train_config.dop_policy)
        self.ui_state.get_var("dop_multiplier").set(str(self.train_config.dop_multiplier))
        self.ui_state.get_var("dop_interval_steps").set(str(self.train_config.dop_interval_steps))
        self.ui_state.get_var("dop_start_step").set(str(self.train_config.dop_start_step))
        self.ui_state.get_var("dop_end_step").set(str(self.train_config.dop_end_step))
        self.ui_state.get_var("dop_adaptive_strength").set(str(self.train_config.dop_adaptive_strength))

    def _clear_warning_traces(self):
        for state, name, trace_id in self._trace_hooks:
            try:
                state.remove_var_trace(name, trace_id)
            except Exception:
                pass
        self._trace_hooks = []

    def _add_trace(self, state: UIState, name: str):
        trace_id = state.add_var_trace(name, self._update_text_encoder_warning)
        self._trace_hooks.append((state, name, trace_id))

    def _install_warning_traces(self):
        self._clear_warning_traces()
        self._add_trace(self.ui_state, "dop_enabled")
        self._add_trace(self.ui_state, "model_type")
        self._add_trace(self.ui_state.get_var("text_encoder"), "train")
        self._add_trace(self.ui_state.get_var("text_encoder_2"), "train")
        self._add_trace(self.ui_state.get_var("text_encoder_3"), "train")
        self._add_trace(self.ui_state.get_var("text_encoder_4"), "train")

    def _active_text_encoder_names(self) -> list[str]:
        model_type = self.train_config.model_type
        relevant = ["text_encoder"]
        if model_type.is_stable_diffusion_xl() or model_type.is_flux_1() or model_type.is_hunyuan_video():
            relevant = ["text_encoder", "text_encoder_2"]
        elif model_type.is_stable_diffusion_3():
            relevant = ["text_encoder", "text_encoder_2", "text_encoder_3"]
        elif model_type.is_hi_dream():
            relevant = ["text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4"]

        labels = {
            "text_encoder": "TE1",
            "text_encoder_2": "TE2",
            "text_encoder_3": "TE3",
            "text_encoder_4": "TE4",
        }
        active = []
        for key in relevant:
            part = getattr(self.train_config, key, None)
            if part is not None and getattr(part, "train", False):
                active.append(labels[key])
        return active

    def _update_text_encoder_warning(self):
        if self.warning_label is None:
            return
        if not self.train_config.dop_enabled:
            self.warning_label.configure(text="")
            return
        active = self._active_text_encoder_names()
        if not active:
            self.warning_label.configure(
                text=self._end_step_warning_text() if self._end_step_warning_text() else ""
            )
            return
        active_str = ", ".join(active)
        message = (
            f"Warning: DOP is enabled but text encoder training is active ({active_str}). "
            f"DOP is currently incompatible with text encoder training and trainer start will fail."
        )
        end_step_note = self._end_step_warning_text()
        if end_step_note:
            message = f"{message}\n{end_step_note}"
        self.warning_label.configure(text=message)

    def _end_step_warning_text(self) -> str:
        if not self.train_config.dop_enabled:
            return ""
        if int(self.train_config.dop_end_step) == 0 and int(self.train_config.dop_start_step) == 0:
            return (
                "Notice: DOP end step is set to 0. Legacy configs may treat this as 'no end' for ALWAYS_ON, "
                "but for clarity set end step to -1."
            )
        return ""
