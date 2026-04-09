"""
Callback for multi-benchmark validation with MMD calculations.

This callback calls the existing validate_epoch_multi_benchmark function
to preserve exact MMD calculation logic.
"""

import pytorch_lightning as pl
from typing import Dict, Any, Iterable
from models.CATs_PlusPlus.utils_training.optimize_multi import validate_epoch_multi_benchmark


class MMDValidationCallback(pl.Callback):
    """
    Callback that performs multi-benchmark validation with MMD calculations
    and supports two logging modes:
      - epoch validation (normal Lightning validation loop)
      - step validation at configured global train step milestones
    """

    def __init__(self, config: Dict[str, Any], multi_evaluator: Any):
        """
        Initialize callback.

        Args:
            config: Full training config dict
            multi_evaluator: MultiBenchmarkEvaluator instance
        """
        super().__init__()
        self.config = config
        self.multi_evaluator = multi_evaluator
        self.training_config = config['training']
        self.eval_config = config['evaluation']
        self.validation_step_marks = sorted(
            self._coerce_step_marks(
                self.training_config.get('validation_step_marks', [])
            )
        )
        self.validation_step_interval = self._coerce_step_interval(
            self.training_config.get('validation_step_interval', None)
        )
        self._validated_step_marks = set()

    @staticmethod
    def _coerce_step_marks(raw_steps: Any) -> Iterable[int]:
        """Normalize config value for validation step milestones."""
        if raw_steps is None:
            return []
        if isinstance(raw_steps, (int, float)):
            raw_steps = [raw_steps]
        elif isinstance(raw_steps, str):
            raw_steps = [raw_steps]

        steps = []
        for entry in raw_steps:
            if isinstance(entry, (int, float)):
                candidate = int(entry)
                if candidate > 0:
                    steps.append(candidate)
                continue
            if isinstance(entry, str):
                for piece in entry.split(','):
                    piece = piece.strip()
                    if not piece:
                        continue
                    candidate = int(float(piece))
                    if candidate > 0:
                        steps.append(candidate)
                continue
            raise TypeError(f'Unsupported validation_step_marks entry: {entry!r}')
        return set(steps)

    @staticmethod
    def _coerce_step_interval(raw_interval: Any) -> int:
        """Normalize periodic validation interval (in train steps)."""
        if raw_interval is None:
            return 0
        if isinstance(raw_interval, str):
            raw_interval = raw_interval.strip()
            if raw_interval == "":
                return 0
            candidate = int(float(raw_interval))
        elif isinstance(raw_interval, (int, float)):
            candidate = int(raw_interval)
        else:
            raise TypeError(f'Unsupported validation_step_interval value: {raw_interval!r}')
        return candidate if candidate > 0 else 0

    def _run_validation(self, trainer: pl.Trainer, pl_module: pl.LightningModule, scope: str, step: int = None):
        datamodule = trainer.datamodule
        if not hasattr(datamodule, 'get_val_dataloaders'):
            raise RuntimeError("DataModule must have get_val_dataloaders() method")

        val_dataloaders = datamodule.get_val_dataloaders()
        if not val_dataloaders:
            raise RuntimeError("No validation dataloaders found")

        mmd_every_n_epochs = self.training_config.get('mmd_every_n_epochs', 0)

        use_motion_aware = self.eval_config.get('use_motion_aware', True)
        min_motion_pixels = self.eval_config.get('min_motion_pixels', 5.0)
        zero_threshold = self.eval_config.get('zero_threshold', 0.5)

        if step is None:
            step = pl_module.get_cumulative_training_steps()

        epoch_for_eval = -1 if scope == 'initial' else trainer.current_epoch

        context = {
            'validation_scope': scope,
            'validation_epoch': 0 if scope == 'initial' else trainer.current_epoch + 1,
            'validation_step': step,
            # For step-scoped validation, target should be the triggering global step.
            # CSV logging deduplicates on validation_target, so this must be step-specific.
            'validation_target': (
                0 if scope == 'initial' else (step if scope == 'step' else trainer.current_epoch + 1)
            ),
        }

        # Step/initial validation runs outside Lightning's normal validation loop.
        # Use Lightning's validation-mode hooks so the whole module enters the same
        # eval context as epoch validation instead of only toggling pl_module.model.
        manage_validation_mode = scope in {'step', 'initial'}
        was_training = bool(pl_module.training)
        try:
            if manage_validation_mode:
                pl_module.on_validation_model_eval()
            val_results = validate_epoch_multi_benchmark(
                net=pl_module.model,
                val_loaders=val_dataloaders,
                device=pl_module.device,
                epoch=epoch_for_eval,
                multi_evaluator=self.multi_evaluator,
                primary_benchmark=self.eval_config['eval_benchmarks'][0],
                use_motion_aware=use_motion_aware,
                min_motion_pixels=min_motion_pixels,
                zero_threshold=zero_threshold,
                mmd_every_n_epochs=mmd_every_n_epochs
            )
        finally:
            if manage_validation_mode:
                if was_training:
                    pl_module.on_validation_model_train()
                else:
                    pl_module.eval()

        pl_module.set_val_context(context)
        pl_module.set_val_results(val_results)
        if getattr(trainer, 'is_global_zero', True):
            self._log_tensorboard(trainer, val_results, context)
            self._log_console(val_results, context)

    def _log_tensorboard(self, trainer: pl.Trainer, val_results: Dict[str, Any], context: Dict[str, Any]):
        scope = context.get('validation_scope', 'epoch')
        if scope == 'step':
            tag_prefix = 'val_step'
            log_step = int(context.get('validation_step', 0))
        elif scope == 'initial':
            tag_prefix = 'val_initial'
            log_step = -1
        else:
            tag_prefix = 'val'
            log_step = int(trainer.current_epoch)

        for benchmark, results in val_results.items():
            trainer.logger.experiment.add_scalar(f'{tag_prefix}/{benchmark}/PCK', results['pck'], log_step)
            trainer.logger.experiment.add_scalar(f'{tag_prefix}/{benchmark}/loss', results['loss'], log_step)

            if 'pck_motion_aware' in results:
                trainer.logger.experiment.add_scalar(
                    f'{tag_prefix}/{benchmark}/PCK_motion_aware',
                    results['pck_motion_aware'],
                    log_step
                )

            if 'motion_binned' in results:
                motion_binned = results['motion_binned']
                for bin_name, bin_data in motion_binned.items():
                    if bin_data.get('count', 0) > 0:
                        trainer.logger.experiment.add_scalar(
                            f'{tag_prefix}/{benchmark}/PCK_motion_{bin_name}',
                            bin_data['mean_pck'],
                            log_step
                        )
                        trainer.logger.experiment.add_scalar(
                            f'{tag_prefix}/{benchmark}/motion_{bin_name}_count',
                            bin_data['count'],
                            log_step
                        )

            if 'zero_flow_metrics' in results:
                zfm = results['zero_flow_metrics']
                trainer.logger.experiment.add_scalar(
                    f'{tag_prefix}/{benchmark}/zero_flow_precision',
                    zfm.get('zero_precision', 0),
                    log_step
                )
                trainer.logger.experiment.add_scalar(
                    f'{tag_prefix}/{benchmark}/zero_flow_recall',
                    zfm.get('zero_recall', 0),
                    log_step
                )
                trainer.logger.experiment.add_scalar(
                    f'{tag_prefix}/{benchmark}/zero_flow_f1',
                    zfm.get('zero_f1', 0),
                    log_step
                )
                trainer.logger.experiment.add_scalar(
                    f'{tag_prefix}/{benchmark}/static_bias_ratio',
                    zfm.get('static_bias_ratio', 0),
                    log_step
                )

            if 'mmd2_pred_corr_vs_pred_miss' in results:
                mmd_val = results['mmd2_pred_corr_vs_pred_miss']
                if isinstance(mmd_val, (int, float)) and mmd_val == mmd_val:
                    trainer.logger.experiment.add_scalar(
                        f'{tag_prefix}/{benchmark}/MMD2_pred_corr_vs_pred_miss',
                        mmd_val,
                        log_step
                    )

            if 'mmd2_pred_corr_vs_gt' in results:
                mmd_val = results['mmd2_pred_corr_vs_gt']
                if isinstance(mmd_val, (int, float)) and mmd_val == mmd_val:
                    trainer.logger.experiment.add_scalar(
                        f'{tag_prefix}/{benchmark}/MMD2_pred_corr_vs_gt',
                        mmd_val,
                        log_step
                    )

            if 'mmd2_pred_miss_vs_gt' in results:
                mmd_val = results['mmd2_pred_miss_vs_gt']
                if isinstance(mmd_val, (int, float)) and mmd_val == mmd_val:
                    trainer.logger.experiment.add_scalar(
                        f'{tag_prefix}/{benchmark}/MMD2_pred_miss_vs_gt',
                        mmd_val,
                        log_step
                    )

            if benchmark == 'tss' and 'pck_by_category' in results:
                for cat, pck in results['pck_by_category'].items():
                    trainer.logger.experiment.add_scalar(
                        f'{tag_prefix}/{benchmark}/{cat}/PCK',
                        pck,
                        log_step
                    )

        avg_pck_scores = [r['pck'] for r in val_results.values()]
        avg_pck = sum(avg_pck_scores) / len(avg_pck_scores) if avg_pck_scores else 0.0
        trainer.logger.experiment.add_scalar(f'{tag_prefix}/average/PCK', avg_pck, log_step)

    @staticmethod
    def _log_console(val_results: Dict[str, Any], context: Dict[str, Any]):
        scope = context.get('validation_scope', 'epoch')
        target = context.get('validation_target', 'N/A')

        if scope == 'step':
            print(f"\nValidation Results (step={target}):")
        elif scope == 'initial':
            print("\nInitial validation results:")
        else:
            print(f"\nValidation Results (epoch={context.get('validation_epoch', 'N/A')}):")

        for benchmark, results in val_results.items():
            print(f"  {benchmark}: PCK={results['pck']:.2f}%, Loss={results['loss']:.4f}")

            if 'mmd2_pred_corr_vs_pred_miss' in results:
                mmd_val = results['mmd2_pred_corr_vs_pred_miss']
                if isinstance(mmd_val, (int, float)) and mmd_val == mmd_val:
                    print(f"    MMD^2 (pred_corr vs pred_miss): {mmd_val:.6f}")
            if 'mmd2_pred_corr_vs_gt' in results:
                mmd_val = results['mmd2_pred_corr_vs_gt']
                if isinstance(mmd_val, (int, float)) and mmd_val == mmd_val:
                    print(f"    MMD^2 (pred_corr vs gt): {mmd_val:.6f}")
            if 'mmd2_pred_miss_vs_gt' in results:
                mmd_val = results['mmd2_pred_miss_vs_gt']
                if isinstance(mmd_val, (int, float)) and mmd_val == mmd_val:
                    print(f"    MMD^2 (pred_miss vs gt): {mmd_val:.6f}")

        avg_pck_scores = [r['pck'] for r in val_results.values()]
        avg_pck = sum(avg_pck_scores) / len(avg_pck_scores) if avg_pck_scores else 0.0
        print(f"  Average PCK: {avg_pck:.2f}%")

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Optional initial validation before any optimizer step."""
        if not self.training_config.get('eval_initial', False):
            return
        # On resume, global_step is non-zero; skip "initial" eval to avoid misleading
        # step-0 rows that are actually from a loaded checkpoint state.
        current_step = int(getattr(trainer, 'global_step', 0))
        if current_step > 0:
            print(f"Skipping initial validation on resume (global_step={current_step})")
            return

        self._run_validation(trainer, pl_module, scope='initial', step=0)

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called at start of each validation pass."""
        self._run_validation(trainer, pl_module, scope='epoch')

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0
    ):
        """Optionally run extra validation at configured cumulative train step milestones."""
        if not self.validation_step_marks and self.validation_step_interval <= 0:
            return

        current_step = int(getattr(trainer, 'global_step', pl_module.get_cumulative_training_steps()))

        if current_step in self._validated_step_marks:
            return

        should_validate = False
        if self.validation_step_marks and current_step in self.validation_step_marks:
            should_validate = True
        if self.validation_step_interval > 0 and current_step > 0:
            if current_step % self.validation_step_interval == 0:
                should_validate = True
        if not should_validate:
            return

        self._validated_step_marks.add(current_step)
        self._run_validation(trainer, pl_module, scope='step', step=current_step)
