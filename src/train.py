from typing import List, Optional

import hydra
import pytorch_lightning as pl
import pyrootutils
import torch
import os
import shutil
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import WandbLogger
from hydra.core.hydra_config import HydraConfig

from src import utils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = utils.get_pylogger(__name__)


# 预训练 baseline checkpoint 中不存在、允许随机初始化的参数
_NEW_PARAM_PREFIXES = ("mr_frontend.", "fusion_scale", "window_mid", "window_short", "window_long")


def load_pretrained_backbone(model: LightningModule, ckpt_path: str) -> None:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in missing if not k.startswith(_NEW_PARAM_PREFIXES)]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"预训练权重与模型不匹配\nmissing: {bad_missing}\nunexpected: {unexpected}"
        )
    log.info(f"Loaded pretrained backbone from {ckpt_path}; newly initialized: {sorted(set(k.split('.')[0] for k in missing))}")


@utils.task_wrapper
def train(cfg: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    try:
        if "seed" in cfg:
            # set seed for random number generators in pytorch, numpy and python.random
            if cfg.get("seed"):
                pl.seed_everything(cfg.seed, workers=True)

        else:
            raise ModuleNotFoundError

    except ModuleNotFoundError:
        print('[Error] seed should be fixed for reproducibility \n=> e.g. python train.py +seed=$SEED')
        exit(-1)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    # Init Lightning model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model.seed = cfg.get("seed")

    # 从预训练 DTTNet（单窗 baseline）加载主干，新加的多分辨率前端保持随机初始化
    if cfg.get("pretrained_ckpt"):
        load_pretrained_backbone(model, cfg.pretrained_ckpt)

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    if "resume_from_checkpoint" in cfg.trainer:
        ckpt_path = cfg.trainer.resume_from_checkpoint
        # get the parent directory of the checkpoint path
        log_dir = os.path.dirname(os.path.dirname(ckpt_path))
        tensorboard_dir = os.path.join(log_dir, "tensorboard")
        if os.path.exists(tensorboard_dir):
            # copy tensorboard dir to the parent directory of the checkpoint path
            # HydraConfig.get().run.dir returns new dir so do not use it! (now fixed)
            shutil.copytree(tensorboard_dir,os.path.join(os.getcwd(),"tensorboard"))

        wandb_dir = os.path.join(log_dir, "wandb")
        if os.path.exists(wandb_dir):
            shutil.copytree(wandb_dir,os.path.join(os.getcwd(),"wandb"))


    logger: List = []
    if "logger" in cfg:
        for _, lg_conf in cfg["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

        for wandb_logger in [l for l in logger if isinstance(l, WandbLogger)]:
            utils.wandb_login(key=cfg.wandb_api_key)
            # utils.wandb_watch_all(wandb_logger, model) # TODO buggy
            break

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    # get env variable use_gloo
    use_gloo = os.environ.get("USE_GLOO", False)
    if use_gloo:
        from pytorch_lightning.strategies import DDPStrategy
        ddp = DDPStrategy(process_group_backend='gloo')
        trainer: Trainer = hydra.utils.instantiate(
            cfg.trainer, strategy=ddp, callbacks=callbacks, logger=logger, _convert_="partial"
        )
    else:
        trainer: Trainer = hydra.utils.instantiate(
            cfg.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
        )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        dict(
            cfg=cfg,
            model=model,
            datamodule=datamodule,
            trainer=trainer,
            callbacks=callbacks,
            logger=logger,
        )
    )

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set after training
    # if not cfg.trainer.get("fast_dev_run"):
    #     log.info("Starting testing!")
    #     trainer.test()

    # Make sure everything closed properly
    log.info("Finalizing!")
    # utils.finish(
    #     config=cfg,
    #     model=model,
    #     datamodule=datamodule,
    #     trainer=trainer,
    #     callbacks=callbacks,
    #     logger=logger,
    # )

    # Print path to best checkpoint
    # log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    # optimized_metric = cfg.get("optimized_metric")
    # if optimized_metric:
    #     return trainer.callback_metrics[optimized_metric]
    return None, None
