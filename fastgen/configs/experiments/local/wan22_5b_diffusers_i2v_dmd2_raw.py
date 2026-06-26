# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

from fastgen.configs.data import VideoLoaderConfig
from fastgen.configs.discriminator import Discriminator_Wan22_5B_Config
import fastgen.configs.methods.config_dmd2 as config_dmd2_default
from fastgen.configs.net import Wan22_I2V_5B_Config


MODEL_DIR = "/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/Wan2.2-TI2V-5B-Diffusers"
# This must be a WebDataset shard directory containing .tar files. A plain
# unpacked mp4/txt directory is not supported by VideoWDSLoader.
RAW_WDS_DIR = "/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/FastGen/vipe_train_data/short_raw"


def create_config():
    config = config_dmd2_default.create_config()

    config.model.net = deepcopy(Wan22_I2V_5B_Config)
    config.model.net.model_id_or_local_path = MODEL_DIR
    config.model.fsdp_meta_init = True
    config.model.enable_preprocessors = True

    config.dataloader_train = deepcopy(VideoLoaderConfig)
    config.dataloader_train.datatags = [f"WDS:{RAW_WDS_DIR}"]
    config.dataloader_train.batch_size = 1
    config.dataloader_train.num_workers = 4
    config.dataloader_train.key_map = {"real": "mp4", "condition": "txt"}
    config.dataloader_train.presets_map = {"neg_condition": "neg_prompt_wan"}

    # Wan2.2 TI2V 5B uses 16x spatial VAE compression and 4x temporal compression.
    # [48, 31, 80, 44] corresponds to 704x1280, 121 frames.
    config.model.input_shape = [48, 31, 80, 44]
    config.dataloader_train.img_size = (config.model.input_shape[-1] * 16, config.model.input_shape[-2] * 16)
    config.dataloader_train.sequence_length = (config.model.input_shape[1] - 1) * 4 + 1

    config.model.precision = "bfloat16"
    config.model.net_optimizer.lr = 1e-5
    config.model.discriminator_optimizer.lr = 1e-5
    config.model.fake_score_optimizer.lr = 1e-5

    config.model.discriminator = deepcopy(Discriminator_Wan22_5B_Config)
    config.model.discriminator.disc_type = "multiscale_down_mlp_large"
    config.model.discriminator.feature_indices = [15, 22, 29]
    config.model.gan_loss_weight_gen = 0.03
    config.model.gan_use_same_t_noise = True
    config.model.fake_score_pred_type = "x0"
    config.model.student_sample_type = "ode"
    config.model.guidance_scale = 5.0

    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 5.0
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999
    config.model.student_sample_steps = 2
    config.model.sample_t_cfg.t_list = [0.999, 0.833, 0.0]

    config.trainer.fsdp = True
    config.trainer.ddp = False
    config.trainer.max_iter = 5001
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500
    config.trainer.validation_iter = 1_000_000
    config.trainer.seed = 1
    config.trainer.checkpointer.save_optimizer = True
    config.trainer.checkpointer.save_scheduler = True
    config.trainer.checkpointer.save_grad_scaler = True
    config.trainer.checkpointer.save_callbacks = True

    # Raw-domain training already keeps the VAE/text encoder online. Disable media
    # logging for the first stable run to avoid an extra full video sample/decode at iter 1.
    config.trainer.callbacks.pop("wandb", None)

    config.log_config.group = "wan22_5b_i2v_dmd2"
    config.log_config.name = "raw_704x1280_121f_2steps_1e-5lr"
    config.log_config.wandb_mode = "disabled"
    return config
