import numpy as np
import warnings
warnings.filterwarnings("ignore")
import hydra
import pandas as pd
from .PseudoDetection3D.amodalizer import main


@hydra.main(config_path="conf", config_name="conf")   
def main(cfg):
    _params_track = f'{cfg.registration.filter_by_width}_\
        {cfg.registration.a_threshold}_\
            {cfg.registration.len_thresh}_\
                {cfg.registration.max_time_track}_\
                    {cfg.registration.fixed_time}_\
                        {cfg.registration.l_change_thresh}_\
                            {cfg.registration.w_change_thresh}_\
                                {cfg.registration.inact_patience}_\
                                    {cfg.registration.use_temporal_weight_track}'

    _params_reg = f'{cfg.registration.outlier_threshold}_\
        {cfg.registration.outlier_kNN}_\
            {cfg.registration.mode}_\
                {cfg.registration.density_thresh}_\
                    {cfg.registration.concat}_\
                        {cfg.registration.means_before}_\
                            {cfg.registration.avg_w_prev}_\
                                {cfg.registration.min_pts_thresh}_\
                                    {cfg.registration.registration_len_thresh}'
    
    cfg.registration.initial_dets = f'{cfg.registration.initial_dets}/{cfg.registration.model}'
    cfg.registration.tracked_dets = f'{cfg.registration.tracked_dets}/{cfg.registration.model}/{_params_track}'
    cfg.registration.registered_dets = f'{cfg.registration.registered_dets}/{cfg.registration.model}/{_params_reg}'
    
    main(cfg)
    

if __name__ == "__main__":
        main()
