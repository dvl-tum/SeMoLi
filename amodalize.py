import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import hydra
import pandas as pd
from PseudoDetection3D.amodalizer import amodalize
import shutil


@hydra.main(config_path="PseudoDetection3D/conf", config_name="conf")   
def main(cfg):
    # add root dir
    cfg.registration.track_data_path = os.path.join(
            cfg.root_dir, cfg.registration.track_data_path)
    cfg.registration.out_path_for_eval = os.path.join(
            cfg.root_dir, cfg.registration.out_path_for_eval)
    
    _params_track = f'{cfg.registration.tracking.filter_by_width}_{cfg.registration.tracking.match_threshold}_{cfg.registration.tracking.len_thresh}_{cfg.registration.tracking.max_time}_{cfg.registration.tracking.fixed_time}_{cfg.registration.tracking.l_change_thresh}_{cfg.registration.tracking.w_change_thresh}_{cfg.registration.tracking.inact_patience}_{cfg.registration.tracking.use_temporal_weight}'

    _params_reg = f'{cfg.registration.registration.outlier_threshold}_{cfg.registration.registration.outlier_kNN}_{cfg.registration.registration.mode}_{cfg.registration.registration.density_thresh}_{cfg.registration.registration.concat}_{cfg.registration.registration.means_before}_{cfg.registration.registration.avg_w_prev}_{cfg.registration.registration.min_pts_thresh}_{cfg.registration.registration.registration_len_thresh}'
    
    cfg.registration.initial_dets_path = f'{cfg.registration.initial_dets_path}/{cfg.registration.model}'
    cfg.registration.tracked_dets_path = f'{cfg.registration.tracked_dets_path}/{cfg.registration.model}/{_params_track}'
    cfg.registration.registered_dets_path = f'{cfg.registration.registered_dets_path}/{cfg.registration.model}/{_params_reg}'
    
    amodalize(cfg)
    shutil.rmtree(f'{cfg.root_dir}/outputs') 

if __name__ == "__main__":
    cfg = main()
