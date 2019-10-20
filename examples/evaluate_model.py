import platform
from truckms.evaluation.evaluate import download_data_if_not_exists, gen_cocoitem2datapoints, gen_cocoitem2framedp, \
    gen_cocoitem2targetdp, get_dataset, target_pred_iter_to_pandas, compute_iou_det_ann_df, compute_stats
from truckms.inference.neural import create_model_efficient, create_model, compute
from truckms.api import model_class_names
from functools import partial
from itertools import tee
import pickle
import os.path as osp
import pandas as pd
from pprint import pprint


def get_dataframes(datalake_path, pred_csv_path, target_csv_path):
    download_data_if_not_exists(datalake_path)
    coco_dset = get_dataset(datalake_path)

    g_tdp_fdp_1, g_tdp_fdp_2 = tee(gen_cocoitem2datapoints(coco_dset))
    g_tdp = gen_cocoitem2targetdp(g_tdp_fdp_1)
    g_fdp = gen_cocoitem2framedp(g_tdp_fdp_2)

    model = create_model_efficient(model_creation_func=partial(create_model, max_operating_res=800, conf_thr=0.05))
    g_pred = compute(g_fdp, model, batch_size=10, filter_classes=model_class_names)

    df_pred, df_target = target_pred_iter_to_pandas(g_tdp, g_pred)

    df_pred.to_csv(pred_csv_path)
    df_target.to_csv(target_csv_path)
    return df_pred, df_target


def compute_stats_for_labels(df_pred, df_target, stats_path):
    label_dataframes = compute_iou_det_ann_df(df_target, df_pred, ann_lbl_col='target.label', det_lbl_col='label')

    stats_label = {}
    for k in ['truck', 'bus', 'train']:
        joined_df = label_dataframes[k][0]
        ann_df = label_dataframes[k][1]
        det_df = label_dataframes[k][2]
        stats_label[k] = compute_stats(joined_df, ann_df, det_df)

    pickle.dump(stats_label, open(stats_path, 'wb'))
    return stats_label


def main():
    force_overwrite_detection = False
    force_overwrite_stats = False
    pred_csv_path, target_csv_path, stats_path = None, None, None
    df_target, df_pred = None, None
    if platform.system() == "Linux":
        datalake_path = r"/data1/workspaces/aiftimie/tms/tms_data"
        pred_csv_path = "/data1/workspaces/aiftimie/tms/tms_experiments/pandas_dataframes/coco_pred.csv"
        target_csv_path = "/data1/workspaces/aiftimie/tms/tms_experiments/pandas_dataframes/coco_target.csv"
        stats_path = "/data1/workspaces/aiftimie/tms/tms_experiments/pandas_dataframes/stats.pkl"
    else:
        datalake_path = r"D:\tms_data"

    if force_overwrite_detection or (not osp.exists(pred_csv_path) or not osp.exists(target_csv_path)):
        df_pred, df_target = get_dataframes(datalake_path, pred_csv_path, target_csv_path)
    else:
        df_pred = pd.read_csv(pred_csv_path)
        df_target = pd.read_csv(target_csv_path)

    if force_overwrite_stats or not osp.exists(stats_path):
        stats_label = compute_stats_for_labels(df_pred, df_target, stats_path)
    else:
        stats_label = pickle.load(open(stats_path, 'rb'))

    pprint (stats_label)
    print ()



if __name__ == "__main__":
    main()
