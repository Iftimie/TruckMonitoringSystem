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
import numpy as np
import os
import matplotlib
if platform.system() != 'Linux':
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mock
from torchvision.datasets import CocoDetection
import time
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import GPUtil


def get_dataframes(datalake_path, pred_csv_path, target_csv_path, max_operating_res):
    download_data_if_not_exists(datalake_path)
    coco_dset = get_dataset(datalake_path)

    g_tdp_fdp_1, g_tdp_fdp_2 = tee(gen_cocoitem2datapoints(coco_dset))
    g_tdp = gen_cocoitem2targetdp(g_tdp_fdp_1)
    g_fdp = gen_cocoitem2framedp(g_tdp_fdp_2)

    model = create_model_efficient(model_creation_func=partial(create_model, max_operating_res=max_operating_res, conf_thr=0.05))
    g_pred = compute(g_fdp, model, batch_size=10, filter_classes=model_class_names)

    df_pred, df_target = target_pred_iter_to_pandas(g_tdp, g_pred)

    df_pred.to_csv(pred_csv_path)
    df_target.to_csv(target_csv_path)
    return df_pred, df_target


def compute_stats_for_labels(label_dataframes, stats_path):
    stats = {}
    for k in ['truck', 'bus', 'train']:
        joined_df = label_dataframes[k][0]
        ann_df = label_dataframes[k][1]
        det_df = label_dataframes[k][2]
        stats_for_label = {}
        for thr in np.linspace(0.05, 0.95, 19):
            joined_df_thr = joined_df[joined_df['score'] >= thr]
            ann_df_thr = ann_df
            det_df_thr = det_df[det_df['score'] >= thr]
            stats_for_label[thr] = compute_stats(joined_df_thr, ann_df_thr, det_df_thr)
        stats[k] = pd.DataFrame(stats_for_label).T

    pickle.dump(stats, open(stats_path, 'wb'))
    return stats


def plot_prec_rec_curves(stats_label, prec_rec_curve_path):
    def label_point_orig(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x'], point['y'], "%.4f" % (point['val'],))

    for k in stats_label:
        fig = plt.figure(figsize=(5, 5))
        ax = stats_label[k].plot(x='recall/DR', y='precision', legend=False)
        ax.set_ylabel('precision')
        ax.set_title('recall-precision curve '+ k)
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        label_point_orig(stats_label[k]['recall/DR'], stats_label[k]['precision'], stats_label[k].index.to_series(), ax)

        plt.savefig(prec_rec_curve_path+k+".png")
        plt.close('all')

@mock.patch.object(CocoDetection, "__len__")
def evaluate_speed(mock_some_obj_some_method, datalake_path, max_operating_res):
    num_eval_frames = 200
    mock_some_obj_some_method.return_value = num_eval_frames

    download_data_if_not_exists(datalake_path)
    coco_dset = get_dataset(datalake_path)

    g_tdp_fdp_1, g_tdp_fdp_2 = tee(gen_cocoitem2datapoints(coco_dset))
    g_tdp = gen_cocoitem2targetdp(g_tdp_fdp_1)
    g_fdp = gen_cocoitem2framedp(g_tdp_fdp_2)

    model = create_model_efficient(
        model_creation_func=partial(create_model, max_operating_res=max_operating_res, conf_thr=0.05))
    g_pred = compute(g_fdp, model, batch_size=10, filter_classes=model_class_names)
    start = time.time()
    df_pred, df_target = target_pred_iter_to_pandas(g_tdp, g_pred)
    end = time.time()
    return (end - start) / num_eval_frames


def plot_pareto_plane(pareto_coordinates):
    label = 'truck'

    X, Y, Z = [], [], []
    GPUs = set()
    for res in pareto_coordinates:
        z_res = pareto_coordinates[res][label]['precision'].tolist()
        x_res = pareto_coordinates[res][label]['speed'].tolist()
        y_res = pareto_coordinates[res][label]['recall/DR'].tolist()
        X.append(x_res)
        Y.append(y_res)
        Z.append(z_res)
        GPUs |= set(pareto_coordinates[res][label]['GPUs'].tolist())
    GPUs = ' '.join(set(g for g in GPUs))
    X, Y, Z = np.array(X), np.array(Y), np.array(Z)

    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('X speed in S '+GPUs.replace('GeForce', '').replace('GTX',''))
    ax.set_ylabel('Y recall/DR')
    ax.set_zlabel('Z precision')
    ax.set_title('pareto surface for class '+label)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.invert_xaxis()
    ax.view_init(elev=26., azim=21)
    # unintentionally I put the color of the surface right

    mres = max(pareto_coordinates.keys())
    z_mres = pareto_coordinates[res][label]['precision'].tolist()
    x_mres = pareto_coordinates[res][label]['speed'].tolist()
    y_mres = pareto_coordinates[res][label]['recall/DR'].tolist()
    t_mres = pareto_coordinates[res][label].index.tolist()
    for t, x, y, z in zip(t_mres, x_mres, y_mres, z_mres):
        label = '%.2f' % (t, )
        ax.text(x, y, z, label, fontsize=7)
    ax.text(x, y + 0.1, z + 0.1, 'model confidence threshold', fontsize=7)

    plt.show()
    pass


def main():
    intermediate_dir, datalake_path = None, None
    force_overwrite_detection = False
    force_overwrite_stats = False
    force_compute_iou = False
    force_speed_performance_eval = False
    force_pareto_generation = False

    if platform.system() == "Linux":
        datalake_path = r"/data1/workspaces/aiftimie/tms/tms_data"
        dumping_dir = r'/data1/workspaces/aiftimie/tms/tms_experiments/pandas_dataframes/'
    else:
        dumping_dir = r'D:\tms_data\cached_stuff'

    if force_pareto_generation or not osp.exists(osp.join(dumping_dir, "pareto_coordinates.pkl")):
        pareto_coordinates = {}
        for max_operating_res in [320, 400, 480, 560, 620, 700, 780, 860, 920, 1000, 1080]:
            intermediate_dir = osp.join(dumping_dir, str(max_operating_res))
            if not osp.exists(intermediate_dir):
                os.mkdir(intermediate_dir)
            pred_csv_path = osp.join(intermediate_dir, "coco_pred.csv")
            target_csv_path = osp.join(intermediate_dir, "coco_target.csv")
            stats_path = osp.join(intermediate_dir, "stats.pkl")
            iou_dataframes_path = osp.join(intermediate_dir, "iou_dataframes.pkl")
            prec_rec_curve_path = osp.join(intermediate_dir, "prec_rec_curve_")
            speed_performance_path = osp.join(intermediate_dir, "speed_performace.pkl")

            if force_speed_performance_eval or (not osp.exists(speed_performance_path)):
                GPUs = ' '.join(set(g.name for g in GPUtil.getGPUs()))
                speed_ = evaluate_speed(datalake_path=datalake_path, max_operating_res=max_operating_res)
                pickle.dump((GPUs, speed_), open(speed_performance_path, 'wb'))
            else:
                (GPUs, speed_) = pickle.load(open(speed_performance_path, 'rb'))

            if force_overwrite_detection or (not osp.exists(pred_csv_path) or not osp.exists(target_csv_path)):
                df_pred, df_target = get_dataframes(datalake_path, pred_csv_path, target_csv_path, max_operating_res)
            else:
                df_pred = pd.read_csv(pred_csv_path)
                df_target = pd.read_csv(target_csv_path)

            if force_compute_iou or not osp.exists(iou_dataframes_path):
                label_dataframes = compute_iou_det_ann_df(df_target, df_pred, ann_lbl_col='target.label', det_lbl_col='label')
                pickle.dump(label_dataframes, open(iou_dataframes_path, 'wb'))
            else:
                label_dataframes = pickle.load(open(iou_dataframes_path, 'rb'))
            if force_overwrite_stats or not osp.exists(stats_path):
                stats_label = compute_stats_for_labels(label_dataframes, stats_path)
            else:
                stats_label = pickle.load(open(stats_path, 'rb'))

            plot_prec_rec_curves(stats_label, prec_rec_curve_path)
            for l in stats_label:
                stats_label[l]['speed'] = speed_
                stats_label[l]['max_operating_res'] = max_operating_res
                stats_label[l]['GPUs'] = GPUs
            pareto_coordinates[max_operating_res] = stats_label

        pickle.dump(pareto_coordinates, open(osp.join(dumping_dir, "pareto_coordinates.pkl"), 'wb'))
    else:
        pareto_coordinates = pickle.load(open(osp.join(dumping_dir, "pareto_coordinates.pkl"), 'rb'))

    plot_pareto_plane(pareto_coordinates)

if __name__ == "__main__":
    main()
