import pickle 
from pathlib import Path 
import fire 
import shutil 
import numpy as np
import os
import paddle
import time 
from configs import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from data.preprocess import merge_second_batch
import core 
import models 
import libs 
from libs.tools.progress_bar import ProgressBar
import data.kitti_common as kitti
from libs.tools.eval import get_official_eval_result, get_coco_eval_result
import traceback
# from pyinstrument import Profiler

def eval(cfg_file = None,
          model_dir = None,
          result_path=None,
          create_folder=False,
          display_step=1,
          summary_step=5,
          pickle_result=False):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_checkpoint_dir = model_dir / 'eval_checkpoints'
    eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if result_path is None:
        result_path = model_dir / 'results'
    config_file_bkp = "pipeline.config"
    shutil.copyfile(cfg_file, str(model_dir / config_file_bkp))

    config = cfg_from_yaml_file(cfg_file, cfg)
    input_cfg = config.TRAIN_INPUT_READER
    eval_input_cfg = config.EVAL_INPUT_READER
    model_cfg = config.MODEL
    train_cfg = config.TRAIN_CONFIG
    class_names = config.CLASS_NAMES 
    ######################
    # BUILD VOXEL GENERATOR
    ######################
    voxel_generator = core.build_voxel_generator(config.VOXEL_GENERATOR)
    ######################
    # BUILD TARGET ASSIGNER
    ######################
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = core.build_box_coder(config.BOX_CODER)
    target_assigner_cfg = config.TARGET_ASSIGNER
    target_assigner = core.build_target_assigner(target_assigner_cfg,
                                                    bv_range, box_coder)
    ######################
    # BUILD NET
    ######################
    net = models.build_network(model_cfg, voxel_generator, target_assigner)

    ######################
    # RESUME
    ######################
    # we need global_step to create lr_scheduler, so restore net first.
    # libs.tools.try_restore_latest_checkpoints(eval_checkpoint_dir, [net])
    if os.path.exists(eval_checkpoint_dir):
        resume_model = os.path.normpath(eval_checkpoint_dir)
        ckpt_path = os.path.join(eval_checkpoint_dir, 'voxelnet.pdparams')
        para_state_dict = paddle.load(ckpt_path)
        net.set_state_dict(para_state_dict)
        print("++++++++++++++++++++++++++++++++LOAD voxelnet.pdparams SUCCESS!!!!!+++++++++++++++++++++++++++++")

    if train_cfg.ENABLE_MIXED_PRECISION:
        net.half()
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    # lr_scheduler = core.build_lr_schedules(optimizer_cfg, optimizer, gstep)
    if train_cfg.ENABLE_MIXED_PRECISION:
        float_dtype = paddle.float16
    else:
        float_dtype = paddle.float32
    ######################
    # PREPARE INPUT
    ######################
    eval_dataset = core.build_input_reader(input_cfg,
                                           model_cfg,
                                           training= False,
                                           voxel_generator=voxel_generator,
                                           target_assigner=target_assigner)

    def _worker_init_fn(worker_id):
        time_seed = np.array(time.time(),dtype=np.int32)
        np.random.seed(time_seed + worker_id)
        print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])

    # print("++++++++++++++++++++++++++++++++++++START LOADER++++++++++++++++++++++++++++++++++++++++++++++++")
    eval_dataloader = paddle.io.DataLoader(
        dataset=eval_dataset,
        batch_size = eval_input_cfg.BATCH_SIZE,
        shuffle=False,
        use_shared_memory=False,
        num_workers=eval_input_cfg.NUM_WORKERS,
        collate_fn=merge_second_batch)

    ######################
    # EVAL
    ######################
    log_path = model_dir / 'eval_log.txt'
    logf = open(log_path, 'a')
    logf.write("\n")
    print("++++++++++++++++++++++++++++++++++++EVAL PREPARE++++++++++++++++++++++++++++++++++++++++++++++++")
    try:
        print("++++++++++++++++++++++++++++++++++++START EVAL++++++++++++++++++++++++++++++++++++++++++++++++", file=logf)
        net.eval()
        center_limit_range = model_cfg.POST_PROCESSING.post_center_limit_range
        result_path_step = result_path / f"eval"
        result_path_step.mkdir(parents=True, exist_ok=True)
        # print("++++++++++++++++++++++++++++++++++++OVER EVAL++++++++++++++++++++++++++++++++++++++++++++++++")
        t = time.time()
        dt_annos = []
        print("**********************************************************************************")
        prog_bar = ProgressBar()
        print("**********************************************************************************")
        prog_bar.start(len(eval_dataset) // eval_input_cfg.BATCH_SIZE + 1)
        print("**********************************************************************************")
        for i,example in enumerate(eval_dataloader()):

            # profiler = Profiler()
            # profiler.start()

            example = example_convert_to_paddle(example, float_dtype)
            if pickle_result:
                    dt_annos += predict_kitti_to_anno(
                        net, example, class_names, center_limit_range,
                        model_cfg.LIDAR_INPUT)
            else:
                try:
                    _predict_kitti_to_file(net, example, result_path_step,
                                                class_names, center_limit_range,
                                                model_cfg.LIDAR_INPUT)
                except Exception as e:
                    traceback.print_exc()
                    
            prog_bar.print_bar()

            # profiler.stop()
            # print(profiler.output_text(unicode=True, color=True))

        sec_per_ex = len(eval_dataset) / (time.time() - t)
        # print(f"avg forward time per example: {net.avg_forward_time:.3f}")
        # print(f"avg postprocess time per example: {net.avg_postprocess_time:.3f}")
        print(f'generate label finished({sec_per_ex:.2f}/s). start eval:')
        print(f'generate label finished({sec_per_ex:.2f}/s). start eval:', file=logf)
        gt_annos = [info["annos"] for info in eval_dataset.dataset.kitti_infos]
        print(gt_annos)
        if not pickle_result:
            dt_annos = kitti.get_label_annos(result_path_step)
            result, mAPbbox, mAPbev, mAP3d, mAPaos = get_official_eval_result(gt_annos, dt_annos, class_names,
                                                                                return_data=True)
            print(result, file=logf)
            print(result)
        print("**********************************************************************************")

        result = get_coco_eval_result(gt_annos, dt_annos, class_names)
        print(result, file=logf)
        print(result)
        
    except Exception as e:
        traceback.print_exc()


def example_convert_to_paddle(example, dtype=paddle.float32) -> dict:
    example_paddle = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "rect",
        "Trv2c", "P2"
    ]

    for k, v in example.items():
        if k in float_names:
            example_paddle[k] = paddle.to_tensor(v, dtype=dtype)
        elif k in ["coordinates", "labels", "num_points"]:
            example_paddle[k] = paddle.to_tensor(
                v, dtype=paddle.int32)
        elif k in ["anchors_mask"]:
            example_paddle[k] = paddle.to_tensor(
                v, dtype=paddle.bool)
        else:
            example_paddle[k] = v

    return example_paddle

def _flat_nested_json_dict(json_dict, flatted, sep=".", start=""):
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, start + sep + k)
        else:
            flatted[start + sep + k] = v


def flat_nested_json_dict(json_dict, sep=".") -> dict:
    """flat a nested json-like dict. this function make shadow copy.
    """
    flatted = {}
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, k)
        else:
            flatted[k] = v
    return flatted


def _predict_kitti_to_file(net,
                           example,
                           result_save_path,
                           class_names,
                           center_limit_range=None,
                           lidar_input=False):
    batch_image_shape = example['image_shape']
    batch_imgidx = example['image_idx']
    predictions_dicts = net(example)
    # t = time.time()
    for i, preds_dict in enumerate(predictions_dicts):
        image_shape = batch_image_shape[i].cpu().numpy()
        img_idx = preds_dict["image_idx"]
        if preds_dict["bbox"] is not None:
            box_2d_preds = preds_dict["bbox"].cpu().numpy()
            # print(box_2d_preds)
            box_preds = preds_dict["box3d_camera"].cpu().numpy()
            scores = preds_dict["scores"].cpu().numpy()
            box_preds_lidar = preds_dict["box3d_lidar"]
            if(len(box_preds_lidar.shape)==1):
                box_preds_lidar = box_preds_lidar.unsqueeze(0)
            box_preds_lidar = box_preds_lidar.cpu().numpy()
            # write pred to file
            box_preds = box_preds[:, [0, 1, 2, 4, 5, 3,
                                      6]]  # lhw->hwl(label file format)
            label_preds = preds_dict["label_preds"].cpu().numpy()
            # label_preds = np.zeros([box_2d_preds.shape[0]], dtype=np.int32)
            result_lines = []
            for box, box_lidar, bbox, score, label in zip(
                    box_preds, box_preds_lidar, box_2d_preds, scores,
                    label_preds):
                if not lidar_input:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                        continue
                    if bbox[2] < 0 or bbox[3] < 0:
                        continue
                # print(img_shape)
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if (np.any(box_lidar[:3] < limit_range[:3])
                            or np.any(box_lidar[:3] > limit_range[3:])):
                        continue

                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])# TODO
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                result_dict = {
                    'name': class_names[int(label)],
                    'alpha': -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6],
                    'bbox': bbox,
                    'location': box[:3],
                    'dimensions': box[3:6],
                    'rotation_y': box[6],
                    'score': score,
                }
                result_line = kitti.kitti_result_line(result_dict)
                result_lines.append(result_line)
        else:
            result_lines = []
        result_file = f"{result_save_path}/{img_idx.numpy()[0]:06d}.txt"
        result_str = '\n'.join(result_lines)
        with open(result_file, 'w') as f:
            f.write(result_str)


def predict_kitti_to_anno(net,
                          example,
                          class_names,
                          center_limit_range=None,
                          lidar_input=False,
                          global_set=None):
    batch_image_shape = example['image_shape']
    batch_imgidx = example['image_idx']
    predictions_dicts = net(example)
    # t = time.time()
    annos = []
    for i, preds_dict in enumerate(predictions_dicts):
        image_shape = batch_image_shape[i].cpu().numpy()
        img_idx = preds_dict["image_idx"]
        if preds_dict["bbox"] is not None:
            box_2d_preds = preds_dict["bbox"].detach().cpu().numpy()
            box_preds = preds_dict["box3d_camera"].detach().cpu().numpy()
            scores = preds_dict["scores"].detach().cpu().numpy()
            box_preds_lidar = preds_dict["box3d_lidar"]
            if(len(box_preds_lidar.shape)==1):
                box_preds_lidar = box_preds_lidar.unsqueeze(0)
            box_preds_lidar = box_preds_lidar.cpu().numpy()
            # write pred to file
            label_preds = preds_dict["label_preds"].detach().cpu().numpy()
            # label_preds = np.zeros([box_2d_preds.shape[0]], dtype=np.int32)
            anno = kitti.get_start_result_anno()
            num_example = 0
            for box, box_lidar, bbox, score, label in zip(
                    box_preds, box_preds_lidar, box_2d_preds, scores,
                    label_preds):
                if not lidar_input:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                        continue
                    if bbox[2] < 0 or bbox[3] < 0:
                        continue
                # print(img_shape)
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if (np.any(box_lidar[:3] < limit_range[:3])
                            or np.any(box_lidar[:3] > limit_range[3:])):
                        continue
                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                anno["name"].append(class_names[int(label)])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["alpha"].append(-np.arctan2(-box_lidar[1], box_lidar[0]) +
                                     box[6])
                anno["bbox"].append(bbox)
                anno["dimensions"].append(box[3:6])
                anno["location"].append(box[:3])
                anno["rotation_y"].append(box[6])
                if global_set is not None:
                    for i in range(100000):
                        if score in global_set:
                            score -= 1 / 100000
                        else:
                            global_set.add(score)
                            break
                anno["score"].append(score)

                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(kitti.empty_result_anno())
        else:
            annos.append(kitti.empty_result_anno())
        num_example = annos[-1]["name"].shape[0]
        annos[-1]["image_idx"] = np.array(
            [img_idx] * num_example, dtype=np.int64)
    return annos


if __name__ == '__main__':
    fire.Fire()
# python eval.py eval --cfg_file=configs/voxelnet_kitti_car.yaml --model_dir=./output