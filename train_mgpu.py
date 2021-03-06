# python train_mgpu.py --config=configs/voxelnet_kitti_car.yaml --model_dir=./output --use_vdl=True # 单
# python -m paddle.distributed.launch train_mgpu.py --config=configs/voxelnet_kitti_car.yaml --model_dir=./output --use_vdl=True # 多卡 May error

import pickle 
from pathlib import Path 
import shutil 
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
from visualdl import LogWriter
from utils import get_sys_env
from utils import logger
import argparse
import random
import numpy as np
# from memory_profiler import profile # for debug

def worker_init_fn(worker_id):
    np.random.seed(random.randint(0, 100000))

# @profile(precision=4,stream=open('memory_profiler.log','w+')) #only for debug
def train(cfg_file = None,
          model_dir = None,
          result_path=None,
          resume = False,
          use_vdl = True,
          create_folder=False,
          display_step=10,
          summary_step=5,
          pickle_result=False):
    
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank

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
    print("num_trainable parameters:", len(list(net.parameters())))

    ######################
    # BUILD OPTIMIZER
    ######################
    # we need global_step to create lr_scheduler, so restore net first.
    # libs.tools.try_restore_latest_checkpoints(model_dir, [net])
    gstep = net.get_global_step() - 1
    optimizer_cfg = train_cfg.OPTIMIZER
    if train_cfg.ENABLE_MIXED_PRECISION:
        net.half()
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    optimizer = core.build_optimizer(optimizer_cfg, net.parameters())
    if train_cfg.ENABLE_MIXED_PRECISION:
        loss_scale = train_cfg.LOSS_SCALE_FACTOR
        mixed_optimizer = libs.tools.MixedPrecisionWrapper(
            optimizer, loss_scale)
    else:
        mixed_optimizer = optimizer
   # must restore optimizer AFTER using MixedPrecisionWrapper
    # libs.tools.try_restore_latest_checkpoints(model_dir,
    #                                           [mixed_optimizer])
    #################
    # RESUME
    #################
    if resume:
        if os.path.exists(model_dir):
            resume_model = os.path.normpath(model_dir)
            ckpt_path = os.path.join(model_dir, 'voxelnet.pdparams')
            para_state_dict = paddle.load(ckpt_path)
            net.set_state_dict(para_state_dict)
            ckpt_path = os.path.join(resume_model, 'voxelnet.pdopt')
            opti_state_dict = paddle.load(ckpt_path)
            mixed_optimizer.set_state_dict(opti_state_dict)
            # print("++++++++++++++++++++++++++++++++LOAD voxelnet.pdparams SUCCESS!!!!!+++++++++++++++++++++++++++++")
            # print("++++++++++++++++++++++++++++++++LOAD voxelnet.pdopt SUCCESS!!!!!+++++++++++++++++++++++++++++")

        else:
            raise ValueError(
                'Directory of the model needed to resume is not Found: {}'.format(model_dir))
    
    if nranks > 1:
        paddle.distributed.fleet.init(is_collective=True)
        mixed_optimizer = paddle.distributed.fleet.distributed_optimizer(
        mixed_optimizer)  # The return is Fleet object
        ddp_net = paddle.distributed.fleet.distributed_model(net)
    
    if train_cfg.ENABLE_MIXED_PRECISION:
        float_dtype = paddle.float16
    else:
        float_dtype = paddle.float32

    ######################
    # PREPARE INPUT
    ######################
    dataset = core.build_input_reader(input_cfg,
                                      model_cfg,
                                      training= True,
                                      voxel_generator=voxel_generator,
                                      target_assigner=target_assigner)
    eval_dataset = core.build_input_reader(input_cfg,
                                           model_cfg,
                                           training= False,
                                           voxel_generator=voxel_generator,
                                           target_assigner=target_assigner)

    batch_sampler = paddle.io.DistributedBatchSampler(dataset, batch_size=input_cfg.BATCH_SIZE, shuffle=True, drop_last=True)
    dataloader = paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        use_shared_memory=False,
        num_workers=input_cfg.NUM_WORKERS,
        collate_fn=merge_second_batch,
        worker_init_fn=worker_init_fn
        )

    eval_batch_sampler = paddle.io.DistributedBatchSampler(eval_dataset, batch_size=eval_input_cfg.BATCH_SIZE, shuffle=False, drop_last=False)
    eval_dataloader = paddle.io.DataLoader(
        dataset=eval_dataset,
        batch_sampler=eval_batch_sampler,
        use_shared_memory=False,
        num_workers=eval_input_cfg.NUM_WORKERS,
        collate_fn=merge_second_batch,
        worker_init_fn=worker_init_fn
        )

    ######################
    # TRAINING
    ######################
    log_path = model_dir / 'log.txt'
    logf = open(log_path, 'a')
    # logf.write(proto_str)
    logf.write("\n")
    if use_vdl:
        log_writer = LogWriter(os.path.normpath(model_dir))

    remain_steps = train_cfg.STEPS - net.get_global_step()
    pre_loop = net.get_global_step() // train_cfg.STEPS_PER_EPOCH + 1
    remain_loop = remain_steps // train_cfg.STEPS_PER_EPOCH + 1
    pd_start_time = time.time()
    clear_metrics_every_epoch = train_cfg.CLEAR_METRICS_EVERY_EPOCH
    print("++++++++++++++++++++++++++++++++++++TRAIN PREPARE++++++++++++++++++++++++++++++++++++++++++++++++")
    try:
        print("++++++++++++++++++++++++++++++++++++START TRAIN++++++++++++++++++++++++++++++++++++++++++++++++")
        for _ in range(remain_loop): # total_loop todo adjust
            if clear_metrics_every_epoch:
                net.clear_metrics()

            for i,example in enumerate(dataloader()):

                print("++++++++++++++++++++++++++++++++++++START LOOP:{}-STEP:{}++++++++++++++++++++++++++++++++++++++++++++++++".format(_+pre_loop,i))
                st = time.time()
                example = example_convert_to_paddle(example, float_dtype)
                batch_size = example["anchors"].shape[0]

                if nranks > 1:
                    ret_dict = ddp_net(example)
                else:
                    ret_dict = net(example)

                # box_preds = ret_dict["box_preds"]
                cls_preds = ret_dict["cls_preds"]
                loss = ret_dict["loss"].mean()
                cls_loss_reduced = ret_dict["cls_loss_reduced"].mean()
                loc_loss_reduced = ret_dict["loc_loss_reduced"].mean()
                cls_pos_loss = ret_dict["cls_pos_loss"]
                cls_neg_loss = ret_dict["cls_neg_loss"]
                loc_loss = ret_dict["loc_loss"]
                cls_loss = ret_dict["cls_loss"]
                dir_loss_reduced = ret_dict["dir_loss_reduced"]
                cared = ret_dict["cared"]
                labels = example["labels"]

                if train_cfg.ENABLE_MIXED_PRECISION:
                    loss *= loss_scale

                if(train_cfg.ENABLE_ACCUMULATION_GRAD): # 是否使用梯度累积
                    loss = loss / train_cfg.ACCUMULATION_STEPS # loss每次都会更新，因此每次都除以steps再加到原来的梯度上面去

                loss.backward()

                if(train_cfg.ENABLE_ACCUMULATION_GRAD): # 是否使用梯度累积
                    if((i+1)%train_cfg.ACCUMULATION_STEPS)==0:
                        mixed_optimizer.step()

                        # update lr
                        if isinstance(mixed_optimizer, paddle.distributed.fleet.Fleet):
                            lr_sche = mixed_optimizer.user_defined_optimizer._learning_rate
                        else:
                            lr_sche = mixed_optimizer._learning_rate
                        if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
                            lr_sche.step()

                        # net.clear_gradients()
                        mixed_optimizer.clear_grad()
                else:
                    mixed_optimizer.step()

                    # update lr
                    if isinstance(mixed_optimizer, paddle.distributed.fleet.Fleet):
                        lr_sche = mixed_optimizer.user_defined_optimizer._learning_rate
                    else:
                        lr_sche = mixed_optimizer._learning_rate
                    if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
                        lr_sche.step()

                    # net.clear_gradients()
                    mixed_optimizer.clear_grad()

                # print("++++++++++++++++++++++++++++++++++++OVER UPDATE++++++++++++++++++++++++++++++++++++++++++++++++")
                net.update_global_step()
                # print("++++++++++++++++++++++++++++++++++++STAR UPDATE_METRICS++++++++++++++++++++++++++++++++++++++++++++++++")
                net_metrics = net.update_metrics(
                    cls_loss_reduced.detach(),
                    loc_loss_reduced.detach(), 
                    cls_preds,
                    labels, cared)
                # print("++++++++++++++++++++++++++++++++++++OVER UPDATE_METRICS++++++++++++++++++++++++++++++++++++++++++++++++")
                step_time = (time.time() - st)
                metrics = {}
                num_pos = int((labels > 0).astype("float32")[0].sum().cpu().numpy())
                num_neg = int((labels == 0).astype("float32")[0].sum().cpu().numpy())
                if 'anchors_mask' not in example:
                    num_anchors = example['anchors'].shape[1]
                else:
                    num_anchors = int(example['anchors_mask'].astype("float32")[0].sum().numpy())

                global_step = net.get_global_step()
                if global_step % display_step == 0:
                    loc_loss_elem = [
                        float(loc_loss[:,:,i].sum().detach().cpu().numpy() /
                        batch_size) for i in range(loc_loss.shape[-1])
                    ]
                    metrics['time'] = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
                    metrics["epoch"] = _ + pre_loop
                    metrics["step"] = global_step
                    metrics["steptime"] = step_time
                    # print("++++++++++++++++++++++++++++++++++++START METRICS_UPDATE++++++++++++++++++++++++++++++++++++++++++++++++")
                    metrics.update(net_metrics)
                    # print("++++++++++++++++++++++++++++++++++++OVER METRICS_UPDATE++++++++++++++++++++++++++++++++++++++++++++++++")
                    metrics["loss"] = {}
                    metrics["loss"]["loc_elem"] = loc_loss_elem
                    metrics["loss"]["cls_pos_rt"] = float(cls_pos_loss.detach().cpu().numpy()[0])
                    metrics["loss"]["cls_neg_rt"] = float(cls_neg_loss.detach().cpu().numpy()[0])

                    if model_cfg.BACKBONE.use_direction_classifier:
                        metrics["loss"]["dir_rt"] = float(dir_loss_reduced.detach().cpu().numpy()[0])
                    metrics["num_vox"] = int(example["voxels"].shape[0])
                    metrics["num_pos"] = int(num_pos)
                    metrics["num_neg"] = int(num_neg)
                    metrics["num_anchors"] = int(num_anchors)
                    metrics["lr"] = float(mixed_optimizer.get_lr())

                    metrics["image_idx"] = example['image_idx'][0]
                    flatted_metrics = flat_nested_json_dict(metrics)
                    flatted_summarys = flat_nested_json_dict(metrics, "/")
                    metrics_str_list = []
                    for k,v in flatted_metrics.items():
                        if isinstance(v,float):
                            metrics_str_list.append(f"{k}={v:.3}")
                        elif isinstance(v, (list, tuple)):
                            if v and isinstance(v[0], float):
                                v_str = ', '.join([f"{e:.3}" for e in v])
                                metrics_str_list.append(f"{k}=[{v_str}]")
                            else:
                                metrics_str_list.append(f"{k}={v}")
                        else:
                            metrics_str_list.append(f"{k}={v}")
                    log_str = ', '.join(metrics_str_list)
                    print("++++++++++++++++++++++++++++++++++++STAR EVAL++++++++++++++++++++++++++++++++++++++++++++++++")
                    print("++++++++++++++++++++++++++++++++++++STAR EVAL++++++++++++++++++++++++++++++++++++++++++++++++",file=logf)
                    print(log_str)
                    print(log_str, file=logf)

                    if use_vdl:
                        log_writer.add_scalar('Train/loss', float(loss.detach().cpu().numpy()[0]), metrics["step"])
                        log_writer.add_scalar('Train/cls_loss', metrics["cls_loss"], metrics["step"])
                        log_writer.add_scalar('Train/loc_loss', metrics["loc_loss"], metrics["step"])
                        log_writer.add_scalar('Train/lr', metrics["lr"], metrics["step"])
                        log_writer.add_scalar('Train/steptime', metrics["steptime"], metrics["step"])            

                pd_elasped_time = time.time() - pd_start_time
                if pd_elasped_time > train_cfg.SAVE_CHECKPOINTS_SECS:
                    paddle.save(net.state_dict(), os.path.join(model_dir, "voxelnet.pdparams"))
                    paddle.save(mixed_optimizer.state_dict(), os.path.join(model_dir, "voxelnet.pdopt"))
                    pd_start_time = time.time()
            
            print("++++++++++++++++++++++++++++++++++++START SAVE++++++++++++++++++++++++++++++++++++++++++++++++")
            paddle.save(net.state_dict(), os.path.join(model_dir, "voxelnet.pdparams"))
            paddle.save(mixed_optimizer.state_dict(), os.path.join(model_dir, "voxelnet.pdopt"))
            # Ensure that all evaluation points are saved forever
            paddle.save(net.state_dict(), os.path.join(eval_checkpoint_dir, "voxelnet.pdparams"))
            paddle.save(mixed_optimizer.state_dict(), os.path.join(eval_checkpoint_dir, "voxelnet.pdopt"))
            print("++++++++++++++++++++++++++++++++++++OVER SAVE++++++++++++++++++++++++++++++++++++++++++++++++")

            if((_+pre_loop+1)%train_cfg.EPOCHS_PER_EVAL==0 or _ ==remain_loop-1):
                net.eval()
                center_limit_range = model_cfg.POST_PROCESSING.post_center_limit_range
                result_path_step = result_path / f"loop_{_}_step_{net.get_global_step()}"
                result_path_step.mkdir(parents=True, exist_ok=True)
                # print("++++++++++++++++++++++++++++++++++++OVER EVAL++++++++++++++++++++++++++++++++++++++++++++++++")
                t = time.time()
                dt_annos = []
                print("************************************START EVAL************************************")
                prog_bar = ProgressBar()
                print("**********************************************************************************")
                prog_bar.start(len(eval_dataset) // eval_input_cfg.BATCH_SIZE + 1)
                print("**********************************************************************************")
                for example in eval_dataloader():
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
                sec_per_ex = len(eval_dataset) / (time.time() - t)
                print("***************************************1*******************************************")
                print(f"avg forward time per example: {net.avg_forward_time:.3f}", file=logf)
                print(f"avg postprocess time per example: {net.avg_postprocess_time:.3f}", file=logf)
                print(f'generate label finished({sec_per_ex:.2f}/s). start eval:', file=logf)
                gt_annos = [info["annos"] for info in eval_dataset.dataset.kitti_infos]
                print("***************************************2*******************************************")
                if not pickle_result:
                    dt_annos = kitti.get_label_annos(result_path_step)
                    result, mAPbbox, mAPbev, mAP3d, mAPaos = get_official_eval_result(gt_annos, dt_annos, class_names,
                                                                                        return_data=True)
                    print("***********************************3***********************************************")
                    print(result, file=logf)
                    print(result)
                
                else:
                    result, mAPbbox, mAPbev, mAP3d, mAPaos = get_official_eval_result(gt_annos, dt_annos, class_names,
                                                                    return_data=True)
                    print("***********************************3***********************************************")
                    print(result, file=logf)
                    print(result)
                print("**************************************4********************************************")

                result = get_coco_eval_result(gt_annos, dt_annos, class_names)
                print(result, file=logf)
                print(result)
                print("++++++++++++++++++++++++++++++++++++OVER EVAL++++++++++++++++++++++++++++++++++++++++++++++++")
                
                net.train()

        time.sleep(0.5)
        if use_vdl:
            log_writer.close()

    except Exception as e:
        paddle.save(net.state_dict(), os.path.join(model_dir, "voxelnet.pdparams"))
        paddle.save(mixed_optimizer.state_dict(), os.path.join(model_dir, "voxelnet.pdopt"))
        traceback.print_exc() # 跟踪错误

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


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
    # params of training
    parser.add_argument(
        '--config', dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        help='The directory for saving the model snapshot',
        type=str,
        default='./output')
    parser.add_argument(
        '--resume',
        dest='resume',
        help='Whether to resume model in save_dir/',
        type=bool,
        default=False)
    parser.add_argument(
        '--use_vdl',
        dest='use_vdl',
        help='Whether to record the data to VisualDL during training',
        type=bool,
        default=True)
    parser.add_argument(
        '--create_folder',
        dest='create_folder',
        help='Whether to create_folder in save_dir/.No use now.',
        type=bool,
        default=False)
    parser.add_argument(
        '--display_step',
        dest='display_step',
        help='Display logging information at every log_iters',
        default=10,
        type=int)
    parser.add_argument(
        '--summary_step',
        dest='summary_step',
        help='do summary. No use now.',
        default=5,
        type=int)
    parser.add_argument(
        '--pickle_result', 
        dest='pickle_result', 
        help='Whther to use pickle_result for eval',
        default=False,
        type=bool)
    return parser.parse_args()


def main(args):

    env_info = get_sys_env()
    info = ['{}: {}'.format(k, v) for k, v in env_info.items()]
    info = '\n'.join(['', format('Environment Information', '-^48s')] + info +
                     ['-' * 48])
    logger.info(info)

    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)

    train(cfg_file=args.cfg,
          model_dir=args.model_dir,
          resume=args.resume,
          use_vdl=args.use_vdl,
          create_folder=args.create_folder,
          display_step=args.display_step,
          summary_step=args.summary_step,
          pickle_result=args.pickle_result
          )

if __name__ == '__main__':
    args = parse_args()
    main(args)