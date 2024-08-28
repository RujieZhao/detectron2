#!/usr/bin/env bash

#python3 visualize_data.py --source annotation --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --output-dir ./visualizer --show #dataloader

#===========50_FPN_1x For selection =====================
#CUDA_VISIBLE_DEVICES=0 ./train_net.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --resume --num-gpus 1 SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.0025 MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl OUTPUT_DIR ./output_3k #--resume SOLVER.MAX_ITER 20000

CUDA_VISIBLE_DEVICES=0 ./train_net.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --resume --num-gpus 1 SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.0025 MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl OUTPUT_DIR ./output_300 #--resume

#CUDA_VISIBLE_DEVICES=0,1 ./train_net.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --resume --num-gpus 2 SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.0025 MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl OUTPUT_DIR ./output_300

#==================50_FPN_1x evluation==============
#CUDA_VISIBLE_DEVICES=0 ./train_net.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --resume --eval-only  OUTPUT_DIR ./output_300/  #MODEL.WEIGHTS /mnt/ssd1/rujie/pytorch/detectron2/tools/output/last_checkpoint  ./output_300/148


#===================ResneXt_101_32_8d_FPN==================
#CUDA_VISIBLE_DEVICES=2,3 ./train_net.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml --resume --num-gpus 2 SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.002 OUTPUT_DIR "./output_mask_rcnn_R_101_FPN_3x" MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl


#=====================keypoint==============================
#CUDA_VISIBLE_DEVICES=1 ./train_net.py --config-file ../configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml --num-gpus 1 SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.0025 MODEL.KEYPOINT_ON True MODEL.WEIGHTS detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl OUTPUT_DIR "./output_keypoint_keypoint_rcnn_R_50_FPN_3x"

#CUDA_VISIBLE_DEVICES=1 ./train_net.py --eval-only --config-file ../configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml MODEL.KEYPOINT_ON True MODEL.WEIGHTS detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl OUTPUT_DIR "./output_keypoint_keypoint_rcnn_R_50_FPN_3x"

#CUDA_VISIBLE_DEVICES=0,1 ./train_net.py --config-file ../configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml --num-gpus 2 SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.002 MODEL.KEYPOINT_ON True MODEL.MASK_ON True MODEL.WEIGHTS detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl OUTPUT_DIR "./output_keypoint_w_mask_rcnn_R_50_FPN_3x"



