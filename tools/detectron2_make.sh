
#!/usr/bin/env bash



#==========Resnet50_FPN================== --resume
#CUDA_VISIBLE_DEVICES=0 ./train_net.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --num-gpus 1  SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl

CUDA_VISIBLE_DEVICES=0,1 ./train_net.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --resume --num-gpus 2 SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.002 OUTPUT_DIR "./output_maskrcnn50fpnx3_2gpu" MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl

#===================ResneXt_101_32_8d_FPN==================
#CUDA_VISIBLE_DEVICES=0,1 ./train_net.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml --num-gpus 2 SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.002 OUTPUT_DIR "./output_mask_rcnn_R_101_FPN_3x" MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl


#CUDA_VISIBLE_DEVICES=1 ./train_net.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --resume --eval-only #MODEL.WEIGHTS /mnt/ssd1/rujie/pytorch/detectron2/tools/output/last_checkpoint

#fasterrcnn_C4
#CUDA_VISIBLE_DEVICES=0,1 ./train_net.py --config-file ../configs/PascalVOC-Detection/faster_rcnn_R_50_C4.yaml --num-gpus 2 OUTPUT_DIR "./output_voc_fasterrcnn_C4" SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.0025
#CUDA_VISIBLE_DEVICES=0 ./train_net.py --config-file ../configs/PascalVOC-Detection/faster_rcnn_R_50_C4.yaml --num-gpus 1 OUTPUT_DIR "./output_voc_fasterrcnn_C4" SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.002

#=====================keypoint==============================
#CUDA_VISIBLE_DEVICES=0,1 ./train_net.py --config-file ../configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml --num-gpus 2 SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.002 MODEL.KEYPOINT_ON True MODEL.MASK_ON True MODEL.WEIGHTS detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl OUTPUT_DIR "./output_keypoint_w_mask_rcnn_R_50_FPN_3x"

#CUDA_VISIBLE_DEVICES=1 ./train_net.py --eval-only --config-file ../configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml MODEL.KEYPOINT_ON True MODEL.WEIGHTS detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl OUTPUT_DIR "./output_keypoint_keypoint_rcnn_R_50_FPN_3x"

#CUDA_VISIBLE_DEVICES=0,1 ./train_net.py --config-file ../configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --resume --num-gpus 2 SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.002 MODEL.KEYPOINT_ON True MODEL.WEIGHTS detectron2://COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x/138363331/model_final_997cc7.pkl OUTPUT_DIR "./output_keypoint_rcnn_R_101_FPN_3x"






