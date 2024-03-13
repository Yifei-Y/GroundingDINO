CUDA_VISIBLE_DEVICES=3 python demo/test_ap_on_coco.py \
-c groundingdino/config/GroundingDINO_SwinT_OGC.py \
-p weights/groundingdino_swint_ogc.pth \
--anno_path datasets/coco/annotations/instances_val2017.json \
--image_dir datasets/coco/val2017 \
--eval_type cls_agn \
--class_names no