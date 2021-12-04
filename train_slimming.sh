nohup python train.py --data dataset/data.yaml  --cfg models/yolov5s-person.yaml --epochs 150 --slimming --name yolov5s_coco-person_slimming --weights ''  --batch-size 32 > log.txt 2>&1 &
