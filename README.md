# Yolov5 Pruning
 Using "[Learning Efficient Convolutional Networks Through Network Slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) (ICCV2017) channel pruning method prune yolov5s on MSCOCO 2017 Person dataset.
## The Dataset
MSCOCO 2017 dataset. Reserve only no crowed person  class and remove all other class label. 

You can download MSCOCO 2017 dataset and use `scripts/create_yolov5_dataset_from_coco.py` to create the dataset.
## The Model
Default yolov5s.
## Results
**The results are generate by `test.py`**

|Network |Sparsity Rate |mAP<sup>val (without finetune)<br>0.5 |mAP<sup>val(finetune)<br>0.5 |mAP<sup>val(finetune)<br>0.5:0.95 |Speed<br><sup>RTX3070 (ms) |params<br><sup>(M) |
|---               |---   |---      |---      |---     |---    |---   
|yolov5s           |      |         |62.1     |45.2    |3.0    |7.0  |
|yolov5s 30% pruned|1e-5  |10.5     |62.4     |44.8    |2.7    |3.5  |
|yolov5s 50% pruned|1e-5  |0        |58.8     |41.5    |2.3    |1.8  |
|yolov5s 70% pruned|1e-5  |0        |46.6     |30.5    |1.8    |0.7  |


## Usage
Set pruning ratio as 0.7 for example.
### Step1: Sparsity training from scrath.
```bash 
python train.py --data dataset/data.yaml  --cfg models/yolov5s-person.yaml --epochs 150 --slimming --name yolov5s_coco-person_slimming --weights ''  --batch-size 32 
```
### Step2: Prune the sparsed net.



```bash
python prune_net.py --weight /path/to/the/sparsed/weight --pruning_ratio 0.7
```

The pruned net weight and cfg file will save to "pruned_net" directory.

### Step3: Finetune the pruned net.
```bash
python train.py --data dataset/data.yaml --hyp data/hyps/hyp.finetune.yaml --cfg pruned_net/prune_0.7_pruned_net.yaml --epochs 30  --name yolov5s_coco-person_slimming_finetune_0.7 --weights pruned_net/prune_0.7_pruned_net.pt  --batch-size 32 
```

## Reference

```
@InProceedings{Liu_2017_ICCV,
    author = {Liu, Zhuang and Li, Jianguo and Shen, Zhiqiang and Huang, Gao and Yan, Shoumeng and Zhang, Changshui},
    title = {Learning Efficient Convolutional Networks Through Network Slimming},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {Oct},
    year = {2017}
}
```
[foolwood/pytorch-slimming](https://github.com/foolwood/pytorch-slimming)
