# RF-DETR to CoreML

Read the [original repository](https://github.com/landchenxuan/rf-detr-to-coreml) for the details.

Export [RF-DETR](https://github.com/roboflow/rf-detr) object detection + instance segmentation models to Core ML model.

## Installation

```bash
git clone https://github.com/ryouchinsa/rf-detr-to-coreml.git
cd rf-detr-to-coreml
pip install -e .
```

## Convert to Core ML model

```bash
# Export pre-trained Nano detection model
python export_coreml.py --model nano

# Export a fine-tuned model
python export_coreml.py --model nano --weights checkpoint_best_total.pth
```

- Object Detection models `nano/small/medium/large`
- Instance Segmentation models `seg-nano/seg-small/seg-medium/seg-large`
