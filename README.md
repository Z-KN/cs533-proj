# cs533-proj
This is a project for cs533

## Step 1: Get ONNX Model
- Resnet18: Download from [onnx repo -> resnet18-v2-7.onnx](https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet18-v2-7.tar.gz)
- Transformer: We constructed the simpliest Transformer Model by invoking `torch.nn.Transformer`. Run following commands to get the model.
```
cd src
python transformer.py
```
## Step 2: Dump out Node Info
Gather node information from the model `MODEL_NAME (transformer/resnet)` and is recorded in a json file `$(MODEL_NAME)_node_info.json`.
```
python analyse.py --model $(MODEL_NAME)
```
