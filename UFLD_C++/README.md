# Ultra-Fast-Lane-Detection

The Pytorch implementation is [Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection).

### 生成权重文件 
```
python3 export_trtcy.py
```
### trt模型生成
#### 修改第十行为 #define USE_FP32,则为FP32模式, 修改第十行为 #define USE_FP16,则为FP16模式
```
mkdir build
cd build
cmake ..
make
./lane_det -transfer             //  'lane_det.engine'
```
### Tensorrt预测
```
./lane_det -infer  ../imgs 

```
