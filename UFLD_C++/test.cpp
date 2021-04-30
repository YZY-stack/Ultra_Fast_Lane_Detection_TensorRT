#include <iostream>
#include <chrono>
#include <string>
#include <sstream>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "utils_test.hpp"
#include "mish.h"

// 定义基本参数
#define USE_FP16  // 注释掉变成 FP32, GTX系列不支持FP16
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1
static const int INPUT_C = 3;
static const int INPUT_H = 288;         // 可以设置的更小, 加快预测速度
static const int INPUT_W = 800;
static const int OUTPUT_C = 101;
static const int OUTPUT_H = 56;
static const int OUTPUT_W = 4;
static const int OUTPUT_SIZE = OUTPUT_C * OUTPUT_H * OUTPUT_W;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

// 使用Tensorrt API 创建trt推理engine模型
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder,  DataType dt) {
    INetworkDefinition* network = builder->createNetwork();
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    // 定义 Tensor
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{INPUT_C, INPUT_H, INPUT_W });  
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../lane.trtcy"); // hpp文件里面的解析函数, 会返回 weightMap
#if 0
    /* 打印layer的名字 */
    for(std::map<std::string, Weights>::iterator iter = weightMap.begin(); iter != weightMap.end() ; iter++)
    {
        std::cout << iter->first << std::endl;
    }
#endif

    // API 搭建网络,将权重放到tensorrt网络里面, 可参照onnx结构编写
    auto conv1 = network->addConvolution(*data, 64, DimsHW{ 7, 7 }, weightMap["model.conv1.weight"], emptywts); // 第一层 64*3*7*7
    assert(conv1);
    conv1->setStride(DimsHW{2, 2});     // 设置 Stride
    conv1->setPadding(DimsHW{3, 3});    // 设置 padding
    conv1->setNbGroups(1);              // 设置 group

    auto bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "model.bn1", 1e-5);                         // BN层
    auto relu0 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);                                 // relu激活层
    IPoolingLayer* pool0 = network->addPooling(*relu0->getOutput(0), PoolingType::kMAX, DimsHW{ 3, 3 });            // Maxpool层, 耗时间
    pool0->setStride( DimsHW{ 2, 2 } );
    pool0->setPadding( DimsHW{ 1, 1 } );
    assert(pool0);

    // 下面是基本网络单元设置, ResidualBlock里面包括: Conv, BN, Relu, Conv, BN等,这些信息可以从onnx文件看出来
    auto layer1_0 = ResidualBlock(network, weightMap, *pool0->getOutput(0), 64, 64, 1, "model.layer1.0.");

    auto layer1_1 = ResidualBlock(network, weightMap, *layer1_0->getOutput(0), 64, 64, 1, "model.layer1.1."); // downsample 在ResidualBlock里面加了

    auto layer2_0 = ResidualBlock(network, weightMap, *layer1_1->getOutput(0), 64, 128, 2, "model.layer2.0."); 

    auto layer2_1 = ResidualBlock(network, weightMap, *layer2_0->getOutput(0), 128, 128, 1, "model.layer2.1.");

    auto layer3_0 = ResidualBlock(network, weightMap, *layer2_1->getOutput(0), 128, 256, 2, "model.layer3.0.");

    auto layer3_1 = ResidualBlock(network, weightMap, *layer3_0->getOutput(0), 256, 256, 1, "model.layer3.1.");

    auto layer4_0 = ResidualBlock(network, weightMap, *layer3_1->getOutput(0), 256, 512, 2, "model.layer4.0.");

    auto layer4_1 = ResidualBlock(network, weightMap, *layer4_0->getOutput(0), 512, 512, 1, "model.layer4.1.");

#if 0
    /*  debug 查看tensor的维度 */
    Dims dims1 = layer4_1->getOutput(0)->getDimensions();
    for (int i = 0; i < dims1.nbDims; i++)
    {
        std::cout << dims1.d[i] << "-" << (int)dims1.type[i] << "   ";
    }
    std::cout << std::endl;
#endif
    // 第二个大网络层
    auto conv2 = network->addConvolution(*layer4_1->getOutput(0), 8, DimsHW{ 1, 1 }, weightMap["pool.weight"], weightMap["pool.bias"]);
    assert(conv2);
    conv2->setStride(DimsHW{1, 1});
    conv2->setPadding(DimsHW{0, 0});
    conv2->setNbGroups(1);

    IShuffleLayer* permute0 = network->addShuffle(*conv2->getOutput(0));
    assert(permute0);
    permute0->setReshapeDimensions( Dims2{1, 1800});

    auto fcwts0 = network->addConstant(nvinfer1::Dims2(2048, 1800), weightMap["cls.0.weight"]);
    auto matrixMultLayer0 = network->addMatrixMultiply(*permute0->getOutput(0), false, *fcwts0->getOutput(0), true);
    // 对应MatMul op, network()->addMatrixMultiply()，解释器会根据导入模型的节点按图拓扑顺序搭建TensorRT的网络节点

    assert(matrixMultLayer0 != nullptr);
    // 添加元素层以增加偏差
    auto fcbias0 = network->addConstant(nvinfer1::Dims2(1, 2048), weightMap["cls.0.bias"]);

    auto addBiasLayer0 = network->addElementWise(*matrixMultLayer0->getOutput(0), *fcbias0->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    assert(addBiasLayer0 != nullptr);

    auto relu = network->addActivation(*addBiasLayer0->getOutput(0), ActivationType::kRELU);

    auto fcwts1 = network->addConstant(nvinfer1::Dims2(22624, 2048), weightMap["cls.2.weight"]); // Gemm层
    auto matrixMultLayer1 = network->addMatrixMultiply(*relu->getOutput(0), false, *fcwts1->getOutput(0), true); // 耗时间

    assert(matrixMultLayer1 != nullptr);
    // 添加元素层以增加偏差
    auto fcbias1 = network->addConstant(nvinfer1::Dims2(1, 22624), weightMap["cls.2.bias"]);

    auto addBiasLayer1 = network->addElementWise(*matrixMultLayer1->getOutput(0), *fcbias1->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    assert(addBiasLayer1 != nullptr);

    IShuffleLayer* permute1 = network->addShuffle(*addBiasLayer1->getOutput(0));
    assert(permute1);
    permute1->setReshapeDimensions( Dims3{ 101, 56, 4 }); // 对应reshape层

    permute1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*permute1->getOutput(0));

    // 构建 engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB

#ifdef USE_FP16  // 如果开启 FP16 半精度模式
    if(builder->platformHasFastFp16()) {
        std::cout << "Platform supports fp16 mode and use it !!!" << std::endl;
        builder->setFp16Mode(true);
    } else {
        std::cout << "Platform doesn't support fp16 mode so you can't use it !!!" << std::endl;
    }
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    std::cout << "Build engine successfully!" << std::endl; // engine 构建成功

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

   return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);

    // 创建模型以填充网络，然后设置输出并创建引擎
    ICudaEngine* engine = createEngine(maxBatchSize, builder, DataType::kFLOAT);    
    assert(engine != nullptr);

    // 序列化 engine
    (*modelStream) = engine->serialize();

    // destroy
    engine->destroy();
    builder->destroy();
}

// 推理代码
void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    //指向输入和输出设备缓冲区的指针，以传递给引擎。
    //引擎完全需要IEngine :: getNbBindings（）缓冲区数。
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    //为了绑定缓冲区，我们需要知道输入和输出张量的名称。
    //请注意，索引必须小于IEngine :: getNbBindings（）
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // 创建 GPU buffers, 分配空间
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // 创建 stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA 将批次数据输入到设备，异步推断批次，并将DMA输出返回主机, 内存拷贝
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float),
          cudaMemcpyHostToDevice, stream));  // 将输入从内存拷贝到显存里面

    context.enqueue(batchSize, buffers, stream, nullptr); // 结果推理

    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float),
          cudaMemcpyDeviceToHost, stream)); // 将输出从显存到内存
    cudaStreamSynchronize(stream);

    // 释放 stream 和 buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

std::vector<float> processImage(cv::Mat & img) // TODO 图片预处理 写到CUDA上
{
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(INPUT_W, INPUT_H));

    cv::Mat img_float;

    resized.convertTo(img_float, CV_32FC3, 1. / 255.);

    // HWC TO CHW
    std::vector<cv::Mat> input_channels(INPUT_C);
    cv::split(img_float, input_channels);

    // normalize
    std::vector<float> result(INPUT_H * INPUT_W * INPUT_C);
    auto data = result.data();
    int channelLength = INPUT_H * INPUT_W;
    static float mean[]= {0.485, 0.456, 0.406};
    static float std[] = {0.229, 0.224, 0.225};
    for (int i = 0; i < INPUT_C; ++i) {
        cv::Mat normed_channel = (input_channels[i] - mean[i]) / std[i];
        memcpy(data, normed_channel.data, channelLength * sizeof(float));
        data += channelLength;
    }

    return result;
}

/* (101,56,4), add softmax on 101_axis and calculate Expect */
void softmax_mul(float* x, float* y, int rows, int cols, int chan)
{
    for(int i = 0, wh = rows * cols; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            float sum = 0.0;
            float expect = 0.0;
            for(int k = 0; k < chan - 1; k++)
            {
                x[k * wh + i * cols + j] = exp(x[k * wh + i * cols + j]);
                sum += x[k * wh + i * cols + j];
            }
            for(int k = 0; k < chan - 1; k++)
            {
                x[k * wh + i * cols + j] /= sum;
            }
            for(int k = 0; k < chan - 1; k++)
            {
                x[k * wh + i * cols + j] = x[k * wh + i * cols + j] * (k + 1);
                expect += x[k * wh + i * cols + j];
            }
            y[i * cols + j] = expect;
        }
    }
}
/* (101,56,4), 计算 最大的 index  101_axis */
void argmax(float* x, float* y, int rows, int cols, int chan)
{
    for(int i = 0,wh = rows * cols; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            int max = -10000000;
            int max_ind = -1;
            for(int k = 0; k < chan; k++)
            {
                if(x[k * wh + i * cols + j] > max)
                {
                    max = x[k * wh + i * cols + j];
                    max_ind = k;
                }
            }
            y[i * cols + j] = max_ind;
        }
    }
}

// 主函数入口
int main(int argc, char** argv)
{
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{ nullptr };
    size_t size{ 0 };

    if (argc == 2 && std::string(argv[1]) == "-transfer")  // 生成engine文件
    {
            IHostMemory* modelStream{ nullptr };
            APIToModel(BATCH_SIZE, &modelStream);
            assert(modelStream != nullptr);
            std::ofstream p("lane_det.engine", std::ios::binary);
            if (!p) {
                    std::cerr << "could not open plan output file" << std::endl;
                    return -1;
            }
            p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
            modelStream->destroy();
            return 0;
    }
    else if (argc == 3 && std::string(argv[1]) == "-infer") // 推理部分
    {
            std::ifstream file("lane_det.engine", std::ios::binary); // 文件的读取,读到model里面
            if (file.good()) {
                    file.seekg(0, file.end); // 读到末尾,获取size
                    size = file.tellg();
                    file.seekg(0, file.beg);
                    trtModelStream = new char[size];
                    assert(trtModelStream);
                    file.read(trtModelStream, size);
                    file.close();
            }
    }
    else
    {
            std::cerr << "arguments not right!" << std::endl;
            std::cerr << "./lane_det -transfer  // serialize model to plan file" << std::endl;
            std::cerr << "./lane_det -infer ../data  // deserialize plan file and run inference" << std::endl;
            return -1;
    }

    /* 数据准备 */
    static float data[BATCH_SIZE * INPUT_C * INPUT_H * INPUT_W];
    static float prob[BATCH_SIZE * OUTPUT_SIZE];

    IRuntime* runtime = createInferRuntime(gLogger);  // 转换到Runtime
    assert(runtime != nullptr);

    nvinfer1::IPluginFactory *m_plugin = new m_pluginFactory();

    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, m_plugin); // 将engine反序列化到显存里面
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    std::vector<std::string> file_names;
    if (read_files_in_dir(argv[2], file_names) < 0) {
            std::cout << "read_files_in_dir failed." << std::endl;  // 读取文件夹的文件
            return -1;
    }
    // 基本参数
    int fcount = 0;
    int vis_h = 720;
    int vis_w = 1280;
    int col_sample_w = 8;
    for (int f = 0; f < (int)file_names.size(); f++)
    {
        cv::Mat vis;
        fcount++;
        if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size()) continue;
        for (int b = 0; b < fcount; b++)
        {   
            auto t1 = std::chrono::system_clock::now();
            cv::Mat img = cv::imread(std::string(argv[2]) + "/" + file_names[f - fcount + 1 + b], 1); // 读取图片, TODO尝试dnn, 改掉两个for循环
            if (img.empty()) continue;
            cv::resize(img, vis, cv::Size(vis_w, vis_h));

            std::vector<float> result(INPUT_C * INPUT_W * INPUT_H);
            result = processImage(img);
            memcpy(data, &result[0], INPUT_C * INPUT_W * INPUT_H * sizeof(float));  // 将处理后的所有图片存到 vector 里面了, 内存拷贝
            auto t2 = std::chrono::system_clock::now();
            std::cout << "图片预处理时间为: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;
            // 这是一个双层循环, 每次都加载一遍
        }

        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, BATCH_SIZE); //prob: size (101, 56, 4)
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time is " // 只统计模型预测时间, 不包含图像预处理后处理
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << " ms" << std::endl;

        std::vector<int> tusimple_row_anchor
            { 64,  68,  72,  76,  80,  84,  88,  92,  96,  100, 104, 108, 112,
              116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
              168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
              220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
              272, 276, 280, 284 };

        float max_ind[BATCH_SIZE * OUTPUT_H * OUTPUT_W];
        float prob_reverse[BATCH_SIZE * OUTPUT_SIZE];
        /* do out_j = out_j[:, ::-1, :] in python list*/
        float expect[BATCH_SIZE * OUTPUT_H * OUTPUT_W];
        for (int k = 0, wh = OUTPUT_W * OUTPUT_H; k < OUTPUT_C; k++)
        {
            for(int j = 0; j < OUTPUT_H; j ++)
            {
                for(int l = 0; l < OUTPUT_W; l++)
                {
                    prob_reverse[k * wh + (OUTPUT_H - 1 - j) * OUTPUT_W + l] =
                        prob[k * wh + j * OUTPUT_W + l];
                }
            }
        }

        argmax(prob_reverse, max_ind, OUTPUT_H, OUTPUT_W, OUTPUT_C);
        /* calculate softmax and Expect */
        softmax_mul(prob_reverse, expect, OUTPUT_H, OUTPUT_W, OUTPUT_C);
        for(int k = 0; k < OUTPUT_H; k++) {
            for(int j = 0; j < OUTPUT_W; j++) {
                max_ind[k * OUTPUT_W + j] == 100 ? expect[k * OUTPUT_W + j] = 0 :
                    expect[k * OUTPUT_W + j] = expect[k * OUTPUT_W + j];
            }
        }
        std::vector<int> i_ind;
        for(int k = 0; k < OUTPUT_W; k++) {
            int ii = 0;
            for(int g = 0; g < OUTPUT_H; g++) {
                if(expect[g * OUTPUT_W + k] != 0)
                    ii++;
            }
            if(ii > 2) {
                i_ind.push_back(k);
            }
        }
        for(int k = 0; k < OUTPUT_H; k++) {
            for(int ll = 0; ll < i_ind.size(); ll++) {
                if(expect[OUTPUT_W * k + i_ind[ll]] > 0) {
                    cv::Point pp =
                        { int(expect[OUTPUT_W * k + i_ind[ll]] * col_sample_w * vis_w / INPUT_W) - 1,
                          int( vis_h * tusimple_row_anchor[OUTPUT_H - 1 - k] / INPUT_H) - 1 };
                    cv::circle(vis, pp, 8, CV_RGB(0, 255 ,0), 2);
                }
            }
        }
        // cv::imshow("lane_vis",vis);
        // cv::waitKey(0);
        cv::imwrite( "../results/res_" + file_names[f], vis); // 将预测好的图片保存
        auto t3 = std::chrono::system_clock::now();
        std::cout << "图片后处理时间为: " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - end).count() << "ms" << std::endl;
    }

    return 0;
}
