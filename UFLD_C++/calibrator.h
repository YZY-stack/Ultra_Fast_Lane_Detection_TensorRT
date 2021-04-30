#ifndef ENTROPY_CALIBRATOR_H
#define ENTROPY_CALIBRATOR_H

#include "NvInfer.h"
#include <string>
#include <vector>

class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    Int8EntropyCalibrator2(int batchsize, int input_w, int input_h, const char* img_dir, const char* calib_table_name, const char* input_blob_name, bool read_cache = true);

    virtual ~Int8EntropyCalibrator2();
    // 都override重载一下
    int getBatchSize() const override; //每次输入数据是多少
    bool getBatch(void* bindings[], const char* names[], int nbBindings) override; // 获取batch size数据到显存
    const void* readCalibrationCache(size_t& length) override; // 读取校准表
    void writeCalibrationCache(const void* cache, size_t length) override;  // 写校准表

private:
    int batchsize_;
    int input_w_;
    int input_h_;
    int img_idx_;
    std::string img_dir_;
    std::vector<std::string> img_files_;
    size_t input_count_;
    std::string calib_table_name_;
    const char* input_blob_name_;
    bool read_cache_;
    void* device_input_;
    std::vector<char> calib_cache_;
};

#endif // ENTROPY_CALIBRATOR_H
