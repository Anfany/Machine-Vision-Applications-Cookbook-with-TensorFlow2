# 目录

* [序](README.md#tensorflow2-object-detection-api-全流程文档)
  + [序：无关编码](README.md#序：无关编码)
  + [扫码关注](README.md#扫码关注)
  + [联系作者](README.md#联系作者)
* [1 Win10系统安装TensorFlow-gpu2.2](chapter1.md#1-win10系统tensorflow-gpu22安装)
  + [1.1  显卡驱动](chapter1.md#11-显卡驱动)
    + [1.1.1 选择版本](chapter1.md#111-选择版本)
    + [1.1.2 安装](chapter1.md#112-安装)
  + [1.2 CUDA工具包](chapter1.md#12-cuda工具包)
    + [1.2.1 选择版本](chapter1.md#121-选择版本)
    + [1.2.2 自定义安装](chapter1.md#122-自定义安装)
    + [1.2.3 添加环境变量](chapter1.md#123-添加环境变量)
    + [1.2.4 验证安装](chapter1.md#124-验证安装)
  + [1.3 cuDNN](chapter1.md#13-cudnn)
    + [1.3.1 选择版本](chapter1.md#131-选择版本)
    + [1.3.2 移动文件](chapter1.md#132-移动文件)
  + [1.4 安装Anaconda](chapter1.md#14-安装anaconda)
    + [1.4.1 下载](chapter1.md#141-下载)
    + [1.4.2 验证安装](chapter1.md#142-验证安装)
    + [1.4.3 conda创建虚拟环境](chapter1.md#143-conda创建虚拟环境)
    + [1.4.4 激活虚拟环境](chapter1.md#144-激活虚拟环境)
  + [1.5 运行方式](chapter1.md#15-运行方式)
    + [1.5.1 Jupyter Notebook](chapter1.md#151-jupyter-notebook)
      + 1.5.1.1 建立python3.8虚拟环境
      + 1.5.1.2 安装tensorflow-gpu 2.2 版本
      + 1.5.1.3 安装ipython
      + 1.5.1.4 安装jupyter notebook
      + 1.5.1.5 安装ipykernel
      + 1.5.1.6 更改文件路径
      + 1.5.1.7 验证安装
    + [1.5.2 PyCharm](chapter1.md#152-pycharm)
      + 1.5.2.1 下载社区版本
      + 1.5.2.2 设置编译器
      + 1.5.2.3 验证
    + [1.5.3 错误问题汇集](chapter1.md#153-错误问题汇集)
      + 1.5.3.1 出现出现.pywrap_tensorflow_internal import错误
* [2 搭建Object Detection环境](chapter2.md#2-搭建object-detection环境)
  + [2.1 安装TenorFlow模型库](chapter2.md#21-安装tensorflow模型库)
  + [2.2 Protobuf安装与编译](chapter2.md#22-protobuf安装与编译)
    + [2.2.1 下载protoc](chapter2.md#221-下载protoc)
    + [2.2.2 验证](chapter2.md#222-验证)
    + [2.2.3 编译](chapter2.md#223-编译)
  + [2.3 安装COCO API](chapter2.md#23-安装coco-api)
    + [2.3.1 安装cython](chapter2.md#231-安装cython)
    + [2.3.2 下载coco-api压缩包](chapter2.md#232-下载coco-api压缩包)
    + [2.3.3 安装](chapter2.md#233-安装)
  + [2.4 安装Object Detection API](chapter2.md#24-安装object-detection-api)
* [3 视频图片目标检测](chapter3.md#3-视频图片目标检测)
  + [3.1 模型下载](chapter3.md#31-模型下载)
    + [3.1.1 获取模型信息](chapter3.md#311-获取模型信息)
    + [3.1.2 下载模型](chapter3.md#312-下载模型)
  + [3.2 图片目标检测](chapter3.md#32-图片目标检测)
  + [3.3 视频目标检测](chapter3.md#33-视频目标检测)
    + [3.3.1 本地视频](chapter3.md#331-本地视频)
    + [3.3.2 在线视频](chapter3.md#332-在线视频)
* [4 训练自定义数据集](chapter4.md#4-训练自定义数据集)
  + [4.1 文件结构搭建](chapter4.md#41-文件结构搭建)
  + [4.2 准备数据集](chapter4.md#42-准备数据集)
    + [4.2.1 数据标注](chapter4.md#421-数据标注)
    + [4.2.2 数据集分割](chapter4.md#422-数据集分割)
    + [4.2.3 创建标签映射](chapter4.md#423-创建标签映射)
    + [4.2.4 转\*.xml为\*.record](chapter4.md#424-转xml为record格式)
  + [4.3 构建训练任务](chapter4.md#43-构建训练任务)
    + [4.3.1 获取预训练模型](chapter4.md#431-获取预训练模型)
    + [4.3.2 配置训练管道](chapter4.md#432-配置训练管道)
      + 4.3.2.1 设置检测对象类的格式
      + 4.3.2.2 修改批训练大小
      + 4.3.2.3 模型配置
      + 4.3.2.4 训练数据集
      + 4.3.2.5 测试数据集
    + [4.3.3 训练模型](chapter4.md#433-训练模型)
      + 4.3.3.1 No module named 'lvis'
      + 4.3.3.2 No module named 'official'
      + 4.3.3.3 No module named 'yaml'
      + 4.3.3.4 开始训练
    + [4.3.4 训练过程可视化](chapter4.md#434-训练过程可视化)
  + [4.4 评估模型](chapter4.md#44-评估模型)
    + [4.4.1 设置评估标准](chapter4.md#441-设置评估标准)
    + [4.4.2 评估](chapter4.md#442-评估)
    + [4.4.3 评估结果查看](chapter4.md#443-评估结果查看)
  + [4.5 模型保存](chapter4.md#45-模型保存)
  + [4.6 模型预测](chapter4.md#46-模型预测)
* [5 身份识别模型](chapter5.md)
* [6 目标检测知识备忘](chapter6.md)
* [7 附录](chapter7.md)