# 基于SSD-Tensorflow目标检测识别 - web版

> 网站图片检测识别示例
> 如有问题，欢迎拍砖，多多指教 😀

## 简要说明

本案例主要使用ssd-tensorflow训练后模型,搭建flask网站进行上传图片检测,其中[caffe中的 web_demo](https://github.com/BVLC/caffe/tree/master/examples/web_demo)给予较多指导.

### 环境配置  
ubuntu16 + python3 + tensorflow1.3 + flask

### 具体使用

- git clone https://github.com/RookieDay/object_detection.git 
- 下载对应依赖包
- 解压checkpoints目录下压缩包后，务必将ssd_300_vgg.ckpt/目录下文件全部挪到checkpoints/目录下面
- python app.py - d (debug模式)
- 浏览器输入 http://0.0.0.0:5000/ 即可

## 网站效果预览

![初始页面](https://github.com/RookieDay/object_detection/blob/master/web_01.png)

![检测识别](https://github.com/RookieDay/object_detection/blob/master/web_02.png)
如上图,显示了图片中所属类别、概率、耗时以及所框选识别检测范围
## 视频效果预览

![视频检测识别演示](https://github.com/RookieDay/object_detection/blob/master/prev_video.gif)

[视频下载演示](https://github.com/RookieDay/object_detection/blob/master/notebooks/mua.mp4)

## 开发计划

- [x] 上传单张图片检测；
- [ ] 实现上传多张图片检测；
- [ ] 实现上传视频检测；
- [ ] 添加进度条；


## 许可

[MIT](./LICENSE) &copy; [RookieDay](https://github.com/RookieDay)
