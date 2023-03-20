# How to run YOLOv7 on NVIDIA Deepstream

Instructions for executing the official [YOLOv7](https://github.com/WongKinYiu/yolov7) on NVIDIA Deepstream.

## 1. Export the Model

The [YOLOv7](https://github.com/WongKinYiu/yolov7) already has a method for exporting a trained model to an [ONNX](https://onnx.ai/) format which can be run like this:

```
python export.py \
--weights ./yolov7.pt \
--grid \
--end2end \
--simplify \
--topk-all 100 \
--iou-thres 0.65 \
--conf-thres 0.35 \
--img-size 640 640
```

This command will create an ONNX model with an [efficientNMS](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin) (Non Max Supression) node inserted. Care should be taken to ensure the parameters `topk-all`, `iou-thres` and `conf-thres` are configured correctly for how you want to execute your model. See the [efficientNMS](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin) README for further details.

## 2. Update the Custom Bounding Box Parser

    extern "C"
    bool NvDsInferParseCustomEfficientNMS (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                       NvDsInferNetworkInfo  const &networkInfo,
                                       NvDsInferParseDetectionParams const &detectionParams,
                                       std::vector<NvDsInferObjectDetectionInfo> &objectList) {
        if(outputLayersInfo.size() != 4)
        {
            std::cerr << "Mismatch in the number of output buffers."
                      << "Expected 4 output buffers, detected in the network :"
                      << outputLayersInfo.size() << std::endl;
            return false;
        }
        const char* log_enable = std::getenv("ENABLE_DEBUG");

        int* p_keep_count = (int *) outputLayersInfo[0].buffer;

        float* p_bboxes = (float *) outputLayersInfo[1].buffer;
        NvDsInferDims inferDims_p_bboxes = outputLayersInfo[1].inferDims;
        int numElements_p_bboxes=inferDims_p_bboxes.numElements;

        float* p_scores = (float *) outputLayersInfo[2].buffer;
        unsigned int* p_classes = (unsigned int *) outputLayersInfo[3].buffer;

        const float threshold = detectionParams.perClassThreshold[0];

        float max_bbox=0;
        for (int i=0; i < numElements_p_bboxes; i++)
        {
            if ( max_bbox < p_bboxes[i] )
                max_bbox=p_bboxes[i];
        }

        if (p_keep_count[0] > 0)
        {
            assert (!(max_bbox < 2.0));
            for (int i = 0; i < p_keep_count[0]; i++) {

                if ( p_scores[i] < threshold) continue;
                assert((unsigned int) p_classes[i] < detectionParams.numClassesConfigured);

                NvDsInferObjectDetectionInfo object;
                object.classId = (int) p_classes[i];
                object.detectionConfidence = p_scores[i];

                object.left=p_bboxes[4*i];
                object.top=p_bboxes[4*i+1];
                object.width=(p_bboxes[4*i+2] - object.left);
                object.height= (p_bboxes[4*i+3] - object.top);

                if(log_enable != NULL && std::stoi(log_enable)) {
                    std::cout << "label/conf/ x/y w/h -- "
                    << p_classes[i] << " "
                    << p_scores[i] << " "
                    << object.left << " " << object.top << " " << object.width << " "<< object.height << " "
                    << std::endl;
                }

                object.left=CLIP(object.left, 0, networkInfo.width - 1);
                object.top=CLIP(object.top, 0, networkInfo.height - 1);
                object.width=CLIP(object.width, 0, networkInfo.width - 1);
                object.height=CLIP(object.height, 0, networkInfo.height - 1);

                objectList.push_back(object);
            }
        }
        return true;
    }

At time of writing, NVIDIA does not support parsing the `efficientNMS` output format into the correct data structure for Deepstream [NvDsObjectMeta](https://docs.nvidia.com/metropolis/deepstream/sdk-api/struct__NvDsObjectMeta.html). 

To be able to map the model outputs from `efficientNMS` to the NVIDIA `NvDsObjectMeta` data structure you need this code has been added in `/opt/nvidia/deepstream/deepstream-6.1/sources/libs/nvdsinfer_customparser/nvdsinfer_custombboxparser.cpp`. A copy of this modified file exists in this repository that you should use to replace the file in that directory.

Once replaced, run `make` in the `/opt/nvidia/deepstream/deepstream-6.1/sources/libs/nvdsinfer_customparser` directory to compile the library with the new function.

## 3. Update your nvinfer Configuration

You then just need to make sure your [nvinfer](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html) stages point to the correct custom library and function like:

```ini
parse-bbox-func-name=NvDsInferParseCustomEfficientNMS
custom-lib-path=/opt/nvidia/deepstream/deepstream-6.1/sources/libs/nvdsinfer_customparser/libnvds_infercustomparser.so
```

# Common Problems

## numClassesConfigured Assertion

Messages like this indicate that at least one of the detections had a class id larger than the `nvinfer` configured `num-detected-classes`. The solution is to update `num-detected-classes` to the correct number of classes for your model.

```
Assertion `(unsigned int) p_classes[i] < detectionParams.numClassesConfigured’ failed.
```

# Common Problems

## FOr test1 app -> check updated config file.
