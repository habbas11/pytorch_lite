import 'dart:async';
import 'dart:io';
import 'dart:math';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:pytorch_lite/classes/result_object_detection.dart';
import 'package:pytorch_lite/enums/model_type.dart';
import 'package:pytorch_lite/native_wrapper.dart';
import 'package:pytorch_lite/post_processor.dart';

export 'enums/dtype.dart';
export 'classes/rect.dart';
export 'classes/result_object_detection.dart';
export 'enums/model_type.dart';

const torchVisionNormMeanRGB = [0.485, 0.456, 0.406];
const torchVisionNormSTDRGB = [0.229, 0.224, 0.225];

const List<double> noMeanRgb = [0, 0, 0];
const List<double> noStdRgb = [1, 1, 1];

// is think the best idea is to make the isolates here instead of pytorchFFI, and make them static like i did there
// and also add a method for running on camera image to avoid making it hard on people

class PytorchLite {
  ///Sets pytorch object detection model (path and lables) and returns Model
  static Future<ModelObjectDetection> loadObjectDetectionModel(
      String path, int numberOfClasses, int imageWidth, int imageHeight,
      {String? labelPath,
      ObjectDetectionModelType objectDetectionModelType =
          ObjectDetectionModelType.yolov5}) async {
    int index = await PytorchFfi.loadModel(path);

    List<String> labels = [];
    if (labelPath != null) {
      if (labelPath.endsWith(".txt")) {
        labels = await _getLabelsTxt(labelPath);
      }
    }
    return ModelObjectDetection(
        index,
        imageWidth,
        imageHeight,
        labels,
        PostProcessorObjectDetection(
            numberOfClasses, imageWidth, imageHeight, objectDetectionModelType),
        modelType: objectDetectionModelType);
  }
}

///get labels in txt format
///each line is a label
Future<List<String>> _getLabelsTxt(String labelPath) async {
  final file = File(labelPath);
  final contents = await file.readAsString();
  return contents.split("\n");
}

class ModelObjectDetection {
  final int _index;
  final int imageWidth;
  final int imageHeight;
  final List<String> labels;
  final ObjectDetectionModelType modelType;
  final PostProcessorObjectDetection postProcessorObjectDetection;

  ModelObjectDetection(this._index, this.imageWidth, this.imageHeight,
      this.labels, this.postProcessorObjectDetection,
      {this.modelType = ObjectDetectionModelType.yolov5});

  ///predicts image but returns the raw net output
  Future<List<ResultObjectDetection>> getImagePredictionList(
      Uint8List imageAsBytes,
      {double minimumScore = 0.5,
      double iOUThreshold = 0.5,
      int boxesLimit = 10,
      List<double> mean = noMeanRgb,
      List<double> std = noStdRgb}) async {
    List<ResultObjectDetection> prediction =
        await PytorchFfi.imageModelInferenceObjectDetection(
            _index,
            imageAsBytes,
            imageHeight,
            imageWidth,
            mean,
            std,
            modelType == ObjectDetectionModelType.yolov5,
            postProcessorObjectDetection.modelOutputLength,
            postProcessorObjectDetection);
    return prediction;
  }

  ///predicts image and returns the supposed label belonging to it
  Future<List<ResultObjectDetection>> getImagePrediction(Uint8List imageAsBytes,
      {double minimumScore = 0.5,
      double iOUThreshold = 0.5,
      int boxesLimit = 10,
      List<double> mean = noMeanRgb,
      List<double> std = noStdRgb}) async {
    List<ResultObjectDetection> prediction = await getImagePredictionList(
        imageAsBytes,
        minimumScore: minimumScore,
        iOUThreshold: iOUThreshold,
        boxesLimit: boxesLimit,
        mean: mean,
        std: std);

    for (var element in prediction) {
      element.className = labels[element.classIndex];
    }

    return prediction;
  }

  ///predicts image but returns the raw net output
  Future<List<ResultObjectDetection>> getCameraImagePredictionList(
      CameraImage cameraImage, int rotation,
      {double minimumScore = 0.5,
      double iOUThreshold = 0.5,
      int boxesLimit = 10,
      List<double> mean = noMeanRgb,
      List<double> std = noStdRgb}) async {
    // On Android the image format is YUV and we get a buffer per channel,
    // in iOS the format is BGRA and we get a single buffer for all channels.
    // So the yBuffer variable on Android will be just the Y channel but on iOS it will be
    // the entire image
    var planes = cameraImage.planes;
    var yBuffer = planes[0].bytes;

    Uint8List? uBuffer;
    Uint8List? vBuffer;

    if (Platform.isAndroid) {
      uBuffer = planes[1].bytes;
      vBuffer = planes[2].bytes;
    }

    List<ResultObjectDetection> prediction =
        await PytorchFfi.cameraImageModelInferenceObjectDetection(
            _index,
            yBuffer,
            uBuffer,
            vBuffer,
            rotation,
            imageHeight,
            imageWidth,
            cameraImage.height,
            cameraImage.width,
            mean,
            std,
            modelType == ObjectDetectionModelType.yolov5,
            postProcessorObjectDetection.modelOutputLength,
            postProcessorObjectDetection);

    return prediction;
  }

  ///predicts image and returns the supposed label belonging to it
  Future<List<ResultObjectDetection>> getCameraImagePrediction(
      CameraImage cameraImage, int rotation,
      {double minimumScore = 0.5,
      double iOUThreshold = 0.5,
      int boxesLimit = 10,
      List<double> mean = noMeanRgb,
      List<double> std = noStdRgb}) async {
    List<ResultObjectDetection> prediction = await getCameraImagePredictionList(
        cameraImage, rotation,
        minimumScore: minimumScore,
        iOUThreshold: iOUThreshold,
        boxesLimit: boxesLimit,
        mean: mean,
        std: std);

    for (var element in prediction) {
      element.className = labels[element.classIndex];
    }

    return prediction;
  }

  /*

   */
  Widget renderBoxesOnImage(
      File image, List<ResultObjectDetection?> recognitions,
      {Color? boxesColor, bool showPercentage = true}) {
    //if (_recognitions == null) return Cont;
    //if (_imageHeight == null || _imageWidth == null) return [];

    //double factorX = screen.width;
    //double factorY = _imageHeight / _imageWidth * screen.width;
    //boxesColor ??= Color.fromRGBO(37, 213, 253, 1.0);

    // print(recognitions.length);
    return LayoutBuilder(builder: (context, constraints) {
      debugPrint(
          'Max height: ${constraints.maxHeight}, max width: ${constraints.maxWidth}');
      double factorX = constraints.maxWidth;
      double factorY = constraints.maxHeight;
      return Stack(
        children: [
          Positioned(
            left: 0,
            top: 0,
            width: factorX,
            height: factorY,
            child: Image.file(
              image,
              fit: BoxFit.fill,
            ),
          ),
          ...recognitions.map((re) {
            if (re == null) {
              return Container();
            }
            Color usedColor;
            if (boxesColor == null) {
              //change colors for each label
              usedColor = Colors.primaries[
                  ((re.className ?? re.classIndex.toString()).length +
                          (re.className ?? re.classIndex.toString())
                              .codeUnitAt(0) +
                          re.classIndex) %
                      Colors.primaries.length];
            } else {
              usedColor = boxesColor;
            }

            // print({
            //   "left": re.rect.left.toDouble() * factorX,
            //   "top": re.rect.top.toDouble() * factorY,
            //   "width": re.rect.width.toDouble() * factorX,
            //   "height": re.rect.height.toDouble() * factorY,
            // });
            return Positioned(
              left: re.rect.left * factorX,
              top: re.rect.top * factorY - 20,
              //width: re.rect.width.toDouble(),
              //height: re.rect.height.toDouble(),

              //left: re?.rect.left.toDouble(),
              //top: re?.rect.top.toDouble(),
              //right: re.rect.right.toDouble(),
              //bottom: re.rect.bottom.toDouble(),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                mainAxisAlignment: MainAxisAlignment.start,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    height: 20,
                    alignment: Alignment.centerRight,
                    color: usedColor,
                    child: Text(
                      "${re.className ?? re.classIndex.toString()}_${showPercentage ? "${(re.score * 100).toStringAsFixed(2)}%" : ""}",
                    ),
                  ),
                  Container(
                    width: re.rect.width.toDouble() * factorX,
                    height: re.rect.height.toDouble() * factorY,
                    decoration: BoxDecoration(
                        border: Border.all(color: usedColor, width: 3),
                        borderRadius:
                            const BorderRadius.all(Radius.circular(2))),
                    child: Container(),
                  ),
                ],
              ),
              /*
              Container(
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.all(Radius.circular(8.0)),
                  border: Border.all(
                    color: boxesColor!,
                    width: 2,
                  ),
                ),
                child: Text(
                  "${re.className ?? re.classIndex} ${(re.score * 100).toStringAsFixed(0)}%",
                  style: TextStyle(
                    background: Paint()..color = boxesColor!,
                    color: Colors.white,
                    fontSize: 12.0,
                  ),
                ),
              ),*/
            );
          }).toList()
        ],
      );
    });
  }
}
