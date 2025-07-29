import 'dart:io';
import 'package:dio/dio.dart';

class ApiService {
  static String baseUrl = 'http://test0.gpstrack.in:9009';

  // api_service.dart
  static Future<Map<String, dynamic>> predictDigit(File imageFile) async {
    try {
      MultipartFile multipartFile = await MultipartFile.fromFile(
        imageFile.path,
        filename: 'digit.png',
      );

      FormData formData = FormData.fromMap({'file': multipartFile});
      Response response = await Dio().post('$baseUrl/predict', data: formData);

      if (response.statusCode == 200 && response.data != null) {
        String prediction = response.data['prediction'].toString();
        double avgConfidence = 0;

        if (response.data['type'] == 'multi') {
          List<dynamic>? confList = response.data['confidences'];
          if (confList != null && confList.isNotEmpty) {
            avgConfidence = confList
                .map((e) => (e as num).toDouble())
                .reduce((a, b) => a + b) /
                confList.length;
          }
        } else if (response.data['type'] == 'single') {
          avgConfidence = (response.data['confidence'] as num).toDouble();
        }

        return {
          'prediction': prediction,
          'confidence': avgConfidence.toStringAsFixed(2),
        };
      }

      return {'prediction': 'Prediction failed'};
    } catch (e) {
      return {'prediction': 'Error: $e'};
    }
  }


}

