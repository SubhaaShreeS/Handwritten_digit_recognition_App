import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/api_service.dart';

class ImageUploadScreen extends StatefulWidget {
  const ImageUploadScreen({super.key});

  @override
  State<ImageUploadScreen> createState() => ImageUploadScreenState();
}

class ImageUploadScreenState extends State<ImageUploadScreen> {
  File? selectedImage;
  String prediction = '';
  String confidence = '';

  Future<void> pickImage(ImageSource source) async {
    XFile? image = await ImagePicker().pickImage(source: source);
    if (image != null) {
      File file = File(image.path);
      Map result = await ApiService.predictDigit(file);
      setState(() {
        selectedImage = file;
        prediction = result['prediction'];
        confidence = result['confidence'] ?? '';
      });
    }
  }

  Widget buildImageView() {
    return Container(
      width: 350,
      height: 350,
      decoration: BoxDecoration(
        border: Border.all(color: Colors.grey),
        borderRadius: BorderRadius.circular(12),
        color: Colors.grey[200],
      ),
      child: selectedImage != null
          ? ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: Image.file(selectedImage!, fit: BoxFit.cover),
      )
          : Center(child: Text('No image selected')),
    );
  }

  Widget buildButtons() {
    return Wrap(
      spacing: 16,
      runSpacing: 10,
      alignment: WrapAlignment.center,
      children: [
        buildButton('Pick from Gallery', () => pickImage(ImageSource.gallery)),
        buildButton('Capture from\nCamera', () => pickImage(ImageSource.camera)),
      ],
    );
  }

  Widget buildButton(String label, VoidCallback onPressed) {
    return SizedBox(
      width: 160,
      height: 45,
      child: ElevatedButton(
        onPressed: onPressed,
        style: ElevatedButton.styleFrom(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
          textStyle: TextStyle(fontSize: 16),
        ),
        child: Text(label, textAlign: TextAlign.center,),

      ),
    );
  }

  Widget buildPredictionDisplay() {
    return Column(
      children: [
        SizedBox(height: 20),
        if (prediction.isNotEmpty)
          Text('📈 Prediction: $prediction',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
        if (confidence.isNotEmpty)
          Text('Confidence: $confidence',
              style: TextStyle(fontSize: 16, color: Colors.grey[700])),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return getScaffold();
  }

  Scaffold getScaffold() {
    return Scaffold(
    appBar: AppBar(title: Text('Upload Digit Image')),
    body: Padding(
      padding: EdgeInsets.symmetric(horizontal: 24),
      child: Center(
        child: SingleChildScrollView(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              buildImageView(),
              SizedBox(height: 30),
              buildButtons(),
              buildPredictionDisplay(),
            ],
          ),
        ),
      ),
    ),
  );
  }
}
