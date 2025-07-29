import 'dart:typed_data';
import 'dart:ui' as ui;
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:path_provider/path_provider.dart';
import '../services/api_service.dart';

class DrawingCanvasScreen extends StatefulWidget {
  const DrawingCanvasScreen({super.key});

  @override
  State<DrawingCanvasScreen> createState() => DrawingCanvasScreenState();
}

class DrawingCanvasScreenState extends State<DrawingCanvasScreen> {
  GlobalKey canvasKey = GlobalKey();
  List<Offset> points = [];
  String prediction = '';
  String confidence = '';

  void clearCanvas() {
    setState(() {
      points.clear();
      prediction = '';
      confidence = '';
    });
  }

  void undoLastStroke() {
    setState(() {
      if (points.isEmpty) return;

      // Remove last stroke (from end back to previous infinite)
      int index = points.lastIndexWhere((point) => point.dx.isInfinite);
      if (index == -1) {
        points.clear();
      } else {
        points = points.sublist(0, index);
      }
    });
  }

  Future<void> predictDigit() async {
    RenderRepaintBoundary? boundary =
        canvasKey.currentContext?.findRenderObject() as RenderRepaintBoundary?;

    if (boundary != null) {
      ui.Image image = await boundary.toImage(pixelRatio: 3.0);
      ByteData? byteData =
          await image.toByteData(format: ui.ImageByteFormat.png);

      if (byteData != null) {
        Uint8List pngBytes = byteData.buffer.asUint8List();
        Directory tempDir = await getTemporaryDirectory();
        String filePath = '${tempDir.path}/digit.png';
        File imageFile = File(filePath);
        await imageFile.writeAsBytes(pngBytes);

        Map result = await ApiService.predictDigit(imageFile);
        setState(() {
          prediction = result['prediction'];
          confidence = result['confidence'] ?? '';
        });
      }
    }
  }

  Widget buildCanvas() {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: RepaintBoundary(
        key: canvasKey,
        child: Container(
          width: 350,
          height: 350,
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(12),
          ),
          child: CustomPaint(painter: CanvasPainter(points)),
        ),
      ),
    );
  }

  Widget buildButtons() {
    return Wrap(
      spacing: 16,
      runSpacing: 10,
      alignment: WrapAlignment.center,
      children: [
        buildButton(Icons.search, 'Predict', predictDigit),
        buildButton(Icons.clear, 'Clear', clearCanvas),
        buildButton(Icons.undo, 'Undo', undoLastStroke),
      ],
    );
  }

  Widget buildButton(IconData icon, String label, VoidCallback onPressed) {
    return SizedBox(
      width: 110,
      height: 45,
      child: ElevatedButton(
        onPressed: onPressed,
        style: ElevatedButton.styleFrom(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
          textStyle: TextStyle(fontSize: 16),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, size: 20),
            SizedBox(width: 6),
            Text(label),
          ],
        ),
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

  Widget buildDrawingArea() {
    return GestureDetector(
      onPanStart: (details) {
        setState(() {
          // Add a separator to start a new stroke
          points.add(Offset.infinite);
          points.add(details.localPosition);
        });
      },
      onPanUpdate: (details) {
        setState(() {
          points.add(details.localPosition);
        });
      },
      onPanEnd: (_) {
        setState(() {
          points.add(Offset.infinite);
        });
      },
      child: buildCanvas(),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Draw Digit')),
      body: Padding(
        padding: EdgeInsets.symmetric(horizontal: 24, vertical: 20),
        child: Column(
          children: [
            Expanded(
              child: Center(
                child: buildDrawingArea(),
              ),
            ),
            SizedBox(height: 30),
            buildButtons(),
            buildPredictionDisplay(),
          ],
        ),
      ),
    );
  }
}

class CanvasPainter extends CustomPainter {
  List<Offset> points;
  CanvasPainter(this.points);

  @override
  void paint(Canvas canvas, Size size) {
    Paint paint = Paint()
      ..color = Colors.black
      ..strokeCap = StrokeCap.round
      ..strokeWidth = 10
      ..isAntiAlias = true;

    canvas.saveLayer(Rect.fromLTWH(0, 0, size.width, size.height), Paint());
    canvas.drawColor(Colors.white, BlendMode.src);

    for (int i = 0; i < points.length - 1; i++) {
      if (!points[i].dx.isInfinite && !points[i + 1].dx.isInfinite) {
        canvas.drawLine(points[i], points[i + 1], paint);
      }
    }

    canvas.restore();
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => true;
}
