import 'package:flutter/material.dart';
import 'drawing_canvas_screen.dart';
import 'upload_image_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: getBody(context),
    );
  }

  Widget getBody(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.edit, size: 32, color: Colors.blueAccent),
                SizedBox(width: 10),
                Text(
                  'Digit Recognition App',
                  style: TextStyle(fontSize: 26, fontWeight: FontWeight.bold),
                ),
              ],
            ),
            const SizedBox(height: 10),
            Text(
              '🔢 Predict single and multi digit',
              style: TextStyle(fontSize: 16, color: Colors.grey[700]),
            ),
            const SizedBox(height: 40),
            buildOptionButton(context, 'Draw Digit', Icons.brush, const DrawingCanvasScreen()),
            buildOptionButton(context, 'Upload Image', Icons.image, const ImageUploadScreen()),
          ],
        ),
      ),
    );
  }

  Widget buildOptionButton(BuildContext context, String label, IconData icon, Widget screen) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 10),
      child: SizedBox(
        width: 240,
        height: 50,
        child: ElevatedButton.icon(
          onPressed: () {
            Navigator.push(context, MaterialPageRoute(builder: (_) => screen));
          },
          icon: Icon(icon, size: 22),
          label: Text(label),
          style: ElevatedButton.styleFrom(
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            textStyle: const TextStyle(fontSize: 18),
          ),
        ),
      ),
    );
  }

}
