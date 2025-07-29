import 'package:flutter/material.dart';
import 'screens/home_screen.dart';

void main() {
  runApp(DigitRecognitionApp());
}

class DigitRecognitionApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Digit Recognition App',
      home: HomeScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}
