import 'package:flutter/material.dart';

void main() {
  runApp(const ElectricalBusinessWebsite());
}

class ElectricalBusinessWebsite extends StatelessWidget {
  const ElectricalBusinessWebsite({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Electrical Business',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Electrical Business'),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: const [
            HeroSection(),
            ServicesSection(),
            AboutSection(),
            ContactSection(),
          ],
        ),
      ),
    );
  }
}
