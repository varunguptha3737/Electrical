class AboutSection extends StatelessWidget {
  const AboutSection({super.key});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: const [
          Text(
            'About Us',
            style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
          ),
          SizedBox(height: 8),
          Text(
            'We are a professional electrical services company with over 10 years of experience. Our goal is to provide safe and efficient electrical solutions for residential, commercial, and industrial clients.',
            style: TextStyle(fontSize: 16),
          ),
        ],
      ),
    );
  }
}
