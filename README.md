
# 🎰 Lottery-Predict: Your Ticket to Winning Big! 🎉

Welcome to **Lottery-Predict**, the ultimate web app that turns your lottery dreams into reality! Upload your past lottery data, and let our intelligent algorithms predict your next winning numbers with style and precision. Ready to strike it lucky? 🍀

## 🌟 What is Lottery-Predict?

Lottery-Predict is a modern web application built with Flask, designed to analyze historical lottery data from Excel files and forecast future numbers using machine learning magic. With a stunning interface, real-time analysis, and animated predictions, this tool makes lottery prediction both fun and insightful!

---

## 🚀 Features That Shine

- **📊 Easy Excel Upload**: Drag and drop your `.xlsx` file with past lottery draws.
- **⚡ Real-Time Insights**: Watch the analysis unfold live with Server-Sent Events.
- **🎲 Animated Predictions**: See your predicted numbers pop up in glowing "lucky balls".
- **📥 Sample File Included**: Download a sample file to get started instantly.
- **📱 Responsive Design**: Looks amazing on both desktop and mobile devices.

---

## 🛠️ Installation Guide

Get up and running in just a few steps!

### Prerequisites
- Python 3.8 or higher 🐍
- pip (Python package manager)

### Steps to Success
1. **Clone the Repo**
   ```bash
   git clone https://github.com/Ishanoshada/lottery-predict.git
   cd lottery-predict
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up the Sample File**
   - Make sure `api/static/previous_data.xlsx` is ready (download it from the app or create it manually).
   - Sample format:
     ```
     Date       | Num1 | Num2 | Num3 | Num4 | Num5 | Num6 | Bonus
     2025-May-30| 37   | 51   | 79   |      |      |      | E
     2025-May-09| 25   | 72   | 77   |      |      |      | Q
     2025-May-22| 42   | 71   | 73   |      |      |      | Q
     ...
     ```

4. **Launch the App**
   ```bash
   python api/app.py
   ```
   - Open `http://localhost:5000` in your browser and start predicting! 🎯

---

## 🎮 How to Use Lottery-Predict

1. **Upload Your Data**
   - Drag and drop your `.xlsx` file into the upload area or click to browse.
   - Need a template? Click "Sample File Download" to grab a ready-made file.

2. **Analyze & Predict**
   - Hit the "Analyze & Predict" button to kick off the magic.
   - Watch the live updates and see your predicted numbers appear in glowing balls!

3. **Check the Sample Data**
   - Click "Show Sample Data" to see the expected Excel format and get inspired.

---

## 📋 File Format Requirements

Your Excel file should follow this structure:
- **Date**: `YYYY-MMM-DD` (e.g., `2025-May-30`)
- **Num1 to Num6**: Lottery numbers (up to 6 columns; empty cells are fine)
- **Bonus**: Single letter (optional, e.g., `E`, `Q`)
- File type: `.xlsx`

---

## 🤝 Contribute to the Magic

We’d love your help to make Lottery-Predict even better! Here’s how to contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-amazing-idea`).
3. Commit your changes (`git commit -m "Add a cool feature"`).
4. Push to the branch (`git push origin feature-amazing-idea`).
5. Open a Pull Request with a clear description of your changes.

Please follow the existing code style and add tests where possible.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgments

- A big shoutout to [Ishanoshada](https://github.com/Ishanoshada/) for crafting this awesome project! 👏
- Powered by Flask, pandas, and scikit-learn.
- Thanks to Font Awesome for the cool icons and the open-source community for endless inspiration.

---

## 📬 Get in Touch

Got questions or ideas? Open an issue on GitHub or reach out to the maintainer at [ic31908@gmail.com](mailto:ic31908@gmail.com).

Let’s predict those winning numbers together! 💰

![Views](https://dynamic-repo-badges.vercel.app/svg/count/6/Repository%20Views/lottery-predict)