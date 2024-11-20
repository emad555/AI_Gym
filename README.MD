
# AI-Powered Dumbbell Counter with Pose Detection

This project is an **AI-powered dumbbell counter** that tracks your arm movements in real-time, counts repetitions, and even resets when you open your arms wide. Built using **Raspberry Pi**, **OpenCV**, and **MediaPipe**, it ensures accurate tracking to supercharge your workouts!

---

## Features

- **Real-time Pose Detection**: Tracks arm movements using AI-powered pose detection.
- **Accurate Repetition Counter**: Increments the count as you lift your dumbbell.
- **Dual Arm Tracking**: Works for both left and right arms independently.
- **Auto Reset**: Resets the count when you open your arms wide.
- **Customizable**: Easily adjust thresholds and parameters for your workout needs.

---

## Requirements

### Hardware
- Raspberry Pi (tested on Raspberry Pi 4 or higher)
- Camera (USB webcam or Raspberry Pi Camera Module)
- Monitor, keyboard, and mouse for setup

### Software
- Python 3.7 or higher
- OpenCV
- Mediapipe

---

## Installation

1. **Install Python Dependencies**:
   ```bash
   pip install opencv-python mediapipe numpy
   ```

2. **Clone the Repository**:
   ```bash
   git clone https://github.com/dumbbell-counter
   cd dumbbell-counter
   ```

3. **Run the Script**:
   ```bash
   python dumbbell_counter.py
   ```

---

## Usage

1. **Setup**: Place your camera at a height where your entire upper body is visible.
2. **Start the Program**: Run the script and follow the instructions on the screen.
3. **Perform Dumbbell Lifts**:
   - Lift with either your left or right arm to see the counter increment.
   - Open your arms wide to reset the counter.
4. **Exit**: Press `q` to exit the program.

---

## Project Structure

```
├── dumbbell_counter.py      # Main Python script for the project
├── README.md                # Project documentation
```

---

## Demonstration

Check out this [demo video](#) to see the AI-powered dumbbell counter in action!

---

## Customization

- **Thresholds**:
  - Adjust the `arm_open_threshold` to fine-tune the distance required to reset the counter.
  - Modify angle thresholds for arm tracking as per your preferences.

---

## Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request with your improvements.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- [OpenCV](https://opencv.org/)
- [MediaPipe](https://mediapipe.dev/)
- Inspiration from AI-powered fitness tracking projects.