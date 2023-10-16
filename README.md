# Ez-Image-Effects

<p align="center">
  <img src="https://github.com/Bassna/EZ-Image-Effects/assets/33616653/0b1d198a-0eb6-4d34-898a-8bd348c126c1" alt="FullIco"/>
</p>

A simple program to create easy animated effects from an image, and convert them into a set amount of exported frames.


## Table of Contents


- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)
- [Credits](#credits)
- [Support](#support)

## Features

- **Multiple Effect Types**: Many pre-set effects to choose from
- **Effect combine**: Effects can be combined together to create new, interesting effects! 
- **Export to X number frames**: Easily enter a set number of frames to export your animation into per effect.
- **Create simple text images**: Enter custom text and save it as a image to be used with effects. 

## Installation

### For Windows Users (.exe) Users:

1. **Download the Executable**: Navigate to the releases section of the GitHub repository and download the latest `EZ-Effects-Studio.exe`.
2. **Run the Executable**: Locate the downloaded `EZ-Effects-Studio.exe` file on your computer, and double-click on it to run the application. If prompted by Windows, click on "Run" or "Allow".

### For Python Script Users:

1. **Prerequisites**: Ensure you have Python installed on your system. If not, download and install it from [python.org](https://www.python.org/downloads/).
2. **Clone or Download the Repository**: Navigate to the main page of the GitHub repository and click on "Code" > "Download ZIP". Extract the ZIP to a location on your computer.
3. **Install Required Libraries**: 
   - Open a terminal or command prompt.
   - Navigate to the extracted repository folder.
   - Run the following command:
     ```
     pip install -r requirements.txt
     ```
   - _Optionally_, if you prefer not to use the `requirements.txt` file, you can manually install the required libraries using:
     ```
     pip install requests numpy Pillow
     ```
5. **Run the Script**: Once the required libraries are installed, you can run the script using the following command: python EZ-Image-Effects.py


## Usage

### 1. Selecting Effects:
- Choose the desired effect from the list.
- For more customization:
  - Combine multiple effects.
  - Adjust the "Effect Strength".
  - Adjust the "Number of Frames" for each effect. (Default is 48 frames.). **Combined effects use the highest set number of frames in the group**.
  - Access these options by expanding the submenu beneath the selected effect name.

### 2. Adding Effects to Queue:
- Click "Add Effect(s) to Queue".
- Upon selecting an effect, you'll be prompted to choose the image on which the effect will be applied.

### 3. Processing Effects Queue:
- Set up the order and combination of effects in the "Effects Queue".
- Preview your animation with "Preview Queue" before finalizing.
- To export the animation, click "Process Queue". The animation will be saved in the your selected folder.


## Screenshots


![NVIDIA_Share_yGQPQYGYzt](https://github.com/Bassna/EZ-Image-Effects/assets/33616653/62dc5ad7-5e1e-4cea-a71a-00e44358aa6a)




**Effects exported into folder as frames**


![explorer_q6dEjavV6l](https://github.com/Bassna/EZ-Image-Effects/assets/33616653/5dd0fbcc-8c74-405d-8417-cc6214690058)



**Create simple text images to use**

![NVIDIA_Share_KJKRKp0a0K](https://github.com/Bassna/EZ-Image-Effects/assets/33616653/b2f23e37-3cda-419d-bc17-d1af2e4da1cb)


## Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the GPL-3.0 License. See `LICENSE` for more information.

## Credits

- **Bassna** - Creator


## Support

Reach out to me at one of the following places!
- Discord: Bassna

---

