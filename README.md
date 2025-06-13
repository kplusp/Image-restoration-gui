# SwinIR Image Enhancement Application

## Project Overview

This application provides advanced image restoration using SwinIR (Swin Transformer for Image Restoration) models for denoising and super-resolution. Developed as a collaborative project by Kevin Paul, Dhruv Maheshwari, Harsh Bhawnani, and Moiz Sheikh, the application offers a user-friendly GUI to enhance image quality.

## Features

- Image Denoising
- Super-Resolution (4x upscaling)
- Modern, intuitive graphical interface
- Supports CUDA and CPU processing
- Customizable model selection

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended, but not required)
- PyTorch with CUDA support (optional)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/swinir-image-enhancement.git
cd swinir-image-enhancement
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download Pre-trained Models:
Place the following pre-trained models in the `./models/` directory:
- Denoising model: `005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth`
- Super-Resolution model: `001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth`

## Usage

Run the application:
```bash
python gui_app.py
```

### GUI Instructions

1. **Input Image**:
   - Click "Browse" to select an input image
   - Supported formats: JPG, JPEG, PNG, BMP

2. **Output Folder**:
   - Select a directory to save enhanced images
   - Default is `./results/`

3. **Model Selection**:
   - Pre-configured models are selected by default
   - Use "Browse" to select custom models (`.pth` files)

4. **Processing Device**:
   - Choose between CUDA (GPU) and CPU
   - CUDA recommended for faster processing

5. **Enhance Image**:
   - Click "Enhance Image" to start processing
   - View previews of original, denoised, and enhanced images

## Output

The application generates two images in the selected output directory:
- `[filename]_denoised.png`: Denoised image
- `[filename]_enhanced.png`: Super-resolved image

## Model Details

The application uses two SwinIR models:
- Denoising Model: Reduces noise in images
- Super-Resolution Model: Upscales image resolution by 4x

## Troubleshooting

- Ensure all dependencies are installed
- Check that model files are in the correct directory
- Verify image file compatibility

## Contributing

Contributions are welcome! Please submit pull requests or open issues on the GitHub repository.

## License

[Specify your project's license]

## Acknowledgments

- Swin Transformer Team
- Original SwinIR Paper Authors
- Project Contributors: Kevin Paul, Dhruv Maheshwari, Harsh Bhawnani, Moiz Sheikh