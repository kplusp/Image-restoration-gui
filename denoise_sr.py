import os
import argparse
import torch
import numpy as np
import cv2
import glob
from models.network_swinir import SwinIR
from utils import util_calculate_psnr_ssim
from collections import OrderedDict

class DenoiseSRApp:
    def __init__(self, dn_model_path, sr_model_path, device='cuda'):
        """Initialize the application with denoising and SR models"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Setup models
        print("Loading denoising model...")
        self.dn_model = self.setup_model('color_dn', dn_model_path)
        
        print("Loading super-resolution model...")
        self.sr_model = self.setup_model('classical_sr', sr_model_path)
    
    def setup_model(self, task, model_path):
        """Setup SwinIR model based on task"""
        if task == 'color_dn':  # Color image denoising
            model = SwinIR(upscale=1, in_chans=3, img_size=128, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='', resi_connection='1conv')
        elif task == 'classical_sr':  # Classical image super-resolution
            model = SwinIR(upscale=4, in_chans=3, img_size=48, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        
        # Load pre-trained model
        pretrained_model = torch.load(model_path, map_location=self.device)
        param_key_g = 'params' if 'params' in pretrained_model.keys() else 'params_ema'
        model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
        
        model.eval()
        model = model.to(self.device)
        return model

    def process_image(self, img, model, task, scale=4):
        """Process an image with SwinIR"""
        # Convert to tensor
        if isinstance(img, np.ndarray):
            img = np.transpose(img, (2, 0, 1))  # HWC to CHW
            img = torch.from_numpy(img).float().unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            # Pad input if needed
            _, _, h_old, w_old = img.size()
            h_pad = (h_old // 8 + 1) * 8 - h_old if h_old % 8 != 0 else 0
            w_pad = (w_old // 8 + 1) * 8 - w_old if w_old % 8 != 0 else 0
            img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_old + h_pad, :]
            img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_old + w_pad]
            
            output = model(img)
            
            # Remove padding
            if task == 'classical_sr':
                output = output[..., :h_old * scale, :w_old * scale]
            else:
                output = output[..., :h_old, :w_old]
        
        # Convert to numpy
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output, (1, 2, 0))  # CHW -> HWC
        
        return output

    def denoise_and_sr(self, img_path):
        """Run denoising followed by super-resolution"""
        # Read image
        img_noisy_lr = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        
        # First step: Denoising
        print("Step 1: Denoising...")
        img_denoised = self.process_image(img_noisy_lr, self.dn_model, 'color_dn', scale=1)
        
        # Second step: Super-resolution
        print("Step 2: Super-resolution...")
        img_denoised_sr = self.process_image(img_denoised, self.sr_model, 'classical_sr', scale=4)
        
        return {
            'original': img_noisy_lr,
            'denoised': img_denoised,
            'final': img_denoised_sr
        }

    def save_results(self, results, original_path, output_dir):
        """Save all results to the output directory"""
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        
        # Save denoised image
        denoised_path = os.path.join(output_dir, f"{base_name}_denoised.png")
        denoised_img = (results['denoised'] * 255.0).round().astype(np.uint8)
        cv2.imwrite(denoised_path, denoised_img)
        
        # Save final denoised + super-resolved image
        final_path = os.path.join(output_dir, f"{base_name}_denoised_sr.png")
        final_img = (results['final'] * 255.0).round().astype(np.uint8)
        cv2.imwrite(final_path, final_img)
        
        return {
            'denoised_path': denoised_path,
            'final_path': final_path
        }

    def process_directory(self, input_dir, output_dir):
        """Process all images in a directory"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all test images
        test_images = sorted(glob.glob(os.path.join(input_dir, '*.*')))
        
        # Process each image
        for img_path in test_images:
            print(f"\nProcessing {os.path.basename(img_path)}...")
            
            # Run denoising and super-resolution
            results = self.denoise_and_sr(img_path)
            
            # Save results
            paths = self.save_results(results, img_path, output_dir)
            
            print(f"Saved denoised image to: {paths['denoised_path']}")
            print(f"Saved final denoised + super-resolved image to: {paths['final_path']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dn_model_path', type=str, required=True, help='path to pre-trained denoising model')
    parser.add_argument('--sr_model_path', type=str, required=True, help='path to pre-trained super-resolution model')
    parser.add_argument('--input_dir', type=str, required=True, help='input test image folder')
    parser.add_argument('--output_dir', type=str, required=True, help='output image folder')
    parser.add_argument('--device', type=str, default='cuda', help='device to use (cuda or cpu)')
    args = parser.parse_args()
    
    # Create and run the app
    app = DenoiseSRApp(
        dn_model_path=args.dn_model_path, 
        sr_model_path=args.sr_model_path, 
        device=args.device
    )
    
    app.process_directory(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()