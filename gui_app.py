import os
import tkinter as tk
from tkinter import filedialog, ttk
import threading
import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
import sys
from pathlib import Path

# Add the SwinIR directory to the path
sys.path.append('.')  # Adjust if needed
from models.network_swinir import SwinIR

class ModernTheme:
    """Modern color theme and styling for the application"""
    BG_COLOR = "#f5f5f7"
    FG_COLOR = "#1d1d1f"
    ACCENT_COLOR = "#0071e3"
    SECONDARY_COLOR = "#86868b"
    CARD_BG = "#ffffff"
    CARD_BORDER = "#e6e6e6"
    
    @classmethod
    def apply_theme(cls, root):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure common styles
        style.configure('TFrame', background=cls.BG_COLOR)
        style.configure('TLabel', background=cls.BG_COLOR, foreground=cls.FG_COLOR)
        style.configure('TLabelframe', background=cls.BG_COLOR, foreground=cls.FG_COLOR)
        style.configure('TLabelframe.Label', background=cls.BG_COLOR, foreground=cls.FG_COLOR, font=('Helvetica', 10, 'bold'))
        
        # Configure button styles
        style.configure('TButton', background=cls.ACCENT_COLOR, foreground='white', borderwidth=0)
        style.map('TButton', background=[('active', '#005bbf'), ('disabled', '#86868b')])
        
        # Configure entry styles
        style.configure('TEntry', fieldbackground=cls.CARD_BG, borderwidth=1, relief='solid')
        
        # Configure progress bar
        style.configure('TProgressbar', background=cls.ACCENT_COLOR, borderwidth=0, troughcolor='#e6e6e6')
        
        # Configure root window
        root.configure(background=cls.BG_COLOR)
        
        # Card style
        style.configure('Card.TFrame', background=cls.CARD_BG, relief='solid', borderwidth=1)
        style.configure('Card.TLabel', background=cls.CARD_BG)
        
        # Section header style
        style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'), background=cls.BG_COLOR)
        
        # Primary button
        style.configure('Primary.TButton', font=('Helvetica', 10, 'bold'))
        
        # Status style
        style.configure('Status.TLabel', foreground=cls.SECONDARY_COLOR)
        
        return style

class DenoiseSRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Restoration Using SwinIR Transformer")
        self.root.geometry("1200x800")
        self.root.minsize(900, 700)
        
        # Apply theme
        self.style = ModernTheme.apply_theme(root)
        
        # Variables
        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path("./results").absolute()))
        self.dn_model_path = tk.StringVar(value=str(Path("./models/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth").absolute()))
        self.sr_model_path = tk.StringVar(value=str(Path("./models/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth").absolute()))
        self.device = tk.StringVar(value="cuda" if torch.cuda.is_available() else "cpu")
        
        # Create main container
        main_container = ttk.Frame(root)
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Label(header_frame, text="Image Restoration with SwinIR", font=('Helvetica', 16, 'bold')).pack(side="left")
        
        # Left panel (settings)
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side="left", fill="y", padx=(0, 15))
        
        # Input settings card
        input_card = ttk.Frame(left_panel, style='Card.TFrame')
        input_card.pack(fill="x", pady=(0, 15), ipadx=15, ipady=15)
        
        ttk.Label(input_card, text="Image Input/Output", style='Header.TLabel').grid(row=0, column=0, columnspan=3, sticky="w", padx=15, pady=(15, 10))
        
        ttk.Label(input_card, text="Input Image:", style='Card.TLabel').grid(row=1, column=0, padx=15, pady=(10, 5), sticky="w")
        input_entry = ttk.Entry(input_card, textvariable=self.input_path, width=30)
        input_entry.grid(row=1, column=1, padx=5, pady=(10, 5), sticky="ew")
        ttk.Button(input_card, text="Browse", command=self.browse_input, width=10).grid(row=1, column=2, padx=15, pady=(10, 5))
        
        ttk.Label(input_card, text="Output Folder:", style='Card.TLabel').grid(row=2, column=0, padx=15, pady=5, sticky="w")
        output_entry = ttk.Entry(input_card, textvariable=self.output_dir, width=30)
        output_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(input_card, text="Browse", command=self.browse_output, width=10).grid(row=2, column=2, padx=15, pady=5)
        
        # Model settings card
        model_card = ttk.Frame(left_panel, style='Card.TFrame')
        model_card.pack(fill="x", ipadx=15, ipady=15)
        
        ttk.Label(model_card, text="Model Configuration", style='Header.TLabel').grid(row=0, column=0, columnspan=3, sticky="w", padx=15, pady=(15, 10))
        
        ttk.Label(model_card, text="Denoising Model:", style='Card.TLabel').grid(row=1, column=0, padx=15, pady=(10, 5), sticky="w")
        ttk.Entry(model_card, textvariable=self.dn_model_path, width=30).grid(row=1, column=1, padx=5, pady=(10, 5), sticky="ew")
        ttk.Button(model_card, text="Browse", command=lambda: self.browse_model('dn'), width=10).grid(row=1, column=2, padx=15, pady=(10, 5))
        
        ttk.Label(model_card, text="Super-Resolution:", style='Card.TLabel').grid(row=2, column=0, padx=15, pady=5, sticky="w")
        ttk.Entry(model_card, textvariable=self.sr_model_path, width=30).grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(model_card, text="Browse", command=lambda: self.browse_model('sr'), width=10).grid(row=2, column=2, padx=15, pady=5)
        
        ttk.Label(model_card, text="Processing Device:", style='Card.TLabel').grid(row=3, column=0, padx=15, pady=5, sticky="w")
        device_combo = ttk.Combobox(model_card, textvariable=self.device, values=["cuda", "cpu"], width=10, state="readonly")
        device_combo.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        
        # Action button
        action_frame = ttk.Frame(left_panel)
        action_frame.pack(fill="x", pady=15)
        
        self.status_var = tk.StringVar(value="Ready")
        self.process_btn = ttk.Button(action_frame, text="Enhance Image", command=self.start_processing, style='Primary.TButton')
        self.process_btn.pack(side="bottom", fill="x", pady=5, ipady=5)
        
        # Progress section
        progress_frame = ttk.Frame(left_panel)
        progress_frame.pack(fill="x")
        
        self.progress = ttk.Progressbar(progress_frame, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", pady=(0, 5))
        
        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var, style='Status.TLabel')
        self.status_label.pack(anchor="w")
        
        # Configure grid weights for the settings panels
        for i in range(3):
            model_card.columnconfigure(i, weight=1)
            input_card.columnconfigure(i, weight=1)
        
        # Right panel (preview)
        right_panel = ttk.Frame(main_container, style='Card.TFrame')
        right_panel.pack(side="right", fill="both", expand=True)
        
        preview_header = ttk.Label(right_panel, text="Image Preview", style='Header.TLabel')
        preview_header.pack(anchor="w", padx=15, pady=15)
        
        # Preview container with scroll support
        preview_container = ttk.Frame(right_panel, style='Card.TFrame')
        preview_container.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        # Canvas with scrollbar for previews
        canvas = tk.Canvas(preview_container, bg=ModernTheme.CARD_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(preview_container, orient="horizontal", command=canvas.xview)
        scrollable_frame = ttk.Frame(canvas, style='Card.TFrame')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(xscrollcommand=scrollbar.set)
        
        canvas.pack(side="top", fill="both", expand=True)
        scrollbar.pack(side="bottom", fill="x")
        
        # Preview images
        self.preview_size = 320  # Fixed preview size
        
        # Input preview
        input_preview_frame = ttk.Frame(scrollable_frame, style='Card.TFrame')
        input_preview_frame.grid(row=0, column=0, padx=10, pady=10)
        
        ttk.Label(input_preview_frame, text="Original Image", style='Card.TLabel').pack(pady=(5, 10))
        self.input_preview = ttk.Label(input_preview_frame, style='Card.TLabel')
        self.input_preview.pack(padx=10, pady=(0, 10))
        
        # Denoised preview
        denoised_preview_frame = ttk.Frame(scrollable_frame, style='Card.TFrame')
        denoised_preview_frame.grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Label(denoised_preview_frame, text="Denoised Image", style='Card.TLabel').pack(pady=(5, 10))
        self.denoised_preview = ttk.Label(denoised_preview_frame, style='Card.TLabel')
        self.denoised_preview.pack(padx=10, pady=(0, 10))
        
        # Final preview
        final_preview_frame = ttk.Frame(scrollable_frame, style='Card.TFrame')
        final_preview_frame.grid(row=0, column=2, padx=10, pady=10)
        
        ttk.Label(final_preview_frame, text="Enhanced Image", style='Card.TLabel').pack(pady=(5, 10))
        self.final_preview = ttk.Label(final_preview_frame, style='Card.TLabel')
        self.final_preview.pack(padx=10, pady=(0, 10))
        
        # Initialize models
        self.dn_model = None
        self.sr_model = None
        
        # Set minimum size for left panel
        left_panel.update()
        left_panel.config(width=left_panel.winfo_reqwidth())
    
    def browse_input(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.input_path.set(path)
            self.update_input_preview()
    
    def browse_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)
    
    def browse_model(self, model_type):
        path = filedialog.askopenfilename(filetypes=[("PyTorch models", "*.pth")])
        if path:
            if model_type == 'dn':
                self.dn_model_path.set(path)
            else:
                self.sr_model_path.set(path)
    
    def update_input_preview(self):
        try:
            img = Image.open(self.input_path.get())
            img.thumbnail((self.preview_size, self.preview_size))
            img_tk = ImageTk.PhotoImage(img)
            self.input_preview.configure(image=img_tk)
            self.input_preview.image = img_tk
        except Exception as e:
            self.status_var.set(f"Error loading preview: {e}")
    
    def update_output_previews(self, denoised_path, final_path):
        try:
            # Update denoised preview
            img_denoised = Image.open(denoised_path)
            img_denoised.thumbnail((self.preview_size, self.preview_size))
            img_denoised_tk = ImageTk.PhotoImage(img_denoised)
            self.denoised_preview.configure(image=img_denoised_tk)
            self.denoised_preview.image = img_denoised_tk
            
            # Update final preview
            img_final = Image.open(final_path)
            img_final.thumbnail((self.preview_size, self.preview_size))
            img_final_tk = ImageTk.PhotoImage(img_final)
            self.final_preview.configure(image=img_final_tk)
            self.final_preview.image = img_final_tk
        except Exception as e:
            self.status_var.set(f"Error loading output previews: {e}")
    
    def setup_models(self):
        device_str = self.device.get()
        device = torch.device(device_str)
        
        # Setup denoising model
        self.status_var.set("Loading denoising model...")
        self.progress['value'] = 10
        self.root.update_idletasks()
        
        dn_model = SwinIR(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        
        dn_pretrained = torch.load(self.dn_model_path.get(), map_location=device)
        param_key_dn = 'params' if 'params' in dn_pretrained.keys() else 'params_ema'
        dn_model.load_state_dict(dn_pretrained[param_key_dn] if param_key_dn in dn_pretrained.keys() else dn_pretrained, strict=True)
        dn_model.eval()
        dn_model = dn_model.to(device)
        
        # Setup super-resolution model
        self.status_var.set("Loading super-resolution model...")
        self.progress['value'] = 20
        self.root.update_idletasks()
        
        sr_model = SwinIR(upscale=4, in_chans=3, img_size=48, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        
        sr_pretrained = torch.load(self.sr_model_path.get(), map_location=device)
        param_key_sr = 'params' if 'params' in sr_pretrained.keys() else 'params_ema'
        sr_model.load_state_dict(sr_pretrained[param_key_sr] if param_key_sr in sr_pretrained.keys() else sr_pretrained, strict=True)
        sr_model.eval()
        sr_model = sr_model.to(device)
        
        return dn_model, sr_model
    
    def process_image_thread(self):
        try:
            # Setup output directory
            os.makedirs(self.output_dir.get(), exist_ok=True)
            
            # Setup device
            device_str = self.device.get()
            device = torch.device(device_str)
            
            # Setup models if not already loaded
            if self.dn_model is None or self.sr_model is None:
                self.dn_model, self.sr_model = self.setup_models()
            
            # Read image
            img_path = self.input_path.get()
            img_noisy_lr = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            
            # First step: Denoising
            self.status_var.set("Step 1: Denoising image...")
            self.progress['value'] = 30
            self.root.update_idletasks()
            
            # Process with denoising model
            img_denoised = self.process_model_image(self.dn_model, img_noisy_lr, 'color_dn', device)
            
            self.progress['value'] = 50
            self.root.update_idletasks()
            
            # Second step: Super-resolution
            self.status_var.set("Step 2: Enhancing resolution...")
            
            # Process with super-resolution model
            img_denoised_sr = self.process_model_image(self.sr_model, img_denoised, 'classical_sr', device)
            
            self.progress['value'] = 70
            self.root.update_idletasks()
            
            # Save results
            self.status_var.set("Saving enhanced images...")
            
            # Get base filename
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Save denoised image
            denoised_path = os.path.join(self.output_dir.get(), f"{base_name}_denoised.png")
            denoised_img = (img_denoised * 255.0).round().astype(np.uint8)
            cv2.imwrite(denoised_path, denoised_img)
            
            # Save final denoised + super-resolved image
            final_path = os.path.join(self.output_dir.get(), f"{base_name}_enhanced.png")
            final_img = (img_denoised_sr * 255.0).round().astype(np.uint8)
            cv2.imwrite(final_path, final_img)
            
            self.progress['value'] = 90
            self.root.update_idletasks()
            
            # Update previews
            self.root.after(0, lambda: self.update_output_previews(denoised_path, final_path))
            self.root.after(0, lambda: self.status_var.set(f"Complete! Saved to {self.output_dir.get()}"))
            self.root.after(0, lambda: self.progress.configure(value=100))
            
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
            self.root.after(0, lambda: self.progress.configure(value=0))
    
    def process_model_image(self, model, img, task, device):
        """Process an image with a SwinIR model"""
        # Convert to tensor
        if isinstance(img, np.ndarray):
            img = np.transpose(img, (2, 0, 1))  # HWC to CHW
            img = torch.from_numpy(img).float().unsqueeze(0).to(device)
        
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
                output = output[..., :h_old * 4, :w_old * 4]  # Scale factor 4
            else:
                output = output[..., :h_old, :w_old]
        
        # Convert to numpy
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output, (1, 2, 0))  # CHW -> HWC
        
        return output
    
    def start_processing(self):
        # Validate inputs
        if not self.input_path.get():
            self.status_var.set("Error: No input image selected")
            return
            
        if not self.dn_model_path.get():
            self.status_var.set("Error: No denoising model selected")
            return
            
        if not self.sr_model_path.get():
            self.status_var.set("Error: No super-resolution model selected")
            return
            
        # Disable button during processing
        self.process_btn.configure(state="disabled")
        self.progress['value'] = 0
        
        # Start processing in a separate thread
        thread = threading.Thread(target=self.process_image_thread)
        thread.daemon = True
        thread.start()
        
        # Check if thread is done
        def check_thread():
            if thread.is_alive():
                self.root.after(100, check_thread)
            else:
                self.process_btn.configure(state="normal")
        
        self.root.after(100, check_thread)

if __name__ == "__main__":
    root = tk.Tk()
    app = DenoiseSRApp(root)
    root.mainloop()