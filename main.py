import os
import threading
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageChops
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import vision_transformer as vits


class DinoVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DINO Visualizer")

        self.image_path = None

        # Main container frame with a scrollbar
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=1)

        self.canvas = tk.Canvas(main_frame)
        self.scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Frame for controls
        control_frame = ttk.Frame(self.scrollable_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.select_button = ttk.Button(control_frame, text="Select Image", command=self.select_image)
        self.select_button.pack(side=tk.LEFT)

        self.process_button = ttk.Button(control_frame, text="Process Image", command=self.process_image)
        self.process_button.pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(control_frame, orient=tk.HORIZONTAL, length=200, mode='indeterminate')
        self.progress.pack(side=tk.LEFT, padx=10)

        # Frame for image display
        self.image_frame = ttk.Frame(self.scrollable_frame)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.original_image_label = ttk.Label(self.image_frame, text="Original Image")
        self.original_image_label.pack()

        self.original_image_canvas = tk.Canvas(self.image_frame, width=480, height=480)
        self.original_image_canvas.pack()

        self.processed_image_label = ttk.Label(self.image_frame, text="Processed Image")
        self.processed_image_label.pack()

        self.processed_image_canvas = tk.Canvas(self.image_frame, width=480, height=480)
        self.processed_image_canvas.pack()

        self.attention_map_label = ttk.Label(self.image_frame, text="Attention Map")
        self.attention_map_label.pack()

        self.attention_map_canvas = tk.Canvas(self.image_frame, width=480, height=480)
        self.attention_map_canvas.pack()

    def select_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            self.show_image(self.image_path, self.original_image_canvas)

    def show_image(self, image_path, canvas):
        img = Image.open(image_path)
        self.original_img = img
        img.thumbnail((480, 480), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img)
        canvas.create_image(240, 240, image=self.tk_img)

    def process_image(self):
        if self.image_path:
            self.progress.start()
            thread = threading.Thread(target=self.run_dino)
            thread.start()

    def run_dino(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # build model
        model = vits.__dict__['vit_small'](patch_size=8, num_classes=0)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.to(device)

        # load weights
        url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url)
        model.load_state_dict(state_dict, strict=True)

        # open image
        with open(self.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        transform = pth_transforms.Compose([
            pth_transforms.Resize((self.original_img.height, self.original_img.width)),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        img = transform(img)

        # make the image divisible by the patch size
        w, h = img.shape[1] - img.shape[1] % 8, img.shape[2] - img.shape[2] % 8
        img = img[:, :w, :h].unsqueeze(0)

        w_featmap = img.shape[-2] // 8
        h_featmap = img.shape[-1] // 8

        attentions = model.get_last_selfattention(img.to(device))

        # we keep only the output patch attention for head 0 (index 0)
        attentions = attentions[0, 0, 0, 1:].reshape(w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0).unsqueeze(0), scale_factor=8, mode="bicubic")[0, 0].cpu().numpy()

        # Schedule the update to be done in the main thread
        self.root.after(0, self.display_attention_map, attentions)

    def display_attention_map(self, attention):
        # Normalize the attention map to range [0, 1]
        attention = (attention - np.min(attention)) / (np.max(attention) - np.min(attention))

        # Convert to image
        attention_img = Image.fromarray((attention * 255).astype(np.uint8)).resize(self.original_img.size, Image.LANCZOS)
        
        # Display the attention map
        self.tk_attention_img = ImageTk.PhotoImage(attention_img)
        self.attention_map_canvas.create_image(240, 240, image=self.tk_attention_img)

        # Smooth the attention map with Gaussian blur
        attention_img = attention_img.filter(ImageFilter.GaussianBlur(radius=8))

        # Create an inverse attention map
        inverse_attention = ImageChops.invert(attention_img)

        # Slightly darken the original image using the inverse attention map
        enhancer_dark = ImageEnhance.Brightness(self.original_img)
        darkened_img = enhancer_dark.enhance(0.8)  # Decrease brightness by a factor of 0.8
        partially_darkened_img = Image.composite(darkened_img, self.original_img, inverse_attention)

        # Enhance brightness and contrast of the original image using the attention map as a mask
        enhancer_light = ImageEnhance.Brightness(partially_darkened_img)
        brightened_img = enhancer_light.enhance(1.8)  # Increase brightness by a factor of 1.5

        enhancer_contrast = ImageEnhance.Contrast(brightened_img)
        contrast_img = enhancer_contrast.enhance(1.2)  # Increase contrast by a factor of 1.5

        masked_img = Image.composite(contrast_img, partially_darkened_img, attention_img)

        self.tk_img_processed = ImageTk.PhotoImage(masked_img)
        self.processed_image_canvas.create_image(240, 240, image=self.tk_img_processed)

        self.progress.stop()


if __name__ == "__main__":
    root = tk.Tk()
    app = DinoVisualizerApp(root)
    root.mainloop()
