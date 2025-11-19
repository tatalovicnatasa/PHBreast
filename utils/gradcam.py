import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping)
    
    Šta radi:
    - Pokazuje koje delove slike model "gleda" kada donosi odluku
    - Kreira heatmap preko originalne slike
    
    Primer:
    - Ako model predikcija BIRADS 4, Grad-CAM pokazuje da li gleda masu,
      kalcifikacije, ili nešto drugo
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: Treniran model
            target_layer: Layer za koji računamo Grad-CAM
                         Preporuka: model.layer4 (poslednji conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Registruj hook-ove za čuvanje activation i gradijenata
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook za čuvanje activation maps."""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook za čuvanje gradijenata."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        """
        Generiši Grad-CAM heatmap.
        
        Args:
            input_image: Input slika (1 x C x H x W tensor)
            target_class: Ciljna klasa (ako None, koristi predviđenu)
        
        Returns:
            cam: Grad-CAM heatmap (H x W numpy array)
            pred_class: Predviđena klasa
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        # Ako nije specificirana klasa, koristi predviđenu
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass za target klasu
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Izračunaj Grad-CAM
        # Korak 1: Global average pooling gradijenata → weights
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Korak 2: Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # Korak 3: ReLU (samo pozitivni uticaji)
        cam = F.relu(cam)
        
        # Korak 4: Normalize na [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        
        return cam, target_class
    
    def visualize(self, input_image, original_image, target_class=None, 
                  save_path=None, alpha=0.5):
        """
        Vizualizuj Grad-CAM preko originalne slike.
        
        Args:
            input_image: Normalizovana input slika (tensor)
            original_image: Originalna slika (numpy array, H x W)
            target_class: Ciljna BIRADS klasa (0-4)
            save_path: Putanja za čuvanje slike
            alpha: Transparentnost heatmap-a (0=nevidljiv, 1=potpuno vidljiv)
        """
        # Generiši CAM
        cam, pred_class = self.generate_cam(input_image, target_class)
        
        # Resize CAM na dimenzije originalne slike
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Konvertuj u heatmap (JET colormap)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Normalize original image na [0, 255]
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        
        # Konvertuj grayscale → RGB ako je potrebno
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        # Overlay heatmap preko originalne slike
        superimposed = heatmap * alpha + original_image * (1 - alpha)
        superimposed = superimposed.astype(np.uint8)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title(f'Grad-CAM\n(Predicted BIRADS: {pred_class+1})', fontsize=12)
        axes[1].axis('off')
        
        axes[2].imshow(superimposed)
        axes[2].set_title('Overlay', fontsize=12)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Grad-CAM saved to: {save_path}")
        
        plt.show()
        
        return superimposed, cam_resized