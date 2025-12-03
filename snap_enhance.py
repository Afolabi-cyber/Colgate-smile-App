import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def enhance_image_snapchat_style(image_path, output_path=None):
    """
    Enhance image similar to Snapchat's default camera enhancement.
    
    Args:
        image_path: Path to input image (string or numpy array)
        output_path: Path to save enhanced image (optional)
    
    Returns:
        Enhanced image as numpy array
    """
    # Load image
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image_path
    
    # Convert to PIL for easier manipulation
    pil_img = Image.fromarray(img_rgb)
    
    # 1. Brightness enhancement (subtle boost)
    brightness_enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = brightness_enhancer.enhance(1.1)  # 10% brighter
    
    # 2. Contrast enhancement
    contrast_enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = contrast_enhancer.enhance(1.15)  # 15% more contrast
    
    # 3. Color/Saturation boost
    color_enhancer = ImageEnhance.Color(pil_img)
    pil_img = color_enhancer.enhance(1.2)  # 20% more saturation
    
    # 4. Slight sharpening
    pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=3))
    
    # Convert back to numpy array
    enhanced = np.array(pil_img)
    
    # 5. Apply subtle skin smoothing (optional - reduces noise)
    enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=20, sigmaSpace=20)
    
    # 6. Warm tone adjustment (slight yellow tint for better skin tones)
    enhanced = enhanced.astype(np.float32)
    enhanced[:, :, 0] = np.clip(enhanced[:, :, 0] * 1.02, 0, 255)  # Red
    enhanced[:, :, 1] = np.clip(enhanced[:, :, 1] * 1.01, 0, 255)  # Green
    enhanced = enhanced.astype(np.uint8)
    
    # Save if output path provided
    if output_path:
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, enhanced_bgr)
        print(f"Enhanced image saved to: {output_path}")
    
    return enhanced


def enhance_image_advanced(image_path, output_path=None, 
                          brightness=1.1, contrast=1.15, 
                          saturation=1.2, sharpness=1.0):
    """
    Advanced version with customizable parameters.
    
    Args:
        image_path: Path to input image
        output_path: Path to save enhanced image (optional)
        brightness: Brightness multiplier (1.0 = no change, >1.0 = brighter)
        contrast: Contrast multiplier (1.0 = no change, >1.0 = more contrast)
        saturation: Saturation multiplier (1.0 = no change, >1.0 = more vivid)
        sharpness: Sharpness level (0.0 to 2.0, 1.0 = original)
    
    Returns:
        Enhanced image as numpy array
    """
    # Load image
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image_path
    
    pil_img = Image.fromarray(img_rgb)
    
    # Apply enhancements
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(brightness)
    
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(contrast)
    
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(saturation)
    
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(sharpness)
    
    enhanced = np.array(pil_img)
    
    # Save if output path provided
    if output_path:
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, enhanced_bgr)
        print(f"Enhanced image saved to: {output_path}")
    
    return enhanced


# Example usage
if __name__ == "__main__":
    # Basic enhancement
    input_image = "input.jpg"
    output_image = "enhanced_output.jpg"
    
    # Simple usage
    enhanced = enhance_image_snapchat_style(input_image, output_image)
    
    # Advanced usage with custom parameters
    # enhanced = enhance_image_advanced(
    #     input_image, 
    #     output_image,
    #     brightness=1.15,
    #     contrast=1.2,
    #     saturation=1.25,
    #     sharpness=1.1
    # )
    
    print("Image enhancement complete!")
    
    # You can also display the result
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(cv2.imread(input_image), cv2.COLOR_BGR2RGB))
    # plt.title("Original")
    # plt.axis('off')
    # plt.subplot(1, 2, 2)
    # plt.imshow(enhanced)
    # plt.title("Enhanced")
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()