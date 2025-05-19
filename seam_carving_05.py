import cv2
import numpy as np

def calculate_energy(img):
    """Calculate image energy (gradient magnitude)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy = np.sqrt(grad_x**2 + grad_y**2)
    return energy

def find_vertical_seam(energy):
    h, w = energy.shape
    dp = energy.copy()
    backtrack = np.zeros_like(dp, dtype=np.int32)
    
    for i in range(1, h):
        for j in range(w):
            if j == 0:
                idx = np.argmin(dp[i-1, j:j+2])
                min_energy = dp[i-1, j + idx]
                backtrack[i, j] = j + idx
            elif j == w - 1:
                idx = np.argmin(dp[i-1, j-1:j+1])
                min_energy = dp[i-1, j - 1 + idx]
                backtrack[i, j] = j - 1 + idx
            else:
                idx = np.argmin(dp[i-1, j-1:j+2])
                min_energy = dp[i-1, j - 1 + idx]
                backtrack[i, j] = j - 1 + idx
            
            dp[i, j] += min_energy
    
    seam = []
    j = np.argmin(dp[-1])
    seam.append(j)
    
    for i in range(h-1, 0, -1):
        j = backtrack[i, j]
        seam.append(j)
    
    return seam[::-1]

def draw_seam(img, seam, color=(0, 0, 255)):
    """Draw seam on the image"""
    img_with_seam = img.copy()
    for i, j in enumerate(seam):
        cv2.circle(img_with_seam, (j, i), 1, color, -1)
    return img_with_seam

def remove_vertical_seam(img, seam):
    h, w = img.shape[:2]
    new_img = np.zeros((h, w-1, 3), dtype=img.dtype)
    
    for i in range(h):
        j = seam[i]
        new_img[i] = np.vstack((img[i, :j], img[i, j+1:]))
    
    return new_img

def seam_carving(img, new_width=None, new_height=None, visualize=False):
    h, w = img.shape[:2]
    
    if new_width is not None and new_width >= w:
        new_width = None
    if new_height is not None and new_height >= h:
        new_height = None
    
    if new_width is None and new_height is None:
        return img
    
    delta_w = w - new_width if new_width is not None else 0
    delta_h = h - new_height if new_height is not None else 0
    
    if visualize and (delta_w > 0 or delta_h > 0):
        # Create visualization window
        cv2.namedWindow("Seam Carving Process", cv2.WINDOW_NORMAL)
    
    for iteration in range(delta_w):
        energy = calculate_energy(img)
        seam = find_vertical_seam(energy)
        
        if visualize:
            # Visualization process
            img_with_seam = draw_seam(img, seam)
            energy_normalized = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            energy_colored = cv2.applyColorMap(energy_normalized, cv2.COLORMAP_JET)
            
            # Combine images into one panel
            h_panel = max(img.shape[0], energy_colored.shape[0])
            w_panel = img.shape[1] + energy_colored.shape[1] + 20
            panel = np.zeros((h_panel, w_panel, 3), dtype=np.uint8)
            panel.fill(240)
            
            # Place images
            panel[:img.shape[0], :img.shape[1]] = img_with_seam
            panel[:energy_colored.shape[0], img.shape[1]+20:] = energy_colored
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(panel, f"Iteration: {iteration+1}/{delta_w}", (10, 30), font, 0.8, (0,0,0), 2)
            cv2.putText(panel, "Image with seam", (10, 60), font, 0.6, (0,0,0), 1)
            cv2.putText(panel, "Energy map", (img.shape[1]+30, 60), font, 0.6, (0,0,0), 1)
            
            cv2.imshow("Seam Carving Process", panel)
            key = cv2.waitKey(100)  # Short delay for observation
            if key == 27:  # ESC - stop visualization
                visualize = False
        
        img = remove_vertical_seam(img, seam)
    
    if delta_h > 0:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        for iteration in range(delta_h):
            energy = calculate_energy(img)
            seam = find_vertical_seam(energy)
            
            if visualize:
                img_with_seam = draw_seam(img, seam)
                energy_normalized = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                energy_colored = cv2.applyColorMap(energy_normalized, cv2.COLORMAP_JET)
                
                h_panel = max(img.shape[0], energy_colored.shape[0])
                w_panel = img.shape[1] + energy_colored.shape[1] + 20
                panel = np.zeros((h_panel, w_panel, 3), dtype=np.uint8)
                panel.fill(240)
                
                panel[:img.shape[0], :img.shape[1]] = img_with_seam
                panel[:energy_colored.shape[0], img.shape[1]+20:] = energy_colored
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(panel, f"Iteration: {iteration+1}/{delta_h} (rotated)", (10, 30), font, 0.8, (0,0,0), 2)
                cv2.putText(panel, "Image with seam", (10, 60), font, 0.6, (0,0,0), 1)
                cv2.putText(panel, "Energy map", (img.shape[1]+30, 60), font, 0.6, (0,0,0), 1)
                
                cv2.imshow("Seam Carving Process", panel)
                key = cv2.waitKey(100)
                if key == 27:
                    visualize = False
            
            img = remove_vertical_seam(img, seam)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    if visualize and (delta_w > 0 or delta_h > 0):
        cv2.destroyWindow("Seam Carving Process")
    
    return img

def show_images(original, result, scale=0.7):
    """Display original and processed images"""
    h, w = original.shape[:2]
    new_h, new_w = result.shape[:2]
    
    # Create background for display
    display = np.zeros((max(h, new_h), w + new_w + 20, 3), dtype=np.uint8)
    display.fill(240)  # Light gray background
    
    # Place images
    display[:h, :w] = original
    display[:new_h, w+20:w+20+new_w] = result
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(display, "Original", (10, 30), font, 0.8, (0,0,0), 2)
    cv2.putText(display, f"{w}x{h}", (10, 60), font, 0.6, (0,0,0), 1)
    cv2.putText(display, "Result", (w+30, 30), font, 0.8, (0,0,0), 2)
    cv2.putText(display, f"{new_w}x{new_h}", (w+30, 60), font, 0.6, (0,0,0), 1)
    
    # Scale for display
    if scale != 1:
        display = cv2.resize(display, None, fx=scale, fy=scale)
    
    cv2.imshow("Seam Carving Comparison", display)

def main():
    # Keyboard input for file paths
    input_path = input("Enter path to source image: ")
    output_path = input("Enter path to save result: ")
    
    # Read image
    original_img = cv2.imread(input_path)
    if original_img is None:
        print(f"Error: failed to load image {input_path}")
        return
    
    img = original_img.copy()
    h, w = img.shape[:2]
    
    # Show original image
    cv2.imshow("Original Image", original_img)
    cv2.waitKey(500)
    
    while True:
        print("\nCurrent image dimensions:", f"{w}x{h}")
        
        # Get user action
        action = input("\nSelect action:\n"
                      "1 - Zoom out\n"
                      "2 - Reset to original\n"
                      "3 - Save result\n"
                      "4 - Exit\n"
                      "Your choice: ")
        
        if action == '1':
            # Get new dimensions
            try:
                new_width = int(input(f"Enter new width (<= {w}): "))
                new_height = int(input(f"Enter new height (<= {h}): "))
                
                # Input validation
                if new_width > w or new_height > h or new_width <= 0 or new_height <= 0:
                    print("Error: invalid dimensions")
                    continue
                
                # Ask for visualization
                visualize = input("Show seam removal process? (y/n): ").lower() == 'y'
                
                print("Processing... This may take some time...")
                result = seam_carving(img.copy(), new_width, new_height, visualize)
                img = result.copy()
                h, w = img.shape[:2]
                
                # Show result
                show_images(original_img, img)
                cv2.waitKey(500)  # Give window time to update
                
            except ValueError:
                print("Error: please enter integers for dimensions")
                
        elif action == '2':
            # Reset to original
            img = original_img.copy()
            h, w = img.shape[:2]
            print("Image reset to original")
            show_images(original_img, img)
            cv2.waitKey(500)
            
        elif action == '3':
            # Save result
            cv2.imwrite(output_path, img)
            print(f"Image saved as {output_path}")
            
        elif action == '4':
            # Exit
            print("Exiting program")
            break
            
        else:
            print("Unknown command")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()