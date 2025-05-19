import cv2
import numpy as np

def calculate_energy(img):
    """Calculating image energy (gradient norm)"""
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

def remove_vertical_seam(img, seam):
    h, w = img.shape[:2]
    new_img = np.zeros((h, w-1, 3), dtype=img.dtype)
    
    for i in range(h):
        j = seam[i]
        new_img[i] = np.vstack((img[i, :j], img[i, j+1:]))
    
    return new_img

def seam_carving(img, new_width=None, new_height=None):
    h, w = img.shape[:2]
    
    if new_width is not None and new_width >= w:
        new_width = None
    if new_height is not None and new_height >= h:
        new_height = None
    
    if new_width is None and new_height is None:
        return img
    
    delta_w = w - new_width if new_width is not None else 0
    delta_h = h - new_height if new_height is not None else 0
    
    for _ in range(delta_w):
        energy = calculate_energy(img)
        seam = find_vertical_seam(energy)
        img = remove_vertical_seam(img, seam)
    
    if delta_h > 0:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        for _ in range(delta_h):
            energy = calculate_energy(img)
            seam = find_vertical_seam(energy)
            img = remove_vertical_seam(img, seam)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return img

def show_images(original, result, scale=0.7):
    """Beautifully displays the original and processed image"""
    h, w = original.shape[:2]
    new_h, new_w = result.shape[:2]
    
    # Create a background image for display
    display = np.zeros((max(h, new_h), w + new_w + 20, 3), dtype=np.uint8)
    display.fill(240)  # Light gray background
    
    # We post images
    display[:h, :w] = original
    display[:new_h, w+20:w+20+new_w] = result
    
    # Adding signatures
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
    # Entering file paths from the keyboard
    input_path = input("Enter the path to the original image: ")
    output_path = input("Enter the path to save the result: ")
    
    # Reading the image
    original_img = cv2.imread(input_path)
    if original_img is None:
        print(f"Error: Failed to load image {input_path}")
        return
    
    img = original_img.copy()
    h, w = img.shape[:2]
    
    while True:
        print("\nCurrent image dimensions:", f"{w}x{h}")
        
        # Requesting an action from the user
        action = input("\nSelect action:\n"
                      "1 - Zoom out\n"
                      "2 - Reset to original\n"
                      "3 - Save result\n"
                      "4 - Exit\n"
                      "Your choice: ")
        
        if action == '1':
            # Request new sizes
            try:
                new_width = int(input(f"Enter new width (<= {w}): "))
                new_height = int(input(f"Enter new height (<= {h}): "))
                
                # Input validation
                if new_width > w or new_height > h or new_width <= 0 or new_height <= 0:
                    print("Error: invalid dimensions")
                    continue
                
                print("Processing... This may take some time....")
                result = seam_carving(img.copy(), new_width, new_height)
                img = result.copy()
                h, w = img.shape[:2]
                
                # Showing the result
                show_images(original_img, img)
                cv2.waitKey(500)  # Give the window time to update
                
            except ValueError:
                print("Error: Enter integers for dimensions")
                
        elif action == '2':
            # Reset to original
            img = original_img.copy()
            h, w = img.shape[:2]
            print("Image reset to original")
            show_images(original_img, img)
            cv2.waitKey(500)
            
        elif action == '3':
            # Saving the result
            cv2.imwrite(output_path, img)
            print(f"Image saved as {output_path}")
            
        elif action == '4':
            # Exit
            print("Exit the program")
            break
            
        else:
            print("Unknown team")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()