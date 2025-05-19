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
    """Find vertical seam with minimum energy"""
    h, w = energy.shape
    dp = energy.copy()
    backtrack = np.zeros_like(dp, dtype=np.int32)
    
    for i in range(1, h):
        for j in range(w):
            if j == 0:
                idx = np.argmin(dp[i-1, j:j+2])
                backtrack[i, j] = j + idx
            elif j == w - 1:
                idx = np.argmin(dp[i-1, j-1:j+1])
                backtrack[i, j] = j - 1 + idx
            else:
                idx = np.argmin(dp[i-1, j-1:j+2])
                backtrack[i, j] = j - 1 + idx
            dp[i, j] += dp[i-1, backtrack[i, j]]
    
    seam = [np.argmin(dp[-1])]
    for i in range(h-1, 0, -1):
        seam.append(backtrack[i, seam[-1]])
    
    return seam[::-1]

def remove_vertical_seam(img, seam):
    """Remove vertical seam from image"""
    h, w = img.shape[:2]
    return np.array([np.delete(img[i], seam[i], axis=0) for i in range(h)])

def seam_carving(img, new_width=None, new_height=None):
    """Perform seam carving to resize image"""
    h, w = img.shape[:2]
    
    if new_width is not None:
        for _ in range(w - new_width):
            energy = calculate_energy(img)
            seam = find_vertical_seam(energy)
            img = remove_vertical_seam(img, seam)
    
    if new_height is not None:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = seam_carving(img, new_width=new_height)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return img

def add_ruler(image, position='left', size=30, step=50):
    """Add ruler to image for size reference"""
    h, w = image.shape[:2]
    if position in ['left', 'right']:
        ruler = np.zeros((h, size, 3), dtype=np.uint8)
        ruler.fill(220)
        for y in range(0, h, step):
            length = 20 if y % (step*5) == 0 else 10
            cv2.line(ruler, (size-length, y), (size-1, y), (0,0,0), 1)
            if y % (step*5) == 0:
                cv2.putText(ruler, str(y), (2, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        return np.hstack((ruler, image)) if position == 'left' else np.hstack((image, ruler))
    else:
        ruler = np.zeros((size, w, 3), dtype=np.uint8)
        ruler.fill(220)
        for x in range(0, w, step):
            length = 20 if x % (step*5) == 0 else 10
            cv2.line(ruler, (x, size-length), (x, size-1), (0,0,0), 1)
            if x % (step*5) == 0:
                cv2.putText(ruler, str(x), (x-10, size-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        return np.vstack((ruler, image))

def show_images(original, result):
    """Display original and result images with rulers"""
    original_with_rulers = add_ruler(add_ruler(original, 'left'), 'top')
    result_with_rulers = add_ruler(add_ruler(result, 'right'), 'top')
    
    h, w = original_with_rulers.shape[:2]
    h2, w2 = result_with_rulers.shape[:2]
    display = np.zeros((max(h, h2), w + w2 + 20, 3), dtype=np.uint8)
    display.fill(240)
    
    display[:h, :w] = original_with_rulers
    display[:h2, w+20:w+w2+20] = result_with_rulers
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(display, "Original", (30, 30), font, 0.8, (0,0,0), 2)
    cv2.putText(display, f"{original.shape[1]}x{original.shape[0]}", (30, 60), font, 0.6, (0,0,0), 1)
    cv2.putText(display, "Result", (w+50, 30), font, 0.8, (0,0,0), 2)
    cv2.putText(display, f"{result.shape[1]}x{result.shape[0]}", (w+50, 60), font, 0.6, (0,0,0), 1)
    
    cv2.namedWindow("Seam Carving", cv2.WINDOW_NORMAL)
    cv2.imshow("Seam Carving", display)
    cv2.waitKey(3000)  # Display for 3 seconds
    cv2.destroyAllWindows()

def main():
    input_path = input("Enter image path: ")
    original_img = cv2.imread(input_path)
    if original_img is None:
        print("Error: Failed to load image")
        return
    
    img = original_img.copy()
    
    while True:
        print(f"\nCurrent size: {img.shape[1]}x{img.shape[0]}")
        print("Select action:")
        print("1 - Zoom out")
        print("2 - Reset to original")
        print("3 - Save result")
        print("4 - Exit")
        
        choice = input("Your choice: ")
        
        if choice == '1':
            try:
                new_width = int(input(f"Enter new width (<= {img.shape[1]}): "))
                new_height = int(input(f"Enter new height (<= {img.shape[0]}): "))
                
                if new_width <= 0 or new_height <= 0 or new_width > img.shape[1] or new_height > img.shape[0]:
                    print("Invalid dimensions!")
                    continue
                
                result = seam_carving(img.copy(), new_width, new_height)
                show_images(original_img, result)
                img = result
                
            except ValueError:
                print("Invalid input! Please enter numbers.")
        
        elif choice == '2':
            img = original_img.copy()
            print("Image reset to original")
        
        elif choice == '3':
            output_path = input("Enter output filename (e.g., 'output.jpg'): ")
            cv2.imwrite(output_path, img)
            print(f"Image saved as {output_path}")
        
        elif choice == '4':
            break
        
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
