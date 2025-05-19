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
    
    # Checking the acceptability of new sizes
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
    
    # Transpose for horizontal seams
    if delta_h > 0:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        for _ in range(delta_h):
            energy = calculate_energy(img)
            seam = find_vertical_seam(energy)
            img = remove_vertical_seam(img, seam)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return img

if __name__ == "__main__":
    input_path = "castle.png"
    output_path = "Result_1.png"
    
    # Reading the image
    img = cv2.imread(input_path)
    if img is None:
        print("Error: Failed to load image")
        exit()
    
    # Showing original dimensions
    h, w = img.shape[:2]
    print(f"Original dimensions: {w}x{h}")
    
    # Requesting new sizes from the user
    new_width = int(input(f"Enter new width (<= {w}): "))
    new_height = int(input(f"Enter new height (<= {h}): "))
    
    # We check that the new dimensions are smaller than the original ones
    new_width = min(new_width, w)
    new_height = min(new_height, h)
    
    # We use seam carving
    result = seam_carving(img, new_width, new_height)
    
    # Save the result
    cv2.imwrite(output_path, result)
    print(f"Image saved in {output_path}")
    
    # Showing the result
    cv2.imshow("Original", img)
    cv2.imshow("Seam Carved", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()















    