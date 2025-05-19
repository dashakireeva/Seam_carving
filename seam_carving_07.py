# вывод 3 окон(оригинальное, выбор и удаление швов, оригинальное и преобразованное фото)
# 2 окно визуализация процесса seam carving, включая отображение энергетической карты, швов и промежуточных результатов.
# замер времени выполнения операции удаления швов и вывод информации о скорости обработки.
import cv2
import numpy as np
import time
from numba import jit

@jit(nopython=True)
def find_vertical_seam(energy):
    """Оптимизированный поиск вертикального шва с Numba"""
    h, w = energy.shape
    dp = energy.copy()
    backtrack = np.zeros_like(dp, dtype=np.int32)
    
    for i in range(1, h):
        for j in range(w):
            if j == 0:
                min_val = min(dp[i-1, j], dp[i-1, j+1])
                backtrack[i, j] = j if dp[i-1, j] == min_val else j + 1
            elif j == w - 1:
                min_val = min(dp[i-1, j-1], dp[i-1, j])
                backtrack[i, j] = j - 1 if dp[i-1, j-1] == min_val else j
            else:
                min_val = min(dp[i-1, j-1], dp[i-1, j], dp[i-1, j+1])
                if dp[i-1, j-1] == min_val:
                    backtrack[i, j] = j - 1
                elif dp[i-1, j] == min_val:
                    backtrack[i, j] = j
                else:
                    backtrack[i, j] = j + 1
            
            dp[i, j] += min_val
    
    seam = np.zeros(h, dtype=np.int32)
    seam[-1] = np.argmin(dp[-1])
    
    for i in range(h-2, -1, -1):
        seam[i] = backtrack[i+1, seam[i+1]]
    
    return seam

def calculate_energy(img):
    """Оптимизированное вычисление энергии с Scharr и CV_32F"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    energy = np.sqrt(grad_x**2 + grad_y**2)
    return energy

# Остальной код без изменений (как в оригинале)
def draw_seam(img, seam, color=(0, 0, 255)):
    """Рисует шов на изображении"""
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
    
    start_time = time.time()
    
    if visualize and (delta_w > 0 or delta_h > 0):
        cv2.namedWindow("Seam Carving Process", cv2.WINDOW_NORMAL)
    
    for iteration in range(delta_w):
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
            elapsed_time = time.time() - start_time
            cv2.putText(panel, f"Iteration: {iteration+1}/{delta_w}", (10, 30), font, 0.8, (0,0,0), 2)
            cv2.putText(panel, f"Time: {elapsed_time:.2f}s | {elapsed_time/(iteration+1):.4f}s per seam", 
                       (10, 70), font, 0.6, (0,0,0), 1)
            cv2.putText(panel, "Image with seam", (10, 100), font, 0.6, (0,0,0), 1)
            cv2.putText(panel, "Energy map", (img.shape[1]+30, 100), font, 0.6, (0,0,0), 1)
            
            cv2.imshow("Seam Carving Process", panel)
            key = cv2.waitKey(100)
            if key == 27:
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
                elapsed_time = time.time() - start_time
                cv2.putText(panel, f"Iteration: {iteration+1}/{delta_h} (rotated)", (10, 30), font, 0.8, (0,0,0), 2)
                cv2.putText(panel, f"Time: {elapsed_time:.2f}s | {elapsed_time/(delta_w + iteration+1):.4f}s per seam", 
                           (10, 70), font, 0.6, (0,0,0), 1)
                cv2.putText(panel, "Image with seam", (10, 100), font, 0.6, (0,0,0), 1)
                cv2.putText(panel, "Energy map", (img.shape[1]+30, 100), font, 0.6, (0,0,0), 1)
                
                cv2.imshow("Seam Carving Process", panel)
                key = cv2.waitKey(100)
                if key == 27:
                    visualize = False
            
            img = remove_vertical_seam(img, seam)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    total_time = time.time() - start_time
    total_seams = delta_w + delta_h
    if total_seams > 0:
        print(f"\nОбщее время обработки: {total_time:.2f} секунд")
        print(f"Удалено швов: {total_seams}")
        print(f"Среднее время на один шов: {total_time/total_seams:.4f} секунд")
        print(f"Скорость обработки: {total_seams/total_time:.2f} швов/секунду")
    
    if visualize and (delta_w > 0 or delta_h > 0):
        cv2.destroyWindow("Seam Carving Process")
    
    return img

def show_images(original, result, scale=0.7):
    """Красиво отображает оригинальное и обработанное изображение"""
    h, w = original.shape[:2]
    new_h, new_w = result.shape[:2]
    
    display = np.zeros((max(h, new_h), w + new_w + 20, 3), dtype=np.uint8)
    display.fill(240)
    
    display[:h, :w] = original
    display[:new_h, w+20:w+20+new_w] = result
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(display, "Original", (10, 30), font, 0.8, (0,0,0), 2)
    cv2.putText(display, f"{w}x{h}", (10, 60), font, 0.6, (0,0,0), 1)
    cv2.putText(display, "Result", (w+30, 30), font, 0.8, (0,0,0), 2)
    cv2.putText(display, f"{new_w}x{new_h}", (w+30, 60), font, 0.6, (0,0,0), 1)
    
    if scale != 1:
        display = cv2.resize(display, None, fx=scale, fy=scale)
    
    cv2.imshow("Seam Carving Comparison", display)

def main():
    input_path = input("Введите путь к исходному изображению: ")
    output_path = input("Введите путь для сохранения результата: ")
    
    original_img = cv2.imread(input_path)
    if original_img is None:
        print(f"Ошибка: не удалось загрузить изображение {input_path}")
        return
    
    img = original_img.copy()
    h, w = img.shape[:2]
    
    cv2.imshow("Original Image", original_img)
    cv2.waitKey(500)
    
    while True:
        print("\nТекущие размеры изображения:", f"{w}x{h}")
        
        action = input("\nВыберите действие:\n"
                      "1 - Уменьшить изображение\n"
                      "2 - Сбросить к оригиналу\n"
                      "3 - Сохранить результат\n"
                      "4 - Выход\n"
                      "Ваш выбор: ")
        
        if action == '1':
            try:
                new_width = int(input(f"Введите новую ширину (<= {w}): "))
                new_height = int(input(f"Введите новую высоту (<= {h}): "))
                
                if new_width > w or new_height > h or new_width <= 0 or new_height <= 0:
                    print("Ошибка: недопустимые размеры")
                    continue
                
                visualize = input("Показывать процесс удаления швов? (y/n): ").lower() == 'y'
                
                print("Обработка... Это может занять некоторое время...")
                result = seam_carving(img.copy(), new_width, new_height, visualize)
                img = result.copy()
                h, w = img.shape[:2]
                
                show_images(original_img, img)
                cv2.waitKey(500)
                
            except ValueError:
                print("Ошибка: введите целые числа для размеров")
                
        elif action == '2':
            img = original_img.copy()
            h, w = img.shape[:2]
            print("Изображение сброшено к оригиналу")
            show_images(original_img, img)
            cv2.waitKey(500)
            
        elif action == '3':
            cv2.imwrite(output_path, img)
            print(f"Изображение сохранено как {output_path}")
            
        elif action == '4':
            print("Выход из программы")
            break
            
        else:
            print("Неизвестная команда")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.runcall(main)
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime').print_stats(10)
