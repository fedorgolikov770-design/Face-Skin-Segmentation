import sys
import cv2
import dlib
import numpy as np
from pathlib import Path


# Детекция ключевых точек (Dlib)

def get_landmarks(img, detector, predictor):
    """Находит лицо и возвращает массив (N,2) ключевых точек."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if len(faces) == 0:
        return None
    shape = predictor(gray, faces[0])
    return np.array([[shape.part(i).x, shape.part(i).y]
                     for i in range(shape.num_parts)], dtype=np.int32)


def extrapolate_forehead(landmarks, scale=0.6):
    """Достраивает точки лба для 68-точечной модели (правило третей)."""
    glabella = ((landmarks[21] + landmarks[22]) / 2).astype(np.int32)
    chin = landmarks[8]
    dist = abs(int(glabella[1]) - int(chin[1]))
    top_y = int(glabella[1]) - int(scale * dist)

    pts = []
    for pt in landmarks[17:27]:  # брови
        pts.append([pt[0], top_y])
    return np.array(pts, dtype=np.int32)


# Геометрическая маска лица

def build_geometric_mask(landmarks, h, w):
    """Строит маску лица: контур челюсти (0-16) + линия лба."""
    mask = np.zeros((h, w), dtype=np.uint8)
    jaw = landmarks[0:17]

    if landmarks.shape[0] >= 81:
        forehead = landmarks[68:81]
    else:
        forehead = extrapolate_forehead(landmarks, scale=0.65)

    contour = np.concatenate([jaw, forehead[::-1]])
    cv2.fillPoly(mask, [contour], 255)
    return mask


# Цветовая фильтрация кожи

def is_grayscale(img):
    """Проверяет, является ли изображение чёрно-белым (R ≈ G ≈ B)."""
    B, G, R = cv2.split(img)
    diff_rg = np.mean(np.abs(R.astype(int) - G.astype(int)))
    diff_rb = np.mean(np.abs(R.astype(int) - B.astype(int)))
    return diff_rg < 5 and diff_rb < 5


def build_color_mask(img):
    """
    Определяет пиксели кожи по цвету в трёх пространствах:
    RGB ∩ (HSV ∪ YCbCr).

    Для ЧБ изображений цветовая фильтрация невозможна —
    используется только геометрическая маска.
    """
    h, w = img.shape[:2]

    if is_grayscale(img):
        return np.full((h, w), 255, dtype=np.uint8)

    # RGB
    B, G, R = [ch.astype(np.int32) for ch in cv2.split(img)]
    rgb = (R > 95) & (G > 40) & (B > 20) & (R > G) & (R > B) & (np.abs(R - G) > 15)

    # HSV
    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    hsv = (H <= 25) & (S >= 45) & (S <= 200) & (V >= 60)

    # YCbCr
    Y, Cr, Cb = [ch.astype(np.int32) for ch in cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))]
    ycbcr = (
        (Y > 80) & (Cb > 77) & (Cb < 127) & (Cr > 133) & (Cr < 173) &
        (Cr <= 1.5862 * Cb + 20) & (Cr >= 0.3448 * Cb + 76.2069) &
        (Cr >= -4.5652 * Cb + 234.5652) & (Cr <= -1.15 * Cb + 301.75) &
        (Cr <= -2.2857 * Cb + 432.85)
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[rgb & (hsv | ycbcr)] = 255
    return mask


# Морфологическое рафинирование

def morphological_refine(mask, close_k=9, open_k=5):
    """Closing заполняет дыры, Opening убирает шум."""
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
    return mask


# Исключение глаз и рта

def eye_aspect_ratio(pts):
    """EAR — отношение высоты глаза к ширине. < 0.20 = закрыт."""
    A = np.linalg.norm(pts[1].astype(float) - pts[5].astype(float))
    B = np.linalg.norm(pts[2].astype(float) - pts[4].astype(float))
    C = np.linalg.norm(pts[0].astype(float) - pts[3].astype(float))
    return (A + B) / (2.0 * C + 1e-6)


def mouth_aspect_ratio(landmarks):
    """MAR — отношение высоты рта к ширине. >= 0.30 = открыт."""
    inner = landmarks[60:68]
    A = np.linalg.norm(inner[1].astype(float) - inner[7].astype(float))
    B = np.linalg.norm(inner[2].astype(float) - inner[6].astype(float))
    C = np.linalg.norm(inner[3].astype(float) - inner[5].astype(float))
    D = np.linalg.norm(inner[0].astype(float) - inner[4].astype(float))
    return (A + B + C) / (3.0 * D + 1e-6)


def exclude_eyes_mouth(mask, landmarks):
    """Вырезает глаза (если открыты) и рот из маски."""
    face_w = abs(int(landmarks[16][0]) - int(landmarks[0][0]))
    expand = max(3, int(face_w * 0.03))

    # Глаза
    for eye in [landmarks[36:42], landmarks[42:48]]:
        if eye_aspect_ratio(eye) < 0.20:
            continue  # закрыт — веко = кожа
        center = eye.mean(axis=0)
        pts = []
        for p in eye:
            d = p.astype(float) - center
            n = np.linalg.norm(d)
            if n > 0:
                d /= n
            pts.append((p + d * expand).astype(np.int32))
        cv2.fillPoly(mask, [np.array(pts, np.int32)], 0)

    # Рот
    if mouth_aspect_ratio(landmarks) >= 0.30:
        cv2.fillPoly(mask, [landmarks[48:60]], 0)  # внешний контур
    else:
        cv2.fillPoly(mask, [landmarks[60:68]], 0)  # внутренний контур

    return mask


# Главная функция

def create_skin_mask(image_path, output_path=None,
                     predictor_path="shape_predictor_81_face_landmarks.dat"):
    """Создаёт изображение, содержащее только пиксели кожи лица."""

    # Загрузка
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")

    h, w = img.shape[:2]
    src = Path(image_path)

    if output_path is None:
        output_path = str(src.with_name(f"{src.stem}_skin.png"))

    # Детекция лица
    if not Path(predictor_path).exists():
        raise FileNotFoundError(
            f"Модель не найдена: {predictor_path}\n"
            "Скачайте shape_predictor_81_face_landmarks.dat и положите рядом с main.py")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    landmarks = get_landmarks(img, detector, predictor)

    if landmarks is None:
        raise RuntimeError("Лицо не обнаружено на изображении.")

    # Построение маски
    geom = build_geometric_mask(landmarks, h, w)
    color = build_color_mask(img)
    skin = cv2.bitwise_and(geom, color)

    # Морфология (размер ядра пропорционален лицу)
    face_h = abs(int(landmarks[8][1]) - int(landmarks[19][1]))
    close_k = max(7, int(face_h * 0.17))
    if close_k % 2 == 0:
        close_k += 1

    skin = morphological_refine(skin, close_k=close_k)
    skin = exclude_eyes_mouth(skin, landmarks)
    skin = cv2.GaussianBlur(skin, (5, 5), 0)

    # Извлечение только кожи и сохранение
    result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = skin
    if not output_path.lower().endswith('.png'):
        output_path = str(Path(output_path).with_suffix('.png'))
    cv2.imwrite(output_path, result)
    print(f"Готово: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python main.py <изображение> [выход]")
        sys.exit(1)

    create_skin_mask(
        sys.argv[1],
        sys.argv[2] if len(sys.argv) > 2 else None
    )