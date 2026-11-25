from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image, UnidentifiedImageError


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "data" / "input"
OUTPUT_FILE = BASE_DIR / "data" / "output" / "result.jpg"
LOG_FILE = BASE_DIR / "logs" / "history.txt"
FILTERS = "Свёртка 3x3, Серый мир"


def write_log(filename, status):
    now = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    with open(LOG_FILE, "a", encoding="UTF-8") as log:
        log.write(
            f"[{now}] Файл: {filename} | Фильтр: {FILTERS} | "
            f"Статус: {status}\n"
        )


def convolution(image):
    """Усредняющая свёртка RGB-изображения ядром 3x3."""
    pixels = np.asarray(image, dtype=np.float64)
    height, width, _ = pixels.shape
    padded = np.pad(pixels, ((1, 1), (1, 1), (0, 0)), mode="edge")
    result = np.zeros_like(pixels)

    for row in range(3):
        for col in range(3):
            result += padded[row:row + height, col:col + width] / 9

    return np.clip(result, 0, 255).astype(np.uint8)


def gray_world(pixels):
    """Коррекция баланса белого по методу «Серого мира»."""
    pixels = pixels.astype(np.float64)
    channel_means = pixels.mean(axis=(0, 1))
    common_mean = channel_means.mean()
    coefficients = np.ones(3, dtype=np.float64)

    for channel in range(3):
        if channel_means[channel] > 0:
            coefficients[channel] = common_mean / channel_means[channel]

    corrected = pixels * coefficients
    return np.clip(corrected, 0, 255).astype(np.uint8)


def flatten_transparency(source):
    """Заменяет прозрачность PNG на чёрный фон перед сохранением в JPEG."""
    rgba = source.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def process_image(filename):
    if (not filename or filename in (".", "..")
            or "/" in filename or "\\" in filename):
        print("Ошибка: Некорректное имя файла")
        write_log(filename, "Ошибка имени файла")
        return

    file_path = INPUT_DIR / filename

    try:
        with Image.open(file_path) as source:
            image = flatten_transparency(source)
            pixels = convolution(image)
            pixels = gray_world(pixels)
            result = Image.fromarray(pixels)
            result.save(OUTPUT_FILE, format="JPEG")

    except FileNotFoundError:
        print("Ошибка: Файл не найден")
        write_log(filename, "Файл не найден")
    except PermissionError:
        print(
            f"Ошибка: Нет прав на доступ к файлу {filename}. "
            "Обратитесь к администратору"
        )
        write_log(filename, "Ошибка доступа")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        print(f"Ошибка обработки изображения: {error}")
        write_log(filename, "Ошибка обработки")
    else:
        print("Изображение успешно обработано!")
        print(f"Результат сохранён: {OUTPUT_FILE}")
        write_log(filename, "Успешно")


def main():
    print("Лабораторная работа № 2. Вариант 29")
    print("Свёртка 3x3 и баланс белого методом «Серого мира»")
    filename = input("Введите имя файла: ").strip()
    try:
        process_image(filename)
    except OSError as error:
        # Например, отсутствует каталог logs или нет прав на запись журнала.
        print(f"Ошибка файловой системы: {error}")


if __name__ == "__main__":
    main()
