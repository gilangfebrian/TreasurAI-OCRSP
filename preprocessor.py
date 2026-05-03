import base64
import io
import logging
from PIL import Image, ImageEnhance, ImageFilter
import pdf2image
import pytesseract

logger = logging.getLogger(__name__)

MAX_PAGES = 10
PDF_DPI = 400
MAX_IMAGE_DIM = 4096


def preprocess_file(raw_bytes: bytes, content_type: str) -> tuple[list[dict], int]:
    """
    Returns (image_blocks, total_page_count).
    total_page_count > len(image_blocks) signals truncation.
    """
    if content_type == "application/pdf":
        all_pages = pdf2image.convert_from_bytes(raw_bytes, dpi=PDF_DPI)
        total = len(all_pages)
        pages = all_pages[:MAX_PAGES]
        if total > MAX_PAGES:
            logger.warning("PDF has %d pages; only processing first %d.", total, MAX_PAGES)
        blocks = [_image_to_b64_block(_normalize_image(page)) for page in pages]
        return blocks, total
    else:
        img = Image.open(io.BytesIO(raw_bytes))
        img = _normalize_image(img)
        return [_image_to_b64_block(img)], 1


def _normalize_image(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    if max(img.size) > MAX_IMAGE_DIM:
        ratio = MAX_IMAGE_DIM / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))

    return img


def extract_raw_text(raw_bytes: bytes, content_type: str) -> tuple[str, int]:
    """
    OCR fallback untuk model text-only.
    Returns (ocr_text, total_page_count).
    """
    if content_type == "application/pdf":
        all_pages = pdf2image.convert_from_bytes(raw_bytes, dpi=PDF_DPI)
        total = len(all_pages)
        pages = all_pages[:MAX_PAGES]
        if total > MAX_PAGES:
            logger.warning("PDF has %d pages; only processing first %d for OCR.", total, MAX_PAGES)
        images = [_normalize_image(p) for p in pages]
    else:
        img = Image.open(io.BytesIO(raw_bytes))
        images = [_normalize_image(img)]
        total = 1

    text = _ocr_images(images)
    return text, total


def _ocr_images(images: list[Image.Image]) -> str:
    """
    Jalankan tesseract pada setiap gambar dan gabungkan hasilnya.
    Konversi ke grayscale dulu — tesseract lebih akurat pada mode L.
    """
    parts: list[str] = []
    for i, img in enumerate(images, start=1):
        gray = img.convert("L")
        text = pytesseract.image_to_string(gray, lang="ind+eng", config="--oem 3 --psm 6")
        cleaned = text.strip()
        if cleaned:
            prefix = f"[Halaman {i}]\n" if len(images) > 1 else ""
            parts.append(f"{prefix}{cleaned}")
    return "\n\n".join(parts)


def _image_to_b64_block(img: Image.Image) -> dict:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    encoded = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": encoded},
    }