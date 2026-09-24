import io
from PIL import Image
from fastapi import HTTPException, UploadFile, status

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit for research X-ray scans


def validate_image_file(file: UploadFile, file_bytes: bytes) -> dict:
    """
    Validates uploaded image file extension, MIME type, size, and integrity.
    Returns metadata dict containing dimensions and format.
    """
    # 1. Filename validation
    filename = file.filename or ""
    extension = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{extension}'. Only JPG, JPEG, and PNG images are supported."
        )

    # 2. MIME type validation
    if file.content_type not in ALLOWED_MIME_TYPES and file.content_type != "application/octet-stream":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid MIME type '{file.content_type}'. Must be image/jpeg or image/png."
        )

    # 3. File size check
    size = len(file_bytes)
    if size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty."
        )
    if size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB."
        )

    # 4. Image integrity check using Pillow
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()  # Verifies file integrity without decoding whole image
        # Re-open to read dimensions since verify() damages the stream
        image = Image.open(io.BytesIO(file_bytes))
        width, height = image.size
        img_format = image.format or extension.replace(".", "").upper()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted or invalid image file. Could not parse image: {str(e)}"
        )

    return {
        "width": width,
        "height": height,
        "format": img_format,
        "size_bytes": size,
        "dimensions": f"{width}x{height}px"
    }
