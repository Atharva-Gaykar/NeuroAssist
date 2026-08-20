import httpx

CNN_SERVICE_URL = "https://gaykar-cnn.hf.space/classify"

async def classify_with_cnn(image_bytes: bytes):
    files = {
        "file": ("mri.jpg", image_bytes, "image/jpeg")
    }

    try:
        # Create an asynchronous HTTP client
        async with httpx.AsyncClient() as client:
            # Awaiting the network I/O call across the internet!
            response = await client.post(CNN_SERVICE_URL, files=files, timeout=30.0)
            response.raise_for_status()
            data = response.json()

            return {
                "has_tumor": data["has_tumor"],
                "tumor_type": data["tumor_type"],
                "confidence": data["confidence"]
            }
    except httpx.HTTPError as e:
        print(f"Connection Error: {e}")
        return None
    
async def process_tumor_detection(image_bytes: bytes) -> dict:
    try:
        # Awaiting our async microservice call
        cnn_result = await classify_with_cnn(image_bytes)
        
        if not cnn_result:
             raise ValueError("CNN returned no result")

        has_tumor = cnn_result.get("has_tumor", False)
        
        if not has_tumor:
            return {"tumor_type": "No_Tumor_Detected", "confidence": 100.0}
        
        return {
            "tumor_type": cnn_result.get("tumor_type", "Unknown"),
            "confidence": float(cnn_result.get("confidence", 0.0))
        }
    except Exception as e:
        print(f"[CNN ERROR] {e}")
        return {"tumor_type": "Error_Detecting", "confidence": 0.0}