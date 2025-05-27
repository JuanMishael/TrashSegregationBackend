# MobileNet Flask API

A simple Flask API that serves a MobileNet `.tflite` model to classify uploaded images via HTTP POST.

## Features

- Accepts image uploads
- Uses TensorFlow Lite for fast inference
- Returns class label and confidence
- Cross-Origin (CORS) enabled

## How to Use

1. Install dependencies:
   pip install -r requirements.txt

2. run the app

3. Send POST request to `/predict` with `form-data` key `image`.

## Sample Response

```json
{
  "label": "cat",
  "confidence": 0.987
}
```

# Note:

this accept images or frames of a video, (make sure to send file formated as .jpg)

## 💡 TensorFlow on Raspberry Pi

TensorFlow is not in `requirements.txt` because it's architecture-specific.

install on Raspberry Pi (ARM64)

di nako nag lagay specific version ng tensorflow dito since di ako maalam sa raspberry, idunno what to put, tensorflow lite would work, try to install the latest nlang
