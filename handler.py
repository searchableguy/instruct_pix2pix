import PIL
import requests
import torch
import runpod
import os
from diffusers import StableDiffusionInstructPix2PixPipeline
from runpod.serverless.utils import upload, validator

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16
).to("cuda")

INPUT_VALIDATIONS = {
    "prompt": {"type": str, "required": True},
    "url": {"type": str, "required": True},
    "negative_prompt": {"type": str, "required": False},
    "num_inference_steps": {"type": int, "required": False},
    "guidance_scale": {"type": float, "required": False},
    "seed": {"type": int, "required": False},
}


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def edit_image(
    url: str,
    prompt: str,
    negative_prompt: str = None,
    steps: int = 20,
    scale: float = 7.0,
    seed: int = 1,
):
    image = download_image(url)
    with torch.no_grad(), torch.autocast("cuda"):
        torch.manual_seed(seed)
        response = pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=steps,
            guidance_scale=scale,
            negative_prompt=negative_prompt,
        )

    return response.images[0]


def runpod_handler(event):
    event_input = event["input"]
    input_errors = validator.validate(event_input, INPUT_VALIDATIONS)
    if input_errors:
        return {"error": input_errors}

    seed = event_input.get("seed", int.from_bytes(os.urandom(2), "big"))
    num_inference_steps = event_input.get("num_inference_steps", 25)
    guidance_scale = event_input.get("guidance_scale", 7.0)
    prompt = event_input.get("prompt", None)
    negative_prompt = event_input.get("negative_prompt", None)
    url = event_input.get("url", None)

    images = edit_image(
        url=url,
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=num_inference_steps,
        scale=guidance_scale,
        seed=seed,
    )

    image_paths = []
    for i, image in enumerate(images):
        image_path = f"/tmp/out-{i}.png"
        image.save(image_path)
        image_paths.append(image_path)

    result = []
    for index, img_path in enumerate(image_paths):
        image_url = upload.upload_image(event["id"], img_path, index)
        result.append({"image": image_url, "seed": event_input["seed"] + index})

    return result


runpod.serverless.start({"handler": runpod_handler})
