import aiohttp
import asyncio
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from io import BytesIO
import logging
import requests
from sklearn.cluster import KMeans

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your Hugging Face API token
API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
headers = {"Authorization": "Bearer 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"}

async def fetch_image_with_huggingface(prompt, max_retries=5, delay=60):
    payload = {"inputs": prompt}
    logger.info(f"Sending payload to Hugging Face API: {payload}")
    
    for attempt in range(max_retries):
        async with aiohttp.ClientSession() as session:
            async with session.post(API_URL, headers=headers, json=payload, ssl=False) as response:
                if response.status == 200:
                    try:
                        image_data = await response.read()
                        logger.info(f"Received image data from Hugging Face API: {len(image_data)} bytes")
                        return image_data
                    except Exception as e:
                        logger.error(f"Error reading image data: {e}")
                elif response.status == 503:
                    logger.warning(f"Model is loading, retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                elif response.status == 400:
                    logger.error(f"Error: Received status code {response.status}, Response: {await response.text()}")
                    break
                else:
                    logger.error(f"Error: Received status code {response.status}, Response: {await response.text()}")
                    break

    logger.error("Error: No image data received from the API after retries")
    return None

def fetch_image_from_bytes(image_data):
    try:
        # Check if the image_data is a valid image
        image = Image.open(BytesIO(image_data)).convert('RGBA')
        return image
    except UnidentifiedImageError as e:
        logger.error(f"Error: {e}")
        logger.error(f"Received data: {image_data[:100]}...")  # Log the first 100 bytes of the received data
        return None

async def generate_image_from_text(prompt, output_path):
    image_data = await fetch_image_with_huggingface(prompt)
    if image_data is None:
        logger.error("Error: No image data received from the API")
        return None
    image = fetch_image_from_bytes(image_data)
    if image is None:
        logger.error("Error: Failed to convert image data to an image")
        return None
    image.save(output_path)


def load_image(image_path):
    if image_path.startswith('http://') or image_path.startswith('https://'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert('RGBA')
    else:
        image = Image.open(image_path).convert('RGBA')
    return np.array(image)

def preprocess_image(image, resize_dim=(100, 100)):
    image = cv2.resize(image, resize_dim)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    return image

def extract_colors(image, num_colors=5):
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(image)
    colors = kmeans.cluster_centers_
    return colors

def remove_background(logo):
    if logo.shape[2] == 3:
        logo = cv2.cvtColor(logo, cv2.COLOR_RGB2RGBA)
    
    gray = cv2.cvtColor(logo, cv2.COLOR_RGBA2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    logo[:, :, 3] = mask
    return logo


def overlay_logo(image_path, logo_path):
    image = load_image(image_path)
    logo = load_image(logo_path)
    
    logo_no_bg = remove_background(logo)

    logo_height, logo_width = 100, 100
    logo_no_bg = cv2.resize(logo_no_bg, (logo_width, logo_height), interpolation=cv2.INTER_AREA)
    
    if logo_no_bg.shape[2] == 4:
        b, g, r, a = cv2.split(logo_no_bg)
        logo_rgb = cv2.merge((b, g, r))
        mask = a
    else:
        logo_rgb = logo_no_bg
        mask = np.ones((logo_no_bg.shape[0], logo_no_bg.shape[1]), dtype=np.uint8) * 255

    roi = image[:logo_height, :logo_width]

    logger.info(f"ROI shape: {roi.shape}, mask shape: {mask.shape}")

    if roi.shape[2] == 4:
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_RGBA2RGB)
    else:
        roi_rgb = roi

    mask = cv2.resize(mask, (roi_rgb.shape[1], roi_rgb.shape[0]))

    mask_inv = cv2.bitwise_not(mask)
    mask_inv_rgb = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    logger.info(f"Resized mask shape: {mask.shape}, mask_inv_rgb shape: {mask_inv_rgb.shape}")

    img_bg = cv2.bitwise_and(roi_rgb, mask_inv_rgb)
    logo_fg = cv2.bitwise_and(logo_rgb, mask_rgb)

    logger.info(f"img_bg shape: {img_bg.shape}, logo_fg shape: {logo_fg.shape}")

    if img_bg.shape != logo_fg.shape:
        logger.error(f"Shape mismatch: img_bg shape: {img_bg.shape}, logo_fg shape: {logo_fg.shape}")
        return image

    dst = cv2.add(img_bg, logo_fg)
    
    if image.shape[2] == 4:
        b, g, r = cv2.split(dst)
        a = mask
        dst = cv2.merge((b, g, r, a))

    image[:logo_height, :logo_width] = dst

    return image


    dst = cv2.add(img_bg, logo_fg)
    
    # Convert dst back to RGBA if the original image is in RGBA format
    if image.shape[2] == 4:
        b, g, r = cv2.split(dst)
        a = mask
        dst = cv2.merge((b, g, r, a))

    image[:logo_height, :logo_width] = dst

    return image




def create_banner(branding_colors, occasion, text, output_path):
    banner_width, banner_height = 1024, 256
    base_color = tuple(branding_colors[0].astype(int))
    gradient = Image.new('RGB', (banner_width, banner_height), base_color)
    for y in range(banner_height):
        for x in range(banner_width):
            gradient.putpixel((x, y), (
                min(base_color[0] - int(50 * (y / banner_height)), 255),
                min(base_color[1] - int(50 * (y / banner_height)), 255),
                min(base_color[2] - int(50 * (y / banner_height)), 255)
            ))
    draw = ImageDraw.Draw(gradient)
    try:
        font = ImageFont.truetype("arial.ttf", 60)
        font_small = ImageFont.truetype("arial.ttf", 45)
    except IOError:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    logo_path = 'static/logo.png'
    logo = Image.open(logo_path).convert("RGBA")
    logo.thumbnail((150, 150), Image.LANCZOS)
    logo = remove_background(np.array(logo))
    logo = Image.fromarray(logo)
    logo_x = 30
    logo_y = (banner_height - logo.height) // 2
    gradient.paste(logo, (logo_x, logo_y), logo)
    occasion_text = f"{occasion} Banner"
    text_bbox = draw.textbbox((0, 0), occasion_text, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    text_x = logo_x + logo.width + 30
    text_y = (banner_height - text_height) // 2 - 30
    draw.text((text_x, text_y), occasion_text, fill=(255, 255, 255), font=font)
    if text:
        additional_text_bbox = draw.textbbox((0, 0), text, font=font_small)
        additional_text_width, additional_text_height = additional_text_bbox[2] - additional_text_bbox[0], additional_text_bbox[3] - additional_text_bbox[1]
        additional_text_x = text_x
        additional_text_y = text_y + text_height + 10
        draw.text((additional_text_x, additional_text_y), text, fill=(255, 255, 255), font=font_small)
    shadow_color = (0, 0, 0)
    draw.text((text_x + 2, text_y + 2), occasion_text, font=font, fill=shadow_color)
    draw.text((additional_text_x + 2, additional_text_y + 2), text, font=font_small, fill=shadow_color)
    draw.line((0, banner_height - 5, banner_width, banner_height - 5), fill=(255, 255, 255), width=5)
    gradient.save(output_path)

def fetch_image_from_bytes(image_data):
    try:
        image = Image.open(BytesIO(image_data)).convert('RGBA')
        return image
    except UnidentifiedImageError as e:
        logger.error(f"Error: {e}")
        logger.error(f"Received data: {image_data[:100]}...")  # Log the first 100 bytes of the received data
        return None
