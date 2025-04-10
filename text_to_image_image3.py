import os

# 设置环境变量代理
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"  # 替换为你的代理地址和端口
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890" # 替换为你的代理地址和端口

# Replace with your actual API key or set it as an environment variable
# Get API key from: https://aistudio.google.com/app/apikey
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

# Replace with your actual API key or set it as an environment variable
# Get API key from: https://aistudio.google.com/app/apikey
client = genai.Client(api_key='AIzaSyBIbskgZ5_35l5p5JzMLWd8lh-NykPlVbs')  # Replace with your key

response = client.models.generate_images(
    model='imagen-3.0-generate-002',
    prompt='Fuzzy bunnies in my kitchen',
    config=types.GenerateImagesConfig(
        number_of_images= 4,
    )
)
for generated_image in response.generated_images:
  image = Image.open(BytesIO(generated_image.image.image_bytes))
  image.show()