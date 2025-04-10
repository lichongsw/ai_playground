from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import time
import PIL.Image

image = PIL.Image.open('gemini-native-image.png')

client = genai.Client(api_key='AIzaSyBIbskgZ5_35l5p5JzMLWd8lh-NykPlVbs')

text_input = ('Hi, This is a picture of me funny pig.'
            'Can you add an small Ultraman who sits on the pig?')

response = client.models.generate_content(
    model="gemini-2.0-flash-exp-image-generation",
    contents=[text_input, image],
    config=types.GenerateContentConfig(
       response_modalities=['Text', 'Image']
    )
)

for part in response.candidates[0].content.parts:
  if part.text is not None:
    print(part.text)
  elif part.inline_data is not None:
    image = Image.open(BytesIO(part.inline_data.data))
    image.save('gemini-native-image'+str(time.time())+'.png')
