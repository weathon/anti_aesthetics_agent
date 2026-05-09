import json
import os
import sys
import anyio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import base64
import dotenv
dotenv.load_dotenv()


with open("classes_new.json", "r") as f:
    classes = f.read()

os.makedirs("captions", exist_ok=True)

prompt = f"""
<role>
You are an expert visual analyst specializing in identifying anti-aesthetic elements in images and producing high-quality captions for dataset curation.
</role>

<task>
Given an image, perform three steps:

1. Identify any anti-aesthetic elements present in the image, drawn from the taxonomy in <anti_aesthetics_taxonomy>. Record matches by their fully-qualified item names (e.g., `clarity_and_focus.intentional_blur`). If the image contains no strong anti-aesthetic elements, return an empty list and skip step 2. Note that it has to show clearly identifiable anti-aesthetic elements to be included in the list; the list should be empty if the image does not show STRONG anti-aesthetic elements, even if it is somewhat anti-aesthetics. Include only one tag per major category (i.e., no clarity_and_focus.digital_artifacts and clarity_and_focus.intentional_blur).

2. If at least one anti-aesthetic element was identified, generate two captions:
   - `objective_caption`: a detailed, objective description of the image content only. Do NOT mention or describe any anti-aesthetic elements.
   - `anti_aesthetic_caption`: a description that covers BOTH the image content AND the anti-aesthetic elements present (not just the category in <anti_aesthetics_taxonomy> but any anti-aesthetic elements).

<anti_aesthetics_taxonomy>
{classes}
</anti_aesthetics_taxonomy>

<output_format>
A single JSON object with this exact shape, with no additional text or formatting:
{{
  "anti_aesthetic_elements": ["category.element_name", ...],
  "objective_caption": "..." | null,
  "anti_aesthetic_caption": "..." | null
}}

If `anti_aesthetic_elements` is empty, both caption fields must be `null`.
</output_format>

<constraints>
- Use only element names that appear in <anti_aesthetics_taxonomy>. Do not invent new categories.
- The objective caption must not leak anti-aesthetic descriptors (e.g., do not say "blurry", "poorly lit", "cluttered").
- Be specific and concrete; avoid vague descriptors.
</constraints>
"""
import time

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)


def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def process_image(sample):
    while True:
        try:
            image_path = sample["image_path"]
            caption_name = os.path.basename(image_path).split(".")[0]+".json"
            caption_path = f"captions/{caption_name}"
            if os.path.exists(caption_path):
                with open(caption_path, "r") as f:
                    try:
                        existing_data = json.load(f)
                        if "response" in existing_data:
                            print(f"Caption already exists for {image_path}, skipping.")
                            return
                    except json.JSONDecodeError:
                        pass

            response = client.chat.completions.create(
                model="qwen/qwen3.6-35b-a3b",
                messages=[
                        {
                            "role": "system",
                            "content": prompt
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}}
                            ]
                        }
                    ],
                response_format={"type": "json_object" },
                extra_body={"reasoning": {"effort": "low"}}
            )
            json_object = response.choices[0].message.content
            json_object = json.loads(json_object)
            sample["response"] = json_object
            with open(caption_path, "w") as f:
                json.dump(sample, f, indent=4)
            return
        except Exception as e:
            print(f"Error processing {sample['image_path']}: {e}")
            time.sleep(5)  # Wait before retrying


import json
import tqdm

with open("dataset.json", "r") as f:
    dataset = json.load(f)

image_paths = []

result = []

inputs = []
for commit in dataset.keys():
    query = dataset[commit]["query"]
    message = dataset[commit]["message"]
    images = dataset[commit]["images"]
    for image in images:
        if image not in image_paths:
            inputs.append(
                {
                    "commit": commit,
                    "query": query,
                    "message": message,
                    "image_path": image
                }
            )
        image_paths.append(image)


print(len(inputs))

with ThreadPoolExecutor(max_workers=100) as executor:
    results = list(tqdm.tqdm(executor.map(process_image, inputs), total=len(inputs)))

        