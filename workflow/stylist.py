import base64
import glob
import io
import json
import math
import os
import re
import sys

import cv2
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv
from google import genai
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

POSITIONS = ("front", "back", "left", "right")
VISION_IMAGES = (*POSITIONS, "hand")
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}

VISION_MODEL = "gemini-3.6-flash"
IMAGE_MODEL = "gemini-2.5-flash-image"

mp_pose = mp.solutions.pose
PoseLandmark = mp_pose.PoseLandmark


def fail(message):
    print(f"ERROR: {message}")
    sys.exit(1)


def load_user_input():
    path = os.path.join(INPUT_DIR, "user_input.json")
    if not os.path.exists(path):
        fail(f"missing {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("height_cm", "weight_kg"):
        if key not in data:
            fail(f"user_input.json is missing required field '{key}'")
    return data


def find_image_path(position):
    matches = sorted(
        p for p in glob.glob(os.path.join(INPUT_DIR, f"{position}_image_input.*"))
        if os.path.splitext(p)[1].lower() in SUPPORTED_EXTS
    )
    if not matches:
        fail(f"no {position} image found — expected {INPUT_DIR}/{position}_image_input.<jpg|jpeg|png>")
    return matches[0]


# === STEP 1: BODY MEASUREMENT ===

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def run_pose(path, pose):
    image = cv2.imread(path)
    if image is None:
        fail(f"could not read image {path}")
    height, width = image.shape[:2]
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        fail(f"MediaPipe could not detect a pose in {path}. "
             f"Use a well-lit, full-body photo of a single person (head to feet visible).")

    landmarks_px = {}
    landmarks_norm = {}
    for lm in PoseLandmark:
        point = results.pose_landmarks.landmark[lm.value]
        landmarks_px[lm.name] = (point.x * width, point.y * height)
        landmarks_norm[lm.name] = (point.x, point.y)

    mask = None
    if results.segmentation_mask is not None:
        mask = results.segmentation_mask > 0.5

    return {"landmarks_px": landmarks_px, "landmarks_norm": landmarks_norm, "mask": mask}


def mask_span_px(mask, y_norm):
    if mask is None:
        return None
    h = mask.shape[0]
    row = min(max(int(round(y_norm * (h - 1))), 0), h - 1)
    cols = np.flatnonzero(mask[row])
    if cols.size < 2:
        return None
    return float(cols[-1] - cols[0])


def average_or_fallback(px_values, scale, fallback_cm):
    valid = [v for v in px_values if v]
    if valid:
        return sum(valid) / len(valid) * scale
    return fallback_cm


def ellipse_circumference(width, depth):
    a = width / 2.0
    b = depth / 2.0
    if a + b == 0:
        return 0.0
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))


def bmi_category(bmi):
    if bmi < 18.5:
        return "underweight"
    if bmi < 25:
        return "normal"
    if bmi < 30:
        return "overweight"
    return "obese"


def classify_body_shape(chest, waist, hip):
    # Checked in spec order; the first matching rule wins.
    if abs(chest - hip) < 5 and (chest - waist) >= 9:
        return "hourglass"
    if hip > chest and (hip - waist) / hip > 0.10:
        return "pear"
    if chest > hip and (chest - hip) / chest > 0.08:
        return "inverted_triangle"
    if abs(chest - waist) < 9 and abs(hip - waist) < 9:
        return "rectangle"
    if waist > hip * 0.85:
        return "apple"
    return "rectangle"


def measure_body(user_input, poses):
    front, back, left, right = (poses[p] for p in POSITIONS)
    fp = front["landmarks_px"]
    fn = front["landmarks_norm"]

    height_cm = float(user_input["height_cm"])
    weight_kg = float(user_input["weight_kg"])

    body_height_px = dist(fp["NOSE"], midpoint(fp["LEFT_ANKLE"], fp["RIGHT_ANKLE"])) * 1.08
    if body_height_px <= 0:
        fail("could not determine body height from the front image landmarks")
    scale = height_cm / body_height_px

    # Normalized y-levels so the same relative rows can be scanned on every image.
    shoulder_y = (fn["LEFT_SHOULDER"][1] + fn["RIGHT_SHOULDER"][1]) / 2.0
    hip_y = (fn["LEFT_HIP"][1] + fn["RIGHT_HIP"][1]) / 2.0
    chest_y = shoulder_y + 0.25 * (hip_y - shoulder_y)
    waist_y = shoulder_y + 0.60 * (hip_y - shoulder_y)

    shoulder_px = dist(fp["LEFT_SHOULDER"], fp["RIGHT_SHOULDER"])
    hip_px = dist(fp["LEFT_HIP"], fp["RIGHT_HIP"])

    def widths_at(y):
        return [mask_span_px(front["mask"], y), mask_span_px(back["mask"], y)]

    def depths_at(y):
        return [mask_span_px(left["mask"], y), mask_span_px(right["mask"], y)]

    chest_width = average_or_fallback(widths_at(chest_y), scale, shoulder_px * 0.90 * scale)
    waist_width = average_or_fallback(widths_at(waist_y), scale, shoulder_px * 0.75 * scale)
    hip_width = average_or_fallback(widths_at(hip_y), scale, hip_px * scale)

    chest_depth = average_or_fallback(depths_at(chest_y), scale, chest_width * 0.65)
    waist_depth = average_or_fallback(depths_at(waist_y), scale, waist_width * 0.70)
    hip_depth = average_or_fallback(depths_at(hip_y), scale, hip_width * 0.70)

    chest_circ = ellipse_circumference(chest_width, chest_depth)
    waist_circ = ellipse_circumference(waist_width, waist_depth)
    hip_circ = ellipse_circumference(hip_width, hip_depth)

    arm_length = (dist(fp["LEFT_SHOULDER"], fp["LEFT_ELBOW"]) + dist(fp["LEFT_ELBOW"], fp["LEFT_WRIST"]) +
                  dist(fp["RIGHT_SHOULDER"], fp["RIGHT_ELBOW"]) + dist(fp["RIGHT_ELBOW"], fp["RIGHT_WRIST"])) / 2 * scale
    inseam = (dist(fp["LEFT_HIP"], fp["LEFT_KNEE"]) + dist(fp["LEFT_KNEE"], fp["LEFT_ANKLE"]) +
              dist(fp["RIGHT_HIP"], fp["RIGHT_KNEE"]) + dist(fp["RIGHT_KNEE"], fp["RIGHT_ANKLE"])) / 2 * scale
    torso_length = dist(midpoint(fp["LEFT_SHOULDER"], fp["RIGHT_SHOULDER"]),
                        midpoint(fp["LEFT_HIP"], fp["RIGHT_HIP"])) * scale

    bmi = weight_kg / (height_cm / 100.0) ** 2

    return {
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "bmi": round(bmi, 1),
        "bmi_category": bmi_category(bmi),
        "shoulder_width_cm": round(shoulder_px * scale, 1),
        "chest_circumference_cm": round(chest_circ, 1),
        "waist_circumference_cm": round(waist_circ, 1),
        "hip_circumference_cm": round(hip_circ, 1),
        "arm_length_cm": round(arm_length, 1),
        "inseam_cm": round(inseam, 1),
        "torso_length_cm": round(torso_length, 1),
        "body_shape_estimate": classify_body_shape(chest_circ, waist_circ, hip_circ),
    }


# === STEP 2: GEMINI VISION ===

SYSTEM_INSTRUCTION = (
    "You are an expert personal stylist and body analyst. You receive 4 views of a person "
    "(front, back, left, right), a close-up photo of their hand for skin tone, precise body "
    "measurements from a pose estimation tool, and a personality description. "
    "Analyze everything together and return ONLY a JSON object."
)

RESPONSE_SCHEMA = """{
  "visual_analysis": {
    "body_shape": "hourglass|pear|apple|rectangle|inverted_triangle",
    "body_shape_confidence": "high|medium|low",
    "skin_tone": "fair|medium|tanned|olive|deep",
    "skin_undertone": "cool|warm|neutral",
    "face_shape": "oval|round|square|heart|diamond|oblong",
    "hair_color": "black|brown|blonde|red|grey|white|other",
    "hair_texture": "straight|wavy|curly|coily",
    "posture": "upright|slightly_slouched|slouched",
    "additional_notes": "<one sentence on anything visually notable>"
  },
  "personality_profile": {
    "big_five": {
      "openness": <1-5>,
      "conscientiousness": <1-5>,
      "extraversion": <1-5>,
      "agreeableness": <1-5>,
      "neuroticism": <1-5>
    },
    "style_archetype": "classic|bohemian|minimalist|maximalist|sporty|creative|edgy",
    "style_boldness": "conservative|moderate|bold",
    "lifestyle": {
      "occupation_type": "corporate|creative|academic|medical|trades|student|other",
      "activity_level": "sedentary|moderate|athletic",
      "dress_code_context": "business_formal|business_casual|smart_casual|casual|streetwear"
    },
    "preferences": {
      "favorite_colors": [],
      "disliked_styles": []
    }
  },
  "style_recommendations": {
    "color_palette": {
      "primary": ["<3 hex colors>"],
      "accent": ["<2 hex colors>"],
      "avoid": ["<2 hex colors>"],
      "rationale": "<one sentence>"
    },
    "silhouettes": {
      "recommended": ["<list of silhouettes/cuts>"],
      "avoid": ["<list to avoid>"],
      "rationale": "<one sentence>"
    },
    "outfit_suggestions": [
      {
        "occasion": "work|casual|evening|weekend",
        "outfit": "<full outfit description>",
        "why_it_works": "<one sentence tying body shape + personality>",
        "image_generation_prompt": "<detailed prompt describing the outfit on a person matching their body type and coloring, ready to feed into an image generation model>"
      }
    ],
    "accessories": ["<list>"],
    "hair_recommendation": "<one sentence>",
    "style_summary": "<2-3 sentence personal style summary>"
  }
}"""

STYLING_RULES = """- pear shape     -> structured/bright tops, A-line or bootcut bottoms, avoid clingy skirts
- apple shape    -> V-neck or empire waist, vertical patterns, avoid belts at natural waist
- hourglass      -> wrap dresses, belted jackets, high-waist bottoms, avoid shapeless cuts
- rectangle      -> peplum, ruffles, color blocking, belts to create waist curve
- inv. triangle  -> V-neck tops, fuller/patterned bottoms, avoid shoulder-widening necklines
- cool undertone -> blues, purples, jewel tones
- warm undertone -> coral, orange, warm reds, olive, camel
- high extraversion (4-5)  -> bold saturated colors, statement pieces
- high conscientiousness   -> tailored, neutrals, timeless cuts
- high openness            -> eclectic prints, diverse color palette
- high neuroticism         -> comfortable, safe classics, muted tones"""


def build_vision_prompt(measurements, personality):
    return f"""The 5 images that follow are, in order: FRONT, BACK, LEFT side, RIGHT side views of the same person,
then a close-up of their HAND. Use the hand close-up as the primary reference for skin_tone and skin_undertone
(e.g. vein color, how the skin reads under the lighting), cross-checked against the body views.

BODY MEASUREMENTS (from pose estimation; body_shape_estimate is ratio-based — cross-validate it against the images):
{json.dumps(measurements, indent=2)}

PERSONALITY DESCRIPTION (from the user):
{personality or "(not provided)"}

STYLING RULES — apply these when forming recommendations:
{STYLING_RULES}

INSTRUCTIONS:
- Generate exactly 4 outfit_suggestions, one each for occasions: work, casual, evening, weekend.
- For each outfit, write a detailed image_generation_prompt describing the full outfit worn by a person
  with this body shape, skin tone, and hair — suitable for a fashion image generation model
  (full-body shot, pose, lighting, background).
- Return ONLY a JSON object matching this exact structure:
{RESPONSE_SCHEMA}"""


def image_block(path):
    mime_type = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return {"type": "image", "data": data, "mime_type": mime_type}


def api_error_message(exc):
    # SDK errors embed the whole JSON error body; surface just the human-readable message.
    match = re.search(r"""['"]message['"]:\s*['"](.+?)['"](?:,|\})""", str(exc))
    return match.group(1) if match else str(exc)[:300]


def check_completed(interaction):
    if interaction.status != "completed":
        details = "; ".join(str(e) for e in interaction.errors or []) or "no details"
        raise RuntimeError(f"interaction ended with status '{interaction.status}': {details}")


def analyze_with_gemini(client, measurements, personality, image_paths):
    inputs = [
        {"type": "text", "text": build_vision_prompt(measurements, personality)},
        *(image_block(image_paths[p]) for p in VISION_IMAGES),
    ]

    for attempt in (1, 2):
        try:
            # store=False: the body photos should not be retained server-side.
            interaction = client.interactions.create(
                model=VISION_MODEL,
                system_instruction=SYSTEM_INSTRUCTION,
                input=inputs,
                response_format={"type": "text", "mime_type": "application/json"},
                store=False,
            )
        except Exception as exc:
            fail(f"Gemini API request failed: {api_error_message(exc)}")
        try:
            check_completed(interaction)
            result = json.loads(interaction.output_text or "")
            if isinstance(result, dict):
                return result
            print(f"WARNING: Gemini Vision returned JSON that is not an object (attempt {attempt}/2)")
        except (ValueError, RuntimeError) as exc:
            print(f"WARNING: Gemini Vision response was not valid JSON (attempt {attempt}/2): {exc}")
    fail("Gemini Vision did not return valid JSON after a retry")


# === STEP 3: IMAGE GENERATION ===

def generate_outfit_images(client, outfits):
    saved = {}
    for outfit in outfits:
        occasion = str(outfit.get("occasion", "outfit"))
        prompt = outfit.get("image_generation_prompt")
        name = re.sub(r"[^a-z0-9]+", "_", occasion.lower()).strip("_") or "outfit"
        path = os.path.join(OUTPUT_DIR, f"outfit_{name}.png")
        if not prompt:
            print(f"ERROR: outfit '{occasion}' has no image_generation_prompt — skipping")
            continue
        try:
            interaction = client.interactions.create(
                model=IMAGE_MODEL,
                input=prompt,
                response_format={"type": "image", "aspect_ratio": "3:4"},
                store=False,
            )
            check_completed(interaction)
            if not interaction.output_image or not interaction.output_image.data:
                raise RuntimeError("no image returned (the prompt may have been filtered)")
            # Images arrive as base64 JPEG; re-encode so the .png filename is accurate.
            image_bytes = base64.b64decode(interaction.output_image.data)
            Image.open(io.BytesIO(image_bytes)).save(path, format="PNG")
            saved[occasion] = path
            print(f"  generated {path}")
        except Exception as exc:
            print(f"ERROR: image generation failed for '{occasion}': {api_error_message(exc)}")
    return saved


# === FINAL OUTPUT ===

def print_report(style_recommendations, image_files):
    print("\n" + "=" * 70)
    print("STYLE SUMMARY")
    print("=" * 70)
    print(style_recommendations.get("style_summary", "(none returned)"))

    print("\n" + "=" * 70)
    print("OUTFIT SUGGESTIONS")
    print("=" * 70)
    for outfit in style_recommendations.get("outfit_suggestions", []):
        occasion = str(outfit.get("occasion", "outfit"))
        print(f"\n[{occasion.upper()}]")
        print(f"  Outfit:       {outfit.get('outfit', '')}")
        print(f"  Why it works: {outfit.get('why_it_works', '')}")

    print("\n" + "=" * 70)
    print("GENERATED IMAGES")
    print("=" * 70)
    if image_files:
        for occasion, path in image_files.items():
            print(f"  {occasion:<8} {path}")
    else:
        print("  (no images were generated)")


def main():
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        fail(f"GEMINI_API_KEY is not set — add it to {os.path.join(BASE_DIR, '.env')}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    user_input = load_user_input()
    image_paths = {p: find_image_path(p) for p in VISION_IMAGES}

    print("Step 1: measuring body with MediaPipe...")
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True) as pose:
        poses = {p: run_pose(image_paths[p], pose) for p in POSITIONS}
    measurements = measure_body(user_input, poses)
    print(json.dumps(measurements, indent=2))

    print(f"\nStep 2: analyzing with {VISION_MODEL}...")
    client = genai.Client(api_key=api_key)
    analysis = analyze_with_gemini(client, measurements, user_input.get("personality", ""), image_paths)
    style_recommendations = analysis.get("style_recommendations", {})

    print(f"\nStep 3: generating outfit images with {IMAGE_MODEL}...")
    image_files = generate_outfit_images(client, style_recommendations.get("outfit_suggestions", []))

    output = {
        "body_profile": measurements,
        "visual_analysis": analysis.get("visual_analysis", {}),
        "personality_profile": analysis.get("personality_profile", {}),
        "style_recommendations": style_recommendations,
    }
    output_path = os.path.join(OUTPUT_DIR, "output.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=Fse)

    print_report(style_recommendations, image_files)
    print(f"\nFull results written to {output_path}")


if __name__ == "__main__":
    main()
