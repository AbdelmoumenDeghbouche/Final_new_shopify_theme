"""
ULTIMATE UGC VIDEO GENERATION SYSTEM V3.0
Enhanced with Intelligent Product-Aware Sequential Prompting
Fixes teleporting/morphing issues with detailed action sequences
"""

import os
import json
import base64
import random
from typing import List, Dict, Optional, Literal, Tuple
from PIL import Image
from io import BytesIO
from datetime import datetime
from enum import Enum

# Core AI clients
from google import genai
from google.genai import types as genai_types
from google.cloud import aiplatform
from openai import OpenAI
import requests
import time


class VideoStyle(Enum):
    """UGC video style types"""

    TALKING_REVIEW = "talking_review"
    SILENT_DEMO = "silent_demonstration"
    UNBOXING = "unboxing"
    LIFESTYLE = "lifestyle"
    TESTIMONIAL = "testimonial"
    COMPARISON = "comparison"
    TUTORIAL = "tutorial"
    HAUL = "haul"


class ProductUsageAnalyzer:
    """Analyzes how products should be used and generates realistic action sequences"""

    @staticmethod
    def analyze_product_usage(product_analysis: Dict, openai_client: OpenAI) -> Dict:
        """
        Generate detailed usage instructions based on product type
        """
        usage_prompt = f"""You are an expert in product demonstrations and UGC content creation.
        
        Based on this product analysis, provide EXTREMELY DETAILED usage instructions that will prevent unrealistic video generation (like teleporting or morphing).
        
        Product Details:
        - Name: {product_analysis.get('product_name')}
        - Category: {product_analysis.get('category')}
        - Type: {product_analysis.get('shape_type')}
        - Opening: {product_analysis.get('opening_type')}
        - Description: {product_analysis.get('ultra_precise_description')}
        
        Create a JSON response with:
        
        1. "product_type_category": Identify the general category (beauty_tool, skincare_application, tech_device, food_consumption, fashion_accessory, home_appliance, etc.)
        
        2. "preparation_steps": List of actions BEFORE using the product
        Example for hair tool: ["pick up product from table", "check temperature setting", "section hair with fingers"]
        
        3. "main_usage_sequence": DETAILED step-by-step usage (minimum 5 steps)
        Example for curling iron: 
        ["hold iron vertically with right hand",
        "take small section of hair with left hand", 
        "open iron clamp with thumb",
        "place hair section near roots",
        "close clamp gently",
        "slowly rotate iron 180 degrees away from face",
        "slide iron down hair length while rotating",
        "hold at ends for 3 seconds",
        "release clamp and pull iron away",
        "let curl fall naturally"]
        
        4. "physical_interactions": How body parts interact with product
        Example: {{"hands": "right hand holds handle, thumb operates clamp", "hair": "wrapped around barrel in sections", "face": "tilted slightly to side"}}
        
        5. "temporal_markers": Time-based cues for realistic pacing
        Example: ["0-2 seconds: picking up product", "2-4 seconds: sectioning hair", "4-8 seconds: curling motion"]
        
        6. "common_mistakes_to_avoid": What NOT to show
        Example: ["hair instantly curling", "product floating", "instant transformations"]
        
        7. "realistic_details": Small details that make it authentic
        Example: ["slight steam from hot iron", "hair spring back after release", "adjusting grip mid-use"]
        
        8. "camera_friendly_angles": Best angles to show usage clearly
        Example: ["side profile for curling", "over shoulder for application", "front view for results"]
        
        Be EXTREMELY specific about hand positions, movements, timing, and transitions."""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in product demonstrations.",
                },
                {"role": "user", "content": usage_prompt},
            ],
            response_format={"type": "json_object"},
        )

        return json.loads(response.choices[0].message.content)


class IntelligentPromptGenerator:
    """Generates highly detailed, sequential prompts for realistic video generation"""

    @staticmethod
    def generate_sequential_prompt(
        scene_type: str,
        product_analysis: Dict,
        usage_analysis: Dict,
        character_profile: Dict,
        setting: str,
    ) -> Tuple[str, str]:
        """
        Generate image and video prompts with proper sequential actions
        """

        # Build temporal sequence for video
        if scene_type == VideoStyle.SILENT_DEMO.value:
            # Silent demonstration - focus on clear, sequential actions
            video_prompt = IntelligentPromptGenerator._build_silent_demo_prompt(
                product_analysis, usage_analysis, character_profile, setting
            )
        elif scene_type == VideoStyle.TALKING_REVIEW.value:
            video_prompt = IntelligentPromptGenerator._build_talking_review_prompt(
                product_analysis, usage_analysis, character_profile, setting
            )
        elif scene_type == VideoStyle.UNBOXING.value:
            video_prompt = IntelligentPromptGenerator._build_unboxing_prompt(
                product_analysis, usage_analysis, character_profile, setting
            )
        else:
            video_prompt = IntelligentPromptGenerator._build_lifestyle_prompt(
                product_analysis, usage_analysis, character_profile, setting
            )

        # Build matching image prompt
        image_prompt = IntelligentPromptGenerator._build_image_prompt(
            scene_type, product_analysis, character_profile, setting
        )

        return image_prompt, video_prompt

    @staticmethod
    def _build_silent_demo_prompt(
        product_analysis: Dict, usage_analysis: Dict, character: Dict, setting: str
    ) -> str:
        """Build detailed silent demonstration prompt with sequential actions"""

        # Get the main usage sequence
        usage_steps = usage_analysis.get("main_usage_sequence", [])
        prep_steps = usage_analysis.get("preparation_steps", [])
        temporal_markers = usage_analysis.get("temporal_markers", [])
        physical_interactions = usage_analysis.get("physical_interactions", {})

        prompt = f"""SILENT PRODUCT DEMONSTRATION - NO TALKING, NATURAL AMBIENT SOUND ONLY

CHARACTER: {character.get('description')}
SETTING: {setting}
PRODUCT: {product_analysis.get('ultra_precise_description')}

CRITICAL: This is a SILENT demonstration. NO dialogue, NO talking. Only natural sounds.

SEQUENTIAL ACTION TIMELINE (8 seconds total):

[0.0-1.0 sec] INTRODUCTION:
- Character enters frame naturally
- {prep_steps[0] if prep_steps else 'Reaches for product'}
- Product clearly visible: {product_analysis.get('product_name')}

[1.0-2.5 sec] PREPARATION:
- {prep_steps[1] if len(prep_steps) > 1 else 'Prepares to use product'}
- Hand position: {physical_interactions.get('hands', 'natural grip')}
- Product held at {usage_analysis.get('camera_friendly_angles', ['front angle'])[0]}

[2.5-6.0 sec] MAIN DEMONSTRATION:
"""

        # Add detailed usage steps with timing
        for i, step in enumerate(usage_steps[:4]):  # Focus on first 4 key steps
            start_time = 2.5 + (i * 0.875)
            end_time = start_time + 0.875
            prompt += f"\n[{start_time:.1f}-{end_time:.1f} sec]: {step}"

        prompt += f"""

[6.0-7.5 sec] COMPLETION:
- {usage_steps[-1] if usage_steps else 'Finishing movement'}
- Natural reaction to result
- Product still clearly visible

[7.5-8.0 sec] RESULT:
- Show final effect/result
- Satisfied expression
- Natural ending

VISUAL REQUIREMENTS:
- Amateur iPhone quality video
- Handheld with slight natural movement
- {setting} with realistic background
- Natural lighting, no filters
- Focus on hands and product interaction
- NO sudden transformations or unrealistic effects

AUDIO: Natural ambient sounds only - NO music, NO talking

AUTHENTICITY MARKERS:
- {usage_analysis.get('realistic_details', ['natural movements'])[0]}
- Slight camera shake
- Occasional refocus
- Real-time actions (no speed up/slow down)

CRITICAL: Every action must flow naturally into the next. NO teleporting, NO instant changes."""

        return prompt

    @staticmethod
    def _build_talking_review_prompt(
        product_analysis: Dict, usage_analysis: Dict, character: Dict, setting: str
    ) -> str:
        """Build talking review prompt with product interaction"""

        prompt = f"""TALKING PRODUCT REVIEW - AUTHENTIC UGC STYLE

CHARACTER: {character.get('description')} speaking directly to camera
SETTING: {setting}
PRODUCT: {product_analysis.get('ultra_precise_description')}

DIALOGUE & ACTION SEQUENCE:

[0.0-2.0 sec] OPENING:
- Looks at camera, slight smile
- "Okay so... I've been using this {product_analysis.get('product_name')}..."
- Holds product up naturally
- Product clearly visible in frame

[2.0-4.0 sec] DEMONSTRATION WHILE TALKING:
- "Let me show you how it works..."
- Begins {usage_analysis.get('main_usage_sequence', ['using product'])[0]}
- Natural hand gestures
- Maintains eye contact with camera occasionally

[4.0-6.0 sec] CONTINUING USE:
- "The thing I love about this is..."
- {usage_analysis.get('main_usage_sequence', ['demonstrating'])[1]}
- Genuine reaction to product working
- "...wait, look at this..."

[6.0-8.0 sec] CONCLUSION:
- Shows result while talking
- "Like... this is actually amazing"
- Natural ending with product visible
- "You guys need to try this"

VISUAL: Amateur iPhone selfie video, natural lighting, {setting}
AUDIO: Natural speech with occasional "um" or pause, ambient background noise
AUTHENTICITY: Slight camera shake, natural facial expressions, genuine reactions

NO professional presentation, NO scripted feeling, NO perfect grammar"""

        return prompt

    @staticmethod
    def _build_unboxing_prompt(
        product_analysis: Dict, usage_analysis: Dict, character: Dict, setting: str
    ) -> str:
        """Build unboxing prompt with excitement and discovery"""

        prompt = f"""UNBOXING EXPERIENCE - FIRST IMPRESSIONS

CHARACTER: {character.get('description')} excited about new product
SETTING: {setting}
PRODUCT IN BOX/PACKAGING: {product_analysis.get('ultra_precise_description')}

SEQUENTIAL UNBOXING:

[0.0-1.5 sec] ANTICIPATION:
- Shows packaged product to camera
- "You guys... it finally arrived!"
- Genuine excitement on face

[1.5-3.5 sec] OPENING:
- Carefully opens packaging
- "I've been waiting for this..."
- Shows packaging details
- Natural unboxing movements

[3.5-5.5 sec] REVEAL:
- Pulls out product slowly
- "Oh my god, look at this!"
- Examines product closely
- Shows all angles to camera

[5.5-7.0 sec] FIRST TOUCH:
- Touches/feels product texture
- "The quality is actually..."
- Natural inspection movements
- Genuine first reactions

[7.0-8.0 sec] EXCITEMENT:
- "I can't wait to try this!"
- Holds product up proudly
- Natural ending

VISUAL: iPhone video, slightly shaky, natural light from window
AUDIO: Authentic excitement, packaging sounds, genuine reactions
DETAILS: Real unboxing pace, no jump cuts, natural discovery"""

        return prompt

    @staticmethod
    def _build_lifestyle_prompt(
        product_analysis: Dict, usage_analysis: Dict, character: Dict, setting: str
    ) -> str:
        """Build lifestyle prompt showing product in daily use"""

        prompt = f"""LIFESTYLE CONTENT - PRODUCT IN DAILY ROUTINE

CHARACTER: {character.get('description')} in their natural routine
SETTING: {setting} - lived-in, authentic space
PRODUCT: {product_analysis.get('ultra_precise_description')}

NATURAL LIFESTYLE SEQUENCE:

[0.0-2.0 sec] CONTEXT:
- Character doing everyday activity
- Product naturally placed in scene
- Casual, unaware of camera initially

[2.0-4.0 sec] NATURAL REACH:
- Notices need for product
- Reaches for it naturally
- "{usage_analysis.get('preparation_steps', ['Picks up product'])[0]}"

[4.0-6.0 sec] CASUAL USE:
- Uses product as part of routine
- Natural, unhurried movements
- Focus on authenticity over demonstration

[6.0-8.0 sec] CONTINUE ROUTINE:
- Product integrated into activity
- Natural satisfaction
- Continues with daily routine

VISUAL: Candid iPhone footage, natural home lighting, authentic environment
AUDIO: Ambient home sounds, no music, natural movements
STYLE: Documentary feel, not staged, genuine daily moment"""

        return prompt

    @staticmethod
    def _build_image_prompt(
        scene_type: str, product_analysis: Dict, character: Dict, setting: str
    ) -> str:
        """Build matching image prompt for reference"""

        action_map = {
            VideoStyle.SILENT_DEMO.value: f"holding and demonstrating the {product_analysis.get('product_name')}",
            VideoStyle.TALKING_REVIEW.value: f"talking to camera while showing the {product_analysis.get('product_name')}",
            VideoStyle.UNBOXING.value: f"excitedly unboxing the {product_analysis.get('product_name')}",
            VideoStyle.LIFESTYLE.value: f"casually using the {product_analysis.get('product_name')} in daily routine",
        }

        prompt = f"""Amateur iPhone photo, Reddit quality, unposed authentic moment

SUBJECT: {character.get('description')}
ACTION: {action_map.get(scene_type, 'using product')}
PRODUCT VISIBLE: {product_analysis.get('ultra_precise_description')}
SETTING: {setting}

CRITICAL DETAILS:
- Product held naturally and clearly visible
- Natural expression, not posing
- Realistic hand position
- Amateur photography quality
- Natural shadows and lighting
- Slightly imperfect framing
- Genuine moment captured

STYLE: Snapchat story quality, authentic UGC, NOT professional photo"""

        return prompt


class UltraRealisticUGCGenerator:
    def __init__(
        self,
        gemini_api_key: str = None,
        openai_api_key: str = None,
        gcp_project_id: str = None,
        gcp_location: str = "us-central1",
    ):
        """Initialize the Ultimate UGC Generator with enhanced prompting"""

        # AI Clients
        self.gemini_client = (
            genai.Client(api_key=gemini_api_key) if gemini_api_key else genai.Client()
        )
        self.openai_client = (
            OpenAI(api_key=openai_api_key) if openai_api_key else OpenAI()
        )

        # Initialize analyzers
        self.usage_analyzer = ProductUsageAnalyzer()
        self.prompt_generator = IntelligentPromptGenerator()

        # Veo3 setup
        self.gcp_project_id = gcp_project_id
        self.gcp_location = gcp_location
        if gcp_project_id:
            aiplatform.init(project=gcp_project_id, location=gcp_location)
            self.veo_client = aiplatform.gapic.PredictionServiceClient()
            self.veo_credentials = self.veo_client._transport._credentials

        print(
            "🚀 Ultra-Realistic UGC Generator V3.0 with Intelligent Prompting Initialized"
        )

    def analyze_product_with_precision(self, image_path: str) -> Dict:
        """Advanced product analysis with extreme detail extraction"""
        print(f"🔬 Performing precision product analysis...")

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        analysis_prompt = """You are a forensic product analyst. Analyze this product with EXTREME precision.
        
        Your analysis will be used to recreate this EXACT product in AI-generated content.
        Miss NO details. Be obsessively specific.
        
        ANALYZE:
        
        1. **Product Identity**
        - Exact product name (as written on package)
        - Brand name and logo details
        - Product category and type
        
        2. **Color Science**
        - Primary color (specific shade, not generic)
        - Secondary colors
        - Gradient or color transitions
        - Finish (matte/glossy/satin/metallic)
        - Transparency/opacity
        
        3. **Physical Dimensions**
        - Shape (cylinder/square/oval/custom)
        - Approximate size (small/medium/large + comparison object)
        - Proportions (height vs width)
        
        4. **Packaging Architecture**
        - Material (glass/plastic/metal/cardboard)
        - Texture (smooth/ribbed/embossed)
        - Opening mechanism (pump/cap/dropper/spray/clamp)
        - Special features (windows/holographic/embossing)
        
        5. **Typography & Graphics**
        - All visible text (word for word)
        - Font styles (modern/classic/handwritten)
        - Text placement and hierarchy
        - Icons or symbols
        - Warning labels or certifications
        
        6. **Unique Identifiers**
        - Serial numbers/batch codes
        - Barcodes/QR codes position
        - Special edition markers
        - Limited edition details
        
        7. **Surface Details**
        - Reflections and highlights
        - Shadows and depth
        - Wear patterns or pristine condition
        - Any stickers or additional labels
        
        8. **Context Clues**
        - Target demographic (luxury/budget/professional)
        - Usage category (daily/special/professional)
        - Size category (travel/full/sample)
        
        Return as JSON with:
        - product_name
        - brand
        - category
        - primary_color
        - secondary_colors
        - shape_type
        - size_category
        - material_type
        - texture_details
        - opening_type
        - all_visible_text
        - unique_features
        - finish_type
        - ultra_precise_description (300+ words, describing EVERYTHING you see)
        - key_visual_anchors (5 most distinctive visual features)
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": analysis_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                },
                            },
                        ],
                    }
                ],
                response_format={"type": "json_object"},
            )

            # Add debugging
            print(f"Response received: {response}")

            if not response.choices:
                print("❌ No choices in response")
                raise Exception("OpenAI API returned no choices")

            if not response.choices[0].message.content:
                print("❌ No content in message")
                print(f"Full response: {response.model_dump()}")

                # Check for refusal
                if (
                    hasattr(response.choices[0].message, "refusal")
                    and response.choices[0].message.refusal
                ):
                    print(
                        f"❌ Request was refused: {response.choices[0].message.refusal}"
                    )
                    raise Exception(
                        f"OpenAI refused the request: {response.choices[0].message.refusal}"
                    )

                raise Exception("OpenAI API returned empty content")

            analysis = json.loads(response.choices[0].message.content)
            print(f"✅ Product DNA extracted: {analysis.get('product_name')}")

            # Enhance with usage analysis
            usage_analysis = self.usage_analyzer.analyze_product_usage(
                analysis, self.openai_client
            )
            analysis["usage_instructions"] = usage_analysis
            print(
                f"✅ Usage patterns analyzed: {usage_analysis.get('product_type_category')}"
            )

            return analysis

        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Content received: {response.choices[0].message.content}")
            raise
        except Exception as e:
            print(f"❌ Error during product analysis: {e}")
            raise

    def generate_intelligent_ugc_campaign(
        self, product_analysis: Dict, campaign_config: Dict
    ) -> Dict:
        """Generate intelligent UGC campaign with perfect scene orchestration"""
        print(f"🎬 Orchestrating {campaign_config.get('video_count', 3)} UGC scenes...")

        # Extract usage instructions
        usage_instructions = product_analysis.get("usage_instructions", {})

        system_prompt = f"""You are an expert UGC content strategist creating ultra-realistic content.

## PRODUCT INFORMATION
Product: {product_analysis.get('product_name')}
Category: {usage_instructions.get('product_type_category')}
Description: {product_analysis.get('ultra_precise_description')}

## USAGE INSTRUCTIONS (MUST FOLLOW)
Preparation: {json.dumps(usage_instructions.get('preparation_steps', []))}
Main Sequence: {json.dumps(usage_instructions.get('main_usage_sequence', []))}
Physical Interactions: {json.dumps(usage_instructions.get('physical_interactions', {}))}
Temporal Markers: {json.dumps(usage_instructions.get('temporal_markers', []))}
Camera Angles: {json.dumps(usage_instructions.get('camera_friendly_angles', []))}

## YOUR MISSION
Create {campaign_config.get('video_count', 3)} diverse UGC scenes.

For each scene, provide:
1. scene_id: unique identifier
2. video_type: {campaign_config.get('video_types', 'one of: silent_demonstration, talking_review, unboxing, lifestyle')}
3. character_profile: detailed character description including demographics
4. setting_description: authentic, lived-in space description
5. model: "veo3_fast"
6. aspect_ratio: "{campaign_config.get('aspect_ratio', '9:16')}"

CRITICAL: Each scene must have different characters and settings for diversity.
Demographics requirement: {campaign_config.get('demographics', 'diverse')}"""

        user_prompt = f"""Generate {campaign_config.get('video_count', 3)} UGC video scenes for this product.
        
        Requirements:
        - Video types: {campaign_config.get('video_types', 'mixed')}
        - Demographics: {campaign_config.get('demographics', 'diverse')}
        - Platform: {campaign_config.get('platform', 'TikTok')}
        
        Ensure maximum authenticity and realistic product usage."""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "ugc_campaign",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "scenes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "scene_id": {"type": "string"},
                                        "video_type": {"type": "string"},
                                        "character_profile": {"type": "string"},
                                        "setting_description": {"type": "string"},
                                        "aspect_ratio": {"type": "string"},
                                        "model": {"type": "string"},
                                    },
                                    "required": [
                                        "scene_id",
                                        "video_type",
                                        "character_profile",
                                        "setting_description",
                                        "aspect_ratio",
                                        "model",
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                            "campaign_strategy": {"type": "string"},
                        },
                        "required": ["scenes", "campaign_strategy"],
                        "additionalProperties": False,
                    },
                },
            },
        )

        campaign = json.loads(response.choices[0].message.content)

        # Generate detailed prompts for each scene
        for scene in campaign["scenes"]:
            character_details = {"description": scene["character_profile"]}

            # Generate sequential prompts using the intelligent prompt generator
            image_prompt, video_prompt = (
                self.prompt_generator.generate_sequential_prompt(
                    scene_type=scene["video_type"],
                    product_analysis=product_analysis,
                    usage_analysis=usage_instructions,
                    character_profile=character_details,
                    setting=scene["setting_description"],
                )
            )

            scene["image_prompt"] = image_prompt
            scene["video_prompt"] = video_prompt
            scene["usage_sequence"] = usage_instructions.get("main_usage_sequence", [])

        print(
            f"✅ Generated {len(campaign['scenes'])} authentic UGC scenes with sequential prompts"
        )
        return campaign

    def generate_ultra_realistic_image(
        self, prompt: str, product_description: str, save_path: str
    ) -> str:
        """Generate ultra-realistic amateur-style image with Nano Banana"""
        print(f"📸 Generating authentic amateur image...")

        response = self.gemini_client.models.generate_content(
            model="gemini-2.5-flash-image-preview", contents=[prompt]
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image = Image.open(BytesIO(part.inline_data.data))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                image.save(save_path)
                print(f"✅ Authentic image saved: {save_path}")
                return save_path

        print("❌ Image generation failed")
        return None

    def generate_ultra_realistic_video(
        self,
        prompt: str,
        aspect_ratio: str,
        model: str,
        save_dir: str,
        reference_image: str = None,
    ) -> str:
        """Generate ultra-realistic video with Veo3 using detailed sequential prompts"""
        print(f"🎥 Generating authentic amateur video...")
        print(f"📝 Prompt length: {len(prompt)} characters")

        model_id = "veo-3.0-fast-generate-preview"

        url = f"https://{self.gcp_location}-aiplatform.googleapis.com/v1/projects/{self.gcp_project_id}/locations/{self.gcp_location}/publishers/google/models/{model_id}:predictLongRunning"

        if not self.veo_credentials.valid:
            from google.auth.transport.requests import Request

            self.veo_credentials.refresh(Request())

        headers = {
            "Authorization": f"Bearer {self.veo_credentials.token}",
            "Content-Type": "application/json",
        }

        instance = {"prompt": prompt}

        # Use reference image for consistency
        if reference_image and os.path.exists(reference_image):
            print(f"🔗 Using reference image for product consistency")
            with open(reference_image, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")

            mime_type = (
                "image/png"
                if reference_image.lower().endswith(".png")
                else "image/jpeg"
            )
            instance["image"] = {
                "bytesBase64Encoded": image_base64,
                "mimeType": mime_type,
            }

        payload = {
            "instances": [instance],
            "parameters": {
                "durationSeconds": 8,
                "aspectRatio": aspect_ratio,
                "resolution": "1080p",
                "sampleCount": 1,
                "generateAudio": True,
            },
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            operation_name = response.json()["name"]
            print(f"⏳ Video generation started with sequential actions...")
            return self._poll_video_operation(operation_name, model_id, save_dir)
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return None

    def _poll_video_operation(
        self, operation_name: str, model_id: str, save_dir: str, max_wait: int = 600
    ) -> str:
        """Poll for video completion"""
        url = f"https://{self.gcp_location}-aiplatform.googleapis.com/v1/projects/{self.gcp_project_id}/locations/{self.gcp_location}/publishers/google/models/{model_id}:fetchPredictOperation"

        if not self.veo_credentials.valid:
            from google.auth.transport.requests import Request

            self.veo_credentials.refresh(Request())

        headers = {
            "Authorization": f"Bearer {self.veo_credentials.token}",
            "Content-Type": "application/json",
        }

        start_time = time.time()
        while time.time() - start_time < max_wait:
            response = requests.post(
                url, headers=headers, json={"operationName": operation_name}
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("done", False):
                    videos = result.get("response", {}).get("videos", [])
                    if videos and "bytesBase64Encoded" in videos[0]:
                        video_data = base64.b64decode(videos[0]["bytesBase64Encoded"])
                        os.makedirs(save_dir, exist_ok=True)
                        video_path = os.path.join(
                            save_dir, f"ugc_{int(time.time())}.mp4"
                        )
                        with open(video_path, "wb") as f:
                            f.write(video_data)
                        print(f"✅ Video saved: {video_path}")
                        return video_path

            time.sleep(15)

        print("❌ Video generation timeout")
        return None

    def execute_perfect_ugc_campaign(
        self,
        product_image_path: str,
        video_count: int = 3,
        video_types: Optional[List[VideoStyle]] = None,
        demographics: Optional[str] = "diverse",
        platform: str = "TikTok",
        output_dir: str = "ugc_output",
        special_requests: str = "",
    ) -> Dict:
        """
        Execute complete UGC campaign with maximum realism and proper sequential actions
        """
        print("\n" + "=" * 80)
        print("🚀 ULTRA-REALISTIC UGC CAMPAIGN GENERATION V3.0")
        print("🎯 Enhanced with Intelligent Sequential Prompting")
        print("=" * 80 + "\n")

        # Step 1: Precision Product Analysis with Usage Instructions
        print("[ PHASE 1: PRODUCT DNA & USAGE EXTRACTION ]")
        product_analysis = self.analyze_product_with_precision(product_image_path)

        # Step 2: Campaign Configuration
        campaign_config = {
            "video_count": video_count,
            "video_types": video_types or "mixed",
            "demographics": demographics,
            "platform": platform,
            "aspect_ratio": "9:16" if platform in ["TikTok", "Instagram"] else "16:9",
            "special_requests": special_requests,
        }

        # Step 3: Generate Campaign Strategy with Sequential Prompts
        print("\n[ PHASE 2: INTELLIGENT CAMPAIGN ORCHESTRATION ]")
        campaign = self.generate_intelligent_ugc_campaign(
            product_analysis, campaign_config
        )

        # Step 4: Execute Scene Generation with Enhanced Prompts
        print("\n[ PHASE 3: SEQUENTIAL CONTENT GENERATION ]")
        results = {
            "product_analysis": product_analysis,
            "usage_instructions": product_analysis.get("usage_instructions", {}),
            "campaign_strategy": campaign.get("campaign_strategy"),
            "generated_content": [],
        }

        for idx, scene in enumerate(campaign["scenes"], 1):
            print(f"\n{'='*60}")
            print(f"🎬 SCENE {idx}/{video_count}: {scene['video_type']}")
            print(f"👤 Character: {scene['character_profile']}")
            print(
                f"📍 Setting: {scene.get('setting_description', 'Natural environment')}"
            )
            if scene.get("usage_sequence"):
                print(f"🎯 Action Sequence: {len(scene['usage_sequence'])} steps")
            print(f"{'='*60}")

            scene_dir = os.path.join(output_dir, f"scene_{scene['scene_id']}")
            os.makedirs(scene_dir, exist_ok=True)

            # Save prompts for debugging
            prompts_file = os.path.join(scene_dir, "prompts.json")
            with open(prompts_file, "w") as f:
                json.dump(
                    {
                        "image_prompt": scene["image_prompt"],
                        "video_prompt": scene["video_prompt"],
                        "usage_sequence": scene.get("usage_sequence", []),
                    },
                    f,
                    indent=2,
                )

            # Generate reference image with detailed prompt
            image_path = os.path.join(scene_dir, "reference.jpg")
            generated_image = self.generate_ultra_realistic_image(
                prompt=scene["image_prompt"],
                product_description=product_analysis.get(
                    "ultra_precise_description", ""
                ),
                save_path=image_path,
            )

            # Generate video with sequential prompt
            video_path = self.generate_ultra_realistic_video(
                prompt=scene["video_prompt"],
                aspect_ratio=scene["aspect_ratio"],
                model=scene["model"],
                save_dir=scene_dir,
                reference_image=generated_image,
            )

            results["generated_content"].append(
                {
                    "scene_id": scene["scene_id"],
                    "type": scene["video_type"],
                    "character": scene["character_profile"],
                    "image": generated_image,
                    "video": video_path,
                    "prompts": {
                        "image": scene["image_prompt"][:200] + "...",
                        "video": scene["video_prompt"][:200] + "...",
                    },
                    "usage_sequence": scene.get("usage_sequence", []),
                }
            )

            print(f"✅ Scene {idx} complete with sequential actions!")

        # Save complete campaign data
        campaign_file = os.path.join(output_dir, "campaign_data.json")
        with open(campaign_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*80}")
        print(f"🎉 ULTRA-REALISTIC UGC CAMPAIGN COMPLETE!")
        print(f"📁 Output: {output_dir}")
        print(f"🎬 Videos Generated: {len(results['generated_content'])}")
        print(f"🎯 Sequential Actions: Implemented")
        print(f"{'='*80}\n")

        return results


# Simplified execution function
def create_perfect_ugc_campaign(
    product_image: str,
    videos: int = 3,
    style: str = "mixed",  # mixed, talking, silent, unboxing
    demographics: str = "diverse",  # diverse, young_female, young_male, mixed_age
    platform: str = "TikTok",  # TikTok, Instagram, YouTube
    gemini_key: str = None,
    openai_key: str = None,
    gcp_project: str = None,
    special_notes: str = "",
) -> Dict:
    """
    Quick function to create perfect UGC campaign with intelligent sequential prompting

    Args:
        product_image: Path to product image
        videos: Number of videos to generate (1-5)
        style: Video style preference
        demographics: Character demographics
        platform: Target platform
        special_notes: Any special requirements

    Returns:
        Campaign results dictionary with enhanced sequential prompts
    """

    # Map style to video types
    style_map = {
        "mixed": None,  # System decides
        "talking": [VideoStyle.TALKING_REVIEW, VideoStyle.TESTIMONIAL],
        "silent": [VideoStyle.SILENT_DEMO, VideoStyle.LIFESTYLE],
        "unboxing": [VideoStyle.UNBOXING],
        "tutorial": [VideoStyle.TUTORIAL],
    }

    generator = UltraRealisticUGCGenerator(
        gemini_api_key=gemini_key, openai_api_key=openai_key, gcp_project_id=gcp_project
    )

    # Add special instructions for better sequential generation
    enhanced_notes = f"""{special_notes}

CRITICAL REQUIREMENTS:
- Every action must flow naturally without jumps or teleporting
- Product usage must follow realistic sequential steps
- No instant transformations or morphing
- Maintain temporal consistency throughout
- Show realistic hand movements and transitions"""

    return generator.execute_perfect_ugc_campaign(
        product_image_path=product_image,
        video_count=videos,
        video_types=style_map.get(style),
        demographics=demographics,
        platform=platform,
        special_requests=enhanced_notes,
    )


# Example usage with enhanced sequential prompting
if __name__ == "__main__":
    results = create_perfect_ugc_campaign(
        product_image="test/UGC/images/lipstick.jpg",
        videos=6,
        style="mixed",
        demographics="young_female",
        platform="TikTok",
        gemini_key=os.getenv("GEMINI_API_KEY"),
        openai_key=os.getenv("OPENAI_API_KEY"),
        gcp_project=os.getenv("GCP_PROJECT_ID"),
        special_notes="Beauty product - focus on texture, color payoff, and wearability throughout the day",
    )

    print(f"\n✅ Campaign Results:")
    print(
        f"Generated {len(results['generated_content'])} videos with sequential actions"
    )

    for content in results["generated_content"]:
        print(f"\n📹 Scene: {content['type']}")
        print(f"   Character: {content['character']}")
        print(f"   Video: {content['video']}")
        print(f"   Actions: {len(content.get('usage_sequence', []))} sequential steps")

        # Print first few action steps
        if content.get("usage_sequence"):
            print("   Sample sequence:")
            for i, step in enumerate(content["usage_sequence"][:3], 1):
                print(f"     {i}. {step}")
