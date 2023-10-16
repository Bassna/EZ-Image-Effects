import os, pathlib, requests, webbrowser, random, math, re, random, datetime, sys, tempfile, uuid
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk, messagebox, colorchooser, font
from random import randint
from datetime import datetime
from PIL import (
    Image,
    ImageFilter,
    ImageOps,
    ImageEnhance,
    ImageChops,
    ImageDraw,
    ImageTk,
    ImageFont,
)
import numpy as np



FONTS = {
    "Arial": "arial.ttf",
    "Times New Roman": "times.ttf",
    "Courier New": "cour.ttf",
    "Comic Sans MS": "comic.ttf",
    "Georgia": "georgia.ttf",
    "Verdana": "verdana.ttf",
    "Tahoma": "tahoma.ttf",
    "Trebuchet MS": "trebuc.ttf",
    "Lucida Sans Unicode": "l_10646.ttf",
    "Impact": "impact.ttf",
    "Calibri": "calibri.ttf",
    "Segoe UI": "segoeui.ttf",
    "Consolas": "consola.ttf",
    "Microsoft Sans Serif": "micross.ttf",
    "Palatino Linotype": "pala.ttf",
    "MS Gothic": "msgothic.ttc",
    "Arial Black": "ariblk.ttf",
}


CURRENT_VERSION = "v0.9.9"

# Define a dictionary to store selected effects with their attributes
selected_effects = {}  # {effect_name: {"strength": value, "frames": value}}
# Define a global dictionary to store values of deselected effects
deselected_effects_values = {}
# Global variable to store the window object
text_image_window = None
# Default values for effect attributes
default_strength_value = 100
default_frames_value = 48


# Class for the main application
class ImageEffectApp:
    def __init__(self):
        self.animation_playing = (
            False  # Flag to check if an animation is currently playing
        )
        self.current_frame = 0  # Current frame in the animation sequence


# Instantiate the main application
app = ImageEffectApp()


# BaseEffect: Base class for all image effects
class BaseEffect:
    def __init__(self, name, priority=None):
        self.name = name
        self.priority = priority if priority is not None else 5

    def apply(self, image, progress, fill_color=None):
        raise NotImplementedError(
            "Effect subclasses must implement the 'apply' method."
        )

    def get_fill(self, image, fill_color=None):
        """Determine the fill color or transparency based on the image mode."""
        if image.mode == "RGBA":  # If the original image has an alpha channel
            return (0, 0, 0, 0)  # Transparent fill
        else:
            if not fill_color:
                fill_color = get_fill_color(image)
            return fill_color


# EffectRegistry: A registry to store and retrieve registered effects
class EffectRegistry:
    def __init__(self):
        self.effects = {}  # Dictionary to store registered effects

    def register(self, effect):
        # Register a new effect. Effect should be a subclass of BaseEffect.
        if not issubclass(effect, BaseEffect):
            raise ValueError("Only effects subclassing BaseEffect can be registered.")
        self.effects[effect().name] = effect

    def get_effect(self, name):
        # Retrieve an effect by its name. Returns None if not found.
        return self.effects.get(name, None)


# Specific effect classes
class GrowEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Grow Effect", priority=4)

    def apply(self, image, progress, fill_color=None):
        fill = self.get_fill(
            image, fill_color
        )  # Use the get_fill method from BaseEffect

        original = image.copy()
        img_width, img_height = original.size
        new_width = max(1, int(img_width * progress))
        new_height = max(1, int(img_height * progress))
        resized_img = original.resize((new_width, new_height))

        bg = Image.new(
            image.mode, original.size, fill
        )  # Respect the mode of the original image
        x = (img_width - new_width) // 2
        y = (img_height - new_height) // 2
        bg.paste(
            resized_img, (x, y), resized_img if resized_img.mode == "RGBA" else None
        )
        return bg


class BlurEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Blur Effect")

    def apply(self, image, progress):
        adjusted_progress = progress**0.2
        return image.filter(
            ImageFilter.GaussianBlur(radius=100 * (1 - adjusted_progress))
        )


class SharpenEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Sharpen Effect")

    def apply(self, image, progress):
        sharpened = image.filter(ImageFilter.SHARPEN)
        return Image.blend(image, sharpened, alpha=progress)


class SepiaToneEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Sepia Tone Effect")

    def apply(self, image, progress):
        sepia_image = ImageOps.colorize(image.convert("L"), "#704214", "#C0A080")

        # If the original image has an alpha channel, extract it and add it back after blending.
        if image.mode == "RGBA":
            alpha = image.split()[3]
            blended_rgb = Image.blend(image.convert("RGB"), sepia_image, alpha=progress)
            return Image.merge("RGBA", (*blended_rgb.split(), alpha))
        else:
            return Image.blend(image, sepia_image, alpha=progress)


class GrayscaleEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Grayscale Effect")

    def apply(self, image, progress):
        gray = ImageOps.grayscale(image).convert("RGB")

        # If the original image has an alpha channel, extract it and add it back after blending.
        if image.mode == "RGBA":
            alpha = image.split()[3]
            blended_rgb = Image.blend(image.convert("RGB"), gray, alpha=progress)
            return Image.merge("RGBA", (*blended_rgb.split(), alpha))
        else:
            return Image.blend(image, gray, alpha=progress)


class VignetteEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Vignette Effect")

    def apply(self, image, progress):
        original = image.copy()
        width, height = original.size
        overlay = Image.new("L", (width, height), color=255)
        draw = ImageDraw.Draw(overlay)

        max_dimen = max(width, height)
        ellip_bbox = (
            -max_dimen * (1 - progress) + width // 2,
            -max_dimen * (1 - progress) + height // 2,
            max_dimen * (1 - progress) + width // 2,
            max_dimen * (1 - progress) + height // 2,
        )

        draw.ellipse(ellip_bbox, fill=0)
        overlay = overlay.filter(ImageFilter.GaussianBlur(width / 4))

        # Use a transparent background for the composite step if the original image is in "RGBA" mode
        if original.mode == "RGBA":
            bg_color = (0, 0, 0, 0)
            img_as_np_array = Image.composite(
                original, Image.new("RGBA", (width, height), bg_color), overlay
            )
        else:
            bg_color = "black"
            img_as_np_array = Image.composite(
                original, Image.new("RGB", (width, height), bg_color), overlay
            )

        return img_as_np_array


class ColorOverlayEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Color Overlay Effect")

    def apply(self, image, progress):
        overlay_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        if image.mode == "RGBA":
            # Preserve alpha channel
            overlay_color = (*overlay_color, 255)
            overlay = Image.new("RGBA", image.size, overlay_color)
            rgb_blended = Image.blend(
                image.convert("RGB"), overlay.convert("RGB"), alpha=progress
            )
            return Image.merge("RGBA", (*rgb_blended.split(), image.split()[3]))
        else:
            overlay = Image.new("RGB", image.size, overlay_color)
            return Image.blend(image, overlay, alpha=progress)


class PixelationEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Pixelation Effect")

    def apply(self, image, progress):
        pixel_size = 2 + int(progress * 98)
        return image.resize((pixel_size, pixel_size), resample=Image.BILINEAR).resize(
            image.size, resample=Image.NEAREST
        )


class PixelationEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Pixelation Effect")

    def apply(self, image, progress):
        pixel_size = 2 + int(progress * 98)
        return image.resize((pixel_size, pixel_size), resample=Image.BILINEAR).resize(
            image.size, resample=Image.NEAREST
        )


class NegativeEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Negative Effect")

    def apply(self, image, progress):
        if image.mode == "RGBA":
            # Separate RGB and alpha channels
            rgb_image = image.convert("RGB")
            alpha = image.split()[3]

            # Invert the RGB channels
            inverted_rgb = ImageOps.invert(rgb_image)

            # Blend the inverted and original RGB images
            blended_rgb = Image.blend(rgb_image, inverted_rgb, alpha=progress)

            # Merge the blended RGB image with the original alpha channel
            return Image.merge("RGBA", (*blended_rgb.split(), alpha))
        else:
            inverted = ImageOps.invert(image)
            return Image.blend(image, inverted, alpha=progress)


class EmbossEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Emboss Effect")

    def apply(self, image, progress):
        embossed = image.filter(ImageFilter.EMBOSS)
        if image.mode == "RGBA":
            alpha = image.split()[3]
            blended_rgb = Image.blend(
                image.convert("RGB"), embossed.convert("RGB"), alpha=progress
            )
            return Image.merge("RGBA", (*blended_rgb.split(), alpha))
        return Image.blend(image, embossed, alpha=progress)


class EdgeHighlightEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Edge Highlight Effect")

    def apply(self, image, progress):
        cartooned = image.filter(ImageFilter.FIND_EDGES)
        return Image.blend(image, cartooned, alpha=progress)


class OilPaintingEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Oil Painting Effect")

    def apply(self, image, progress):
        painted = image.filter(ImageFilter.EDGE_ENHANCE_MORE).filter(
            ImageFilter.SMOOTH_MORE
        )
        return Image.blend(image, painted, alpha=progress)


class WatercolorEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Watercolor Effect")

    def apply(self, image, progress):
        watercolored = image.filter(ImageFilter.CONTOUR)

        if (
            image.mode == "RGBA"
        ):  # Check if the original image has an alpha channel (transparency)
            # Extract the alpha channel
            alpha = image.split()[3]

            # Blend the original image and the watercolored version without considering alpha
            blended_rgb = Image.blend(
                image.convert("RGB"), watercolored.convert("RGB"), alpha=progress
            )

            # Merge the blended RGB channels with the original alpha channel
            final_image = Image.merge("RGBA", (*blended_rgb.split(), alpha))

        else:
            final_image = Image.blend(image, watercolored, alpha=progress)

        return final_image


class MosaicEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Mosaic Effect")

    def apply(self, image, progress):
        adjusted_progress = progress**2.6
        max_mosaic_size = (
            min(image.width, image.height) // 2
        )  # Maximum size is half of the smallest dimension
        mosaic_size = (
            max(image.width, image.height)
            if adjusted_progress == 1
            else int(adjusted_progress * (max_mosaic_size - 4) + 4)
        )
        return image.resize((mosaic_size, mosaic_size), resample=Image.BILINEAR).resize(
            image.size, resample=Image.NEAREST
        )


class SolarizationEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Solarization Effect")

    def apply(self, image, progress):
        if image.mode == "RGBA":
            # Preserve alpha channel
            solarized_rgb = ImageOps.solarize(image.convert("RGB"))
            blended_rgb = Image.blend(
                image.convert("RGB"), solarized_rgb, alpha=progress
            )
            return Image.merge("RGBA", (*blended_rgb.split(), image.split()[3]))
        else:
            solarized = ImageOps.solarize(image)
            return Image.blend(image, solarized, alpha=progress)


class LensFlareEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Lens Flare Effect")

    def apply(self, image, progress):
        flare = Image.new(
            image.mode,
            image.size,
            (255, 255, 255, 0) if image.mode == "RGBA" else (255, 255, 255),
        )
        draw = ImageDraw.Draw(flare)
        ellipse_size = (int(image.size[0] * progress), int(image.size[1] * progress))
        draw.ellipse(
            [
                (
                    image.size[0] // 2 - ellipse_size[0] // 2,
                    image.size[1] // 2 - ellipse_size[1] // 2,
                ),
                (
                    image.size[0] // 2 + ellipse_size[0] // 2,
                    image.size[1] // 2 + ellipse_size[1] // 2,
                ),
            ],
            fill=None,
            outline=(255, 255, 255, 255) if image.mode == "RGBA" else (255, 255, 255),
            width=int(5 * progress),
        )

        if image.mode == "RGBA":
            alpha = image.split()[3]
            blended_rgb = Image.blend(
                image.convert("RGB"), flare.convert("RGB"), alpha=progress
            )
            return Image.merge("RGBA", (*blended_rgb.split(), alpha))
        else:
            return Image.blend(image, flare, alpha=progress)


class RetroEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Retro Effect")

    def apply(self, image, progress):
        retro = ImageOps.colorize(image.convert("L"), "#704214", "#FFC085")

        if image.mode == "RGBA":
            alpha = image.split()[3]
            blended_rgb = Image.blend(
                image.convert("RGB"), retro.convert("RGB"), alpha=progress
            )
            return Image.merge("RGBA", (*blended_rgb.split(), alpha))
        else:
            return Image.blend(image, retro, alpha=progress)


class RadialZoomEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Radial Zoom Effect", priority=6)

    def apply(self, image, progress):
        zoom_factor = 1 + (progress)
        zoomed = image.resize(
            (int(image.size[0] * zoom_factor), int(image.size[1] * zoom_factor))
        )
        crop_box = (
            zoomed.size[0] // 2 - image.size[0] // 2,
            zoomed.size[1] // 2 - image.size[1] // 2,
            zoomed.size[0] // 2 + image.size[0] // 2,
            zoomed.size[1] // 2 + image.size[1] // 2,
        )
        return zoomed.crop(crop_box)


class LeftToRightEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Left to Right", priority=5)

    def apply(self, image, progress, fill_color=None):
        dx = image.width - int(image.width * 2 * progress)
        fill = self.get_fill(image, fill_color)
        translated = image.transform(
            image.size, Image.AFFINE, (1, 0, dx, 0, 1, 0), fillcolor=fill
        )
        return translated


class TopToBottomEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Top to Bottom", priority=5)

    def apply(self, image, progress, fill_color=None):
        dy = image.height - int(image.height * 2 * progress)
        fill = self.get_fill(image, fill_color)
        translated = image.transform(
            image.size, Image.AFFINE, (1, 0, 0, 0, 1, dy), fillcolor=fill
        )
        return translated


class RandomJitterEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Random Jitter Effect")

    def apply(self, image, progress, fill_color=None):
        max_jitter = 10
        dx, dy = random.randint(-max_jitter, max_jitter), random.randint(
            -max_jitter, max_jitter
        )
        fill = self.get_fill(image, fill_color)
        translated = image.transform(
            image.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=fill
        )
        return translated


class WaveEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Wave Effect")

    def apply(self, image, progress, fill_color=None):
        amplitude = image.height // 10
        frequency = 2 * math.pi * progress
        offset = int(amplitude * math.sin(frequency))
        fill = self.get_fill(image, fill_color)
        output_image = Image.new(image.mode, image.size, fill)
        output_image.paste(image, (0, offset))
        return output_image


class SpinEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Spin Effect", priority=4)

    def apply(self, image, progress, fill_color=None):
        if progress <= 0.5:
            adjusted_progress = 2 * progress
            scale_factor = 1 - adjusted_progress
        else:
            adjusted_progress = 2 * (progress - 0.5)
            scale_factor = adjusted_progress
        new_width = int(image.width * scale_factor)
        new_height = image.height
        if new_width == 0:
            fill = self.get_fill(image, fill_color)
            return Image.new(image.mode, image.size, fill)
        scaled = image.resize((new_width, new_height), Image.BICUBIC)
        fill = self.get_fill(image, fill_color)
        output_image = Image.new(image.mode, image.size, fill)
        x_offset = (image.width - scaled.width) // 2
        output_image.paste(scaled, (x_offset, 0))
        return output_image


class StaticEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Static Effect")

    def apply(self, image, progress):
        amplitude = image.height // 20
        frequency = 2 * math.pi * progress
        img_array = np.array(image)
        y_indices = np.arange(image.height)
        offsets = (amplitude * np.sin(y_indices * frequency)).astype(int)
        x_indices = (np.arange(image.width) + offsets[:, None]) % image.width
        output_array = img_array[y_indices[:, None], x_indices]
        return Image.fromarray(output_array)


class RotationEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Rotation Effect", priority=4)

    def apply(self, image, progress, fill_color=None):
        fill = self.get_fill(image, fill_color)
        angle = -360 * progress
        diagonal = math.sqrt(image.width**2 + image.height**2)
        max_dim = int(diagonal)
        output_image = Image.new(image.mode, (max_dim, max_dim), fill)
        x_offset = (max_dim - image.width) // 2
        y_offset = (max_dim - image.height) // 2
        output_image.paste(image, (x_offset, y_offset))
        rotated = output_image.rotate(
            angle, center=(max_dim // 2, max_dim // 2), resample=Image.BICUBIC
        )
        left = (max_dim - image.width) // 2
        top = (max_dim - image.height) // 2
        right = left + image.width
        bottom = top + image.height
        cropped_image = rotated.crop((left, top, right, bottom))
        return cropped_image


class DistortedMirrorEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Distorted Mirror Effect", priority=4)

    def apply(self, image, progress):
        arr = np.array(image)
        y_idx, x_idx = np.indices(arr.shape[:2])
        cycles = 3
        distortion = np.sin(
            x_idx * cycles * 2 * np.pi / image.width + progress * 2 * np.pi
        ) * (image.height * 0.1)
        new_y = np.clip(y_idx + distortion.astype(int), 0, image.height - 1)
        distorted_arr = arr[new_y, x_idx]
        return Image.fromarray(distorted_arr)


class WarpEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Warp Effect")

    def apply(self, image, progress, fill_color=None):
        fill = self.get_fill(image, fill_color)
        img_width, img_height = image.size
        np_image = np.array(image)
        x, y = np.meshgrid(np.arange(img_width), np.arange(img_height))
        dx = x - img_width // 2
        dy = y - img_height // 2
        distances = np.sqrt(dx**2 + dy**2)
        max_distance = math.sqrt((img_width // 2) ** 2 + (img_height // 2) ** 2)
        adjusted_distances = distances + (max_distance - distances) * progress
        non_zero_mask = distances != 0
        scale = np.ones_like(distances)
        scale[non_zero_mask] = (
            adjusted_distances[non_zero_mask] / distances[non_zero_mask]
        )
        new_x = np.clip(img_width // 2 + dx * scale, 0, img_width - 1).astype(int)
        new_y = np.clip(img_height // 2 + dy * scale, 0, img_height - 1).astype(int)
        output_image_np = np_image[new_y, new_x]
        output_image = Image.fromarray(output_image_np)
        return output_image


class FrostedGlassEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="Frosted Glass Effect")

    def apply(self, image, progress):
        depth = int(progress * 10)
        return image.effect_spread(depth)


class NoEffect(BaseEffect):
    def __init__(self):
        super().__init__(name="No Effect")

    def apply(self, image, progress):
        return image


# Initialize the registry and register the effects
registry = EffectRegistry()
registry.register(GrowEffect)
registry.register(BlurEffect)
registry.register(DistortedMirrorEffect)
registry.register(SharpenEffect)
registry.register(SepiaToneEffect)
registry.register(GrayscaleEffect)
registry.register(VignetteEffect)
registry.register(ColorOverlayEffect)
registry.register(PixelationEffect)
registry.register(NegativeEffect)
registry.register(EmbossEffect)
registry.register(EdgeHighlightEffect)
registry.register(OilPaintingEffect)
registry.register(WatercolorEffect)
registry.register(MosaicEffect)
registry.register(SolarizationEffect)
registry.register(LensFlareEffect)
registry.register(RetroEffect)
registry.register(RadialZoomEffect)
registry.register(LeftToRightEffect)
registry.register(TopToBottomEffect)
registry.register(RandomJitterEffect)
registry.register(WaveEffect)
registry.register(NoEffect)
registry.register(StaticEffect)
registry.register(SpinEffect)
registry.register(RotationEffect)
registry.register(WarpEffect)
registry.register(FrostedGlassEffect)


def create_example_image():
    """
    Create an example image with the word "Example" written in the center.
    Returns:
        img (PIL.Image.Image): The generated example image.
    """
    # Create a white image with the size of 512x512 pixels
    img = Image.new("RGB", (512, 512), color="white")

    # Try to get the Arial font for the text, if unavailable, use default font
    try:
        font = ImageFont.truetype("arial.ttf", 120)
    except:
        font = ImageFont.load_default()

    # Draw the text "Example" in the center of the image
    d = ImageDraw.Draw(img)
    text = "Example"
    text_bbox = d.textbbox(
        (0, 0), text, font=font
    )  # Calculate the bounding box of the text
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    position = ((img.width - text_width) / 2, (img.height - text_height) / 2)
    d.text(position, text, fill="black", font=font)

    return img


def play_preview_animation(frames, canvas):
    """
    Play the given frames as an animation on the provided canvas.
    Args:
        frames (list): List of PIL.Image.Image to be animated.
        canvas (tk.Canvas): Canvas on which the animation should be played.
    """
    # If an animation is already playing, cancel it
    if app.animation_playing:
        canvas.after_cancel(canvas.animation_id)

    canvas.delete("all")  # Clear previous content from the canvas

    def update_frame(frame_num):
        """
        Display a given frame and schedule the display of the next frame.
        Args:
            frame_num (int): Index of the frame to be displayed.
        """
        # Convert the frame to PhotoImage and display it on the canvas
        photo = ImageTk.PhotoImage(frames[frame_num])
        canvas.image = photo
        canvas.create_image(
            256, 256, image=photo
        )  # Place the image at the center of the canvas

        # Schedule the next frame if the current frame is not the last
        if frame_num < len(frames) - 1:
            delay = 100
            app.animation_playing = True
            canvas.animation_id = canvas.after(delay, update_frame, frame_num + 1)
        else:
            app.animation_playing = False
            canvas.delete("all")  # Clear the canvas once the animation ends

    update_frame(0)


def generate_queue_preview_frames():
    """
    Generate preview frames for all the image effects present in the queue.
    Returns:
        all_frames (list): List of all generated frames.
    """
    all_frames = []

    for data in queue_tree.queue_data:
        if len(data) == 4:
            unique_id, effects_details, num_frames, img_path = data

            # Load and resize the image to 512x512 pixels
            img = Image.open(img_path)
            img = img.resize((512, 512), resample=Image.LANCZOS)

            # Initialize frames with the base image
            intermediate_frames = [img.copy() for _ in range(num_frames)]

            for effect, details in effects_details.items():
                effect_strength = details["strength"]
                reverse_state = details.get("reverse", False)
                invert_colors = details.get("invert", False)

                for i in range(num_frames):
                    progress = i / num_frames
                    if reverse_state:
                        progress = 1 - progress

                    img_with_effect = apply_effects(
                        {effect: effect_strength},
                        intermediate_frames[i],
                        progress,
                        invert_colors,
                        False,
                    )
                    intermediate_frames[i] = img_with_effect

            all_frames.extend(intermediate_frames)
        else:
            print(f"Warning: Incorrect data format in queue_tree.queue_data: {data}")

    return all_frames


def preview_selected_effect():
    """
    Generate a preview of the selected image effect and display it on the canvas.
    """
    # Stop any ongoing animation
    if app.animation_playing:
        preview_canvas.after_cancel(preview_canvas.animation_id)
        app.animation_playing = False

    # Check if no effects are selected
    if not selected_effects:
        tk.messagebox.showwarning(
            "No Effect Selected", "Please select an effect before previewing."
        )
        return

    # Create a base example image to apply the effects on
    example_img = create_example_image()

    # Initialize frames
    num_preview_frames = 30
    frames = []

    for i in range(num_preview_frames):
        progress = i / (num_preview_frames - 1)
        frame_img = example_img.copy()

        for selected_effect_item_id, details in selected_effects.items():
            # Extract the effect name from the effects_tree using the stored item ID
            selected_effect_name = (
                effects_tree.item(selected_effect_item_id)["text"]
                .replace("✔ ", "")
                .replace("☐ ", "")
            )
            effect_strength = details["strength"]

            # Initialize states for "Reverse" and "Invert Colors"
            reverse_effect = False
            invert_colors = False

            # Check children of the tree item to determine if "Reverse" or "Invert Colors" is selected
            children = effects_tree.get_children(selected_effect_item_id)
            if not reverse_effect:
                reverse_effect = any(
                    "✔ Reverse" == effects_tree.item(child)["text"]
                    for child in children
                )
            if not invert_colors:
                invert_colors = any(
                    "✔ Invert Colors" == effects_tree.item(child)["text"]
                    for child in children
                )

            # Process the effect and apply it to the frame
            if selected_effect_name not in ["Reverse", "Invert Colors"]:
                frame_img = apply_effects(
                    {selected_effect_name: effect_strength},
                    frame_img,
                    progress,
                    invert_colors,
                    reverse_effect,
                )

        frames.append(frame_img)

    # Display the generated frames as an animation
    play_preview_animation(frames, preview_canvas)


def get_fill_color(image):
    """
    Determine the fill color by averaging the colors of the corners of the image.
    Returns:
        tuple: Either white (255, 255, 255) or black (0, 0, 0), depending on which is closer to the average color.
    """
    corner_pixels = [
        image.getpixel((0, 0)),
        image.getpixel((image.width - 1, 0)),
        image.getpixel((0, image.height - 1)),
        image.getpixel((image.width - 1, image.height - 1)),
    ]

    avg_red = sum(px[0] for px in corner_pixels) / 4
    avg_green = sum(px[1] for px in corner_pixels) / 4
    avg_blue = sum(px[2] for px in corner_pixels) / 4

    avg_color = (int(avg_red), int(avg_green), int(avg_blue))

    # Determine the distance of the average color to white and black
    distance_to_white = sum(
        [(c1 - c2) ** 2 for c1, c2 in zip(avg_color, (255, 255, 255))]
    )
    distance_to_black = sum([(c1 - c2) ** 2 for c1, c2 in zip(avg_color, (0, 0, 0))])

    return (255, 255, 255) if distance_to_white < distance_to_black else (0, 0, 0)


def blend_with_strength(original, transformed, strength):
    """
    Blend the original image with the transformed image based on the given strength.
    Returns:
        PIL.Image.Image: Blended image.
    """
    alpha = strength / 100.0
    transformed = transformed.resize(original.size, Image.LANCZOS)

    if original.mode == "RGBA" or transformed.mode == "RGBA":
        original_rgb = original.convert("RGB")
        transformed_rgb = transformed.convert("RGB")
        blended_rgb = Image.blend(original_rgb, transformed_rgb, alpha)
        if original.mode == "RGBA":
            alpha_channel = original.split()[3]
            blended_image = Image.merge("RGBA", (*blended_rgb.split(), alpha_channel))
        else:
            blended_image = blended_rgb.convert("RGBA")
    else:
        blended_image = Image.blend(original, transformed, alpha)

    return blended_image


def generate_unique_id():
    return str(uuid.uuid4())


def apply_effect(effect_name, image, progress, effect_strength, invert_colors):
    """
    Apply the specified effect to the image.
    Returns:
        PIL.Image.Image: Image with the applied effect.
    """
    effect_name = effect_name.replace("✔ ", "").replace("☐ ", "")
    effect_class = registry.get_effect(effect_name)

    # If the effect_name is not found in the registry, return the original image
    if not effect_class:
        print(f"Effect '{effect_name}' not found in registry.")
        return image

    # Create an instance of the effect
    effect_instance = effect_class()

    # Adjust the progress based on effect strength
    if effect_strength > 0:
        adjusted_progress = (1 - effect_strength / 100.0) + progress * (
            effect_strength / 100.0
        )
    elif effect_strength < 0:
        adjusted_progress = (-effect_strength / 100.0) - progress * (
            -effect_strength / 100.0
        )
    else:
        return image  # If strength is 0, return the original image

    # Apply the effect
    transformed_image = effect_instance.apply(image, adjusted_progress)

    # Use the transformed image directly
    blended_image = transformed_image

    # Invert the colors if required
    if invert_colors:
        if blended_image.mode == "RGBA":
            r, g, b, a = blended_image.split()
            inverted_image = ImageOps.invert(Image.merge("RGB", (r, g, b)))
            blended_image = Image.merge("RGBA", (inverted_image.split() + (a,)))
        else:
            blended_image = ImageOps.invert(blended_image)

    return blended_image


def apply_effects(effects_and_strengths, image, progress, invert_colors, reverse=False):
    """
    Apply multiple effects to the image in the order of their priority.
    Returns:
        PIL.Image.Image: Image with the applied effects.
    """
    # Separate effects into those with and without priorities
    effects_with_priority = [
        e
        for e in effects_and_strengths.keys()
        if registry.get_effect(e)().priority is not None
    ]
    effects_without_priority = [
        e
        for e in effects_and_strengths.keys()
        if registry.get_effect(e)().priority is None
    ]

    # Sort effects with priority
    sorted_effects_with_priority = sorted(
        effects_with_priority, key=lambda e: registry.get_effect(e)().priority
    )

    # Combine the lists
    sorted_effects = sorted_effects_with_priority + effects_without_priority

    # Adjust the progress based on reverse flag
    if reverse:
        progress = 1 - progress

    # Apply the effects in the sorted order
    for effect in sorted_effects:
        image = apply_effect(
            effect,
            image,
            progress,
            effects_and_strengths[effect],
            invert_colors if effect == sorted_effects[-1] else False,
        )

    return image


def add_to_queue():
    selected_effect_names = [
        effects_tree.item(item_id)["text"].replace("✔ ", "").replace("☐ ", "")
        for item_id in selected_effects
    ]
    if not selected_effect_names:
        messagebox.showwarning(
            "No Effect Selected", "Please select an effect before adding to the queue."
        )
        return

    file_path = filedialog.askopenfilename(
        initialdir=app.last_image_directory,
        title=f"Select an Image for {', '.join(selected_effect_names)}",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif")],
    )
    if file_path:
        app.last_image_directory = os.path.dirname(file_path)
    else:
        return

    # Create a dictionary mapping effect names to their individual details
    effects_details = {}
    for effect_item_id in selected_effects:
        effect_name = (
            effects_tree.item(effect_item_id)["text"]
            .replace("✔ ", "")
            .replace("☐ ", "")
        )
        strength = selected_effects[effect_item_id]["strength"]
        frames = selected_effects[effect_item_id]["frames"]

        children = effects_tree.get_children(effect_item_id)
        reverse_effect = any(
            "✔ Reverse" == effects_tree.item(child)["text"] for child in children
        )
        invert_colors = any(
            "✔ Invert Colors" == effects_tree.item(child)["text"] for child in children
        )

        effects_details[effect_name] = {
            "strength": strength,
            "frames": frames,
            "reverse": reverse_effect,
            "invert": invert_colors,
        }

    # Determine the maximum number of frames among the selected effects
    max_frames = max(details["frames"] for details in effects_details.values())

    # Update the frame count of each effect to max_frames
    for details in effects_details.values():
        details["frames"] = max_frames

    # Store the data for each combined effect
    unique_id = generate_unique_id()
    combined_effect_entry = (unique_id, effects_details, max_frames, file_path)

    if not hasattr(queue_tree, "queue_data"):
        queue_tree.queue_data = []
    queue_tree.queue_data.append(combined_effect_entry)

    combined_effect_names = ", ".join(selected_effect_names)
    combined_effect_parent = queue_tree.insert(
        "", "end", text=combined_effect_names, values=(unique_id, "", "")
    )

    for effect_name, details in effects_details.items():
        effect_child = queue_tree.insert(
            combined_effect_parent, "end", text=effect_name, values=("",)
        )

        queue_tree.insert(
            effect_child,
            "end",
            text=f"Effect Strength: {details['strength']}%",
            tags=("non_selectable",),
        )
        queue_tree.insert(
            effect_child,
            "end",
            text=f"Number of Frames: {max_frames}",
            tags=("non_selectable",),
        )

        if details["reverse"]:
            queue_tree.insert(
                effect_child, "end", text="✔ Reverse", tags=("non_selectable",)
            )
        else:
            queue_tree.insert(
                effect_child, "end", text="☐ Reverse", tags=("non_selectable",)
            )

        if details["invert"]:
            queue_tree.insert(
                effect_child, "end", text="✔ Invert Colors", tags=("non_selectable",)
            )
        else:
            queue_tree.insert(
                effect_child, "end", text="☐ Invert Colors", tags=("non_selectable",)
            )

    effects_tree.selection_set([])


def move_item(direction):
    selected_items = queue_tree.selection()
    if not selected_items:
        return

    selected_item = selected_items[0]
    index = queue_tree.index(selected_item)

    if direction == "up" and index > 0:
        queue_tree.queue_data[index], queue_tree.queue_data[index - 1] = (
            queue_tree.queue_data[index - 1],
            queue_tree.queue_data[index],
        )
        queue_tree.move(selected_item, "", index - 1)

    elif direction == "down" and index < len(queue_tree.get_children()) - 1:
        queue_tree.queue_data[index], queue_tree.queue_data[index + 1] = (
            queue_tree.queue_data[index + 1],
            queue_tree.queue_data[index],
        )
        queue_tree.move(selected_item, "", index + 1)


def get_index_of_selected_item(tree, selected_id):
    for parent in tree.get_children():
        if parent == selected_id:
            return tree.get_children().index(parent)
        children = tree.get_children(parent)
        if selected_id in children:
            return children.index(selected_id)
    return None


def remove_selected_effect():
    # Get the ID of the currently selected item
    selected_items = queue_tree.selection()

    if selected_items:
        # Get the index of the selected item
        selected_index = get_index_of_selected_item(queue_tree, selected_items[0])

        if selected_index is not None:
            # Delete the selected effect from the treeview
            queue_tree.delete(selected_items[0])

            # Remove the associated data from queue_data
            del queue_tree.queue_data[selected_index]


def process_queued_effects():
    # Check if the queue is empty
    if not queue_tree.queue_data:
        messagebox.showwarning(
            "Warning", "Please add effects to the queue before processing."
        )
        return
    app.current_frame = 0  # Reset current frame to 0
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    base_output_path = filedialog.askdirectory(
        initialdir=app.last_output_directory, title="Select Output Folder"
    )
    root.destroy()
    if base_output_path:
        app.last_output_directory = base_output_path
    else:
        print("Output folder not selected. Exiting.")
        return

    now = datetime.now()
    timestamped_folder_name = (
        f"animation_{now.strftime('%m-%d-%Y')}_{now.strftime('%H-%M-%S')}"
    )
    output_path = Path(base_output_path) / timestamped_folder_name

    if not output_path.exists():
        try:
            os.makedirs(output_path)
        except Exception as e:
            print(f"Error creating directory: {e}")
            return
    #print(f"Debug (process_queued_effects): queue_tree.queue_data: {queue_tree.queue_data}")

    # Handle both 3 and 4 tuple formats
    total_frames = sum(
        data[2] if len(data) == 4 else data[1] for data in queue_tree.queue_data
    )

    popup, progress_var = show_progress_bar(total_frames)
    popup.update_idletasks()

    for data in queue_tree.queue_data:
        # Handle both 3 and 4 tuple formats
        if len(data) == 4:
            _, effects_details, frames, img_path = data
        else:
            effects_details, frames, img_path = data

        # Print the effect details before processing
        #print(f"Debug (process_queued_effects): Processing effect {effects_details} with {frames} frames")

        img = Image.open(img_path)

        # Determine the maximum frames among the combined effects
        max_frames = max(
            effect_settings.get("frames", frames)
            for effect_settings in effects_details.values()
        )

        # Prepare a dictionary for effects with their strengths, reverse and invert states
        effects_and_strengths = {
            effect_name: effect_settings.get("strength", 100)
            for effect_name, effect_settings in effects_details.items()
        }
        reverse_states = {
            effect_name: effect_settings.get("reverse", False)
            for effect_name, effect_settings in effects_details.items()
        }
        invert_states = {
            effect_name: effect_settings.get("invert", False)
            for effect_name, effect_settings in effects_details.items()
        }

        # Apply each effect over the entire duration of the combined animation
        for i in range(max_frames):
            progress = i / max_frames
            img_with_effects = img.copy()
            for effect_name, strength in effects_and_strengths.items():
                reverse_state = reverse_states.get(effect_name, False)
                invert_colors = invert_states.get(effect_name, False)
                img_with_effects = apply_effects(
                    {effect_name: strength},
                    img_with_effects,
                    progress,
                    invert_colors,
                    reverse_state,
                )
            frame_path = output_path / f"frame_{app.current_frame + i + 1:05d}.png"
            img_with_effects.save(frame_path)
            if progress_var:  # Update the progress bar if it's provided
                progress_var.set(app.current_frame + i + 1)

        app.current_frame += max_frames

    # Close the progress bar pop-up
    popup.destroy()


def create_effect_sequence(
    effects_and_strengths,
    image,
    num_frames,
    output_folder,
    reverse_state,
    invert_colors,
    progress_var=None,
):
    script_dir = Path(__file__).parent
    output_path = script_dir / output_folder

    # Check if output directory exists, if not create it
    if not Path(output_path):
        output_path.mkdir(parents=True, exist_ok=True)

    generated_frames = []
    for i in range(1, num_frames + 1):
        progress = i / num_frames
        img_with_effects = apply_effects(
            effects_and_strengths, image.copy(), progress, invert_colors
        )
        generated_frames.append(img_with_effects)

    # If reverse_state is True, reverse the order of frames for this effect only
    if reverse_state:
        generated_frames = generated_frames[::-1]

    for i, img_with_effects in enumerate(generated_frames, 1):
        frame_path = output_path / f"frame_{app.current_frame + i:05d}.png"
        img_with_effects.save(frame_path)
        print(f"Saved frame {app.current_frame + i} to {frame_path}")
        if progress_var:  # update the progress bar if it's provided
            progress_var.set(app.current_frame + i)

    app.current_frame += num_frames


def clear_queue():
    queue_tree.queue_data.clear()

    for item in queue_tree.get_children():
        queue_tree.delete(item)

    app.current_frame = 0


class ToggleSwitch(tk.Frame):
    def __init__(
        self, parent, text="", alt_text=None, variable=None, default_on=False, **kwargs
    ):
        super().__init__(parent, **kwargs)

        self.toggle_frame = tk.Frame(self, width=40, height=20, bd=1, relief="solid")
        self.toggle_frame.grid(row=0, column=0)

        self.status = default_on

        if self.status:
            self.toggle = tk.Frame(self.toggle_frame, bg="green", width=20, height=20)
            self.toggle.place(x=20, y=0)
        else:
            self.toggle = tk.Frame(self.toggle_frame, bg="red", width=20, height=20)
            self.toggle.place(x=0, y=0)

        # Bind the click event to both the toggle_frame and the toggle
        self.toggle_frame.bind("<Button-1>", self.switch)
        self.toggle.bind("<Button-1>", self.switch)

        self.label = tk.Label(self, text=text)
        self.label.grid(row=0, column=1, padx=(10, 0))

        # Store the variable to be updated
        self.variable = variable

        # Store the alternate text
        self.text = text
        self.alt_text = alt_text if alt_text else text

        # Update the label text based on the initial state
        self.update_label_text()

    def switch(self, event=None):
        if self.status:
            self.toggle.configure(bg="red")
            self.toggle.place(x=0, y=0)
            self.status = False
            if self.variable:
                self.variable.set(False)

                # If this is the theme toggle
                if self.variable == theme_var:
                    apply_theme("light")  # Apply light theme
        else:
            self.toggle.configure(bg="green")
            self.toggle.place(x=20, y=0)
            self.status = True
            if self.variable:
                self.variable.set(True)

                # If this is the theme toggle
                if self.variable == theme_var:
                    apply_theme("dark")  # Apply dark theme

        # Update the label text after toggling
        self.update_label_text()

    def update_label_text(self):
        """Updates the label text based on the current status."""
        if self.status:
            self.label.config(text=self.alt_text)
        else:
            self.label.config(text=self.text)


class DragDropTree(ttk.Treeview):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)

        # Add right-click context menu
        self.context_menu = tk.Menu(self, tearoff=0)
        self.context_menu.add_command(label="Move Up", command=self.move_up)
        self.context_menu.add_command(label="Move Down", command=self.move_down)

        self.bind("<Button-3>", self.show_context_menu)  # Bind right-click
        self.bind("<Button-1>", self.set_selection)  # Bind left-click

    def show_context_menu(self, event):
        """Show the context menu on right-click."""
        item = self.identify_row(event.y)
        if item and not self.parent(
            item
        ):  # Check if the item exists and is a top-level item
            self.selection_set(item)
            self.context_menu.post(event.x_root, event.y_root)
        else:
            self.selection_remove(self.selection())

    def set_selection(self, event):
        """Set the selection on left-click."""
        item = self.identify_row(event.y)
        if not self.parent(item):  # Check if the item is a top-level item
            self.selection_set(item)
        else:
            self.selection_remove(self.selection())

    def move_up(self):
        move_item("up")

    def move_down(self):
        move_item("down")


def get_all_children(widget):
    """Recursively get all child widgets."""
    children = widget.winfo_children()
    for child in children:
        children += get_all_children(child)
    return children


def show_progress_bar(max_val):
    """
    Create a pop-up window with a progress bar.
    Returns the pop-up window and the progress bar widget.
    """
    popup = tk.Toplevel(window)
    popup.title("Processing...")

    # Center the pop-up window
    window_width = window.winfo_width()
    window_height = window.winfo_height()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x_cordinate = int((screen_width / 2) - (window_width / 2))
    y_cordinate = int((screen_height / 2) - (window_height / 2))
    popup.geometry(f"300x50+{x_cordinate}+{y_cordinate}")

    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(popup, variable=progress_var, maximum=max_val)
    progress_bar.pack(pady=15, padx=20, fill="x")

    return popup, progress_var


def deselect_categories(event=None):
    for item in effects_tree.selection():
        # If the item has no parent, it's a category. But if the item is "No Effect", we skip the deselection.
        if (
            effects_tree.parent(item) == ""
            and effects_tree.item(item)["text"] != "No Effect"
        ):
            effects_tree.selection_remove(item)


def handle_selection(event):
    global selected_effects

    tree = event.widget
    item = tree.identify_row(event.y)
    col = tree.identify_column(event.x)
    element = tree.identify_element(event.x, event.y)
    parent = tree.parent(item)
    text = tree.item(item)["text"]
    tree.update_idletasks()
    bbox = tree.bbox(item)

    if element == "button":
        return

    if "Effect Strength" in text:
        adjust_effect_strength(item, tree)
    elif "Number of Frames" in text:
        adjust_num_frames(item, tree)

    if not parent and text != "No Effect":
        return

    bbox = tree.bbox(item)
    if not bbox:
        return
    x, y, width, height = bbox

    if text in ["☐ Reverse", "✔ Reverse", "☐ Invert Colors", "✔ Invert Colors"]:
        if "✔" in text:
            new_text = text.replace("✔", "☐")
        else:
            new_text = text.replace("☐", "✔")
        tree.item(item, text=new_text)
        return "break"

    if col == "#0" and parent and text.startswith(("☐", "✔")):
        if not (x + 40 <= event.x <= x + 50):
            return

    if text.startswith("☐"):
        tree.item(item, text=text.replace("☐", "✔"))
        effect_name = text.replace("✔ ", "").replace("☐ ", "")
        effect_class = registry.get_effect(effect_name)

        if effect_class:
            effect_priority = effect_class().priority

            ordered_effects = sorted(
                selected_effects.items(),
                key=lambda x: registry.get_effect(
                    tree.item(x[0])["text"].replace("✔ ", "").replace("☐ ", "")
                )().priority,
            )
            selected_effects = {}
            inserted = False
            for id, values in ordered_effects:
                current_effect_name = (
                    tree.item(id)["text"].replace("✔ ", "").replace("☐ ", "")
                )
                if (
                    not inserted
                    and effect_priority
                    < registry.get_effect(current_effect_name)().priority
                ):
                    selected_effects[item] = {
                        "strength": default_strength_value,
                        "frames": default_frames_value,
                    }
                    inserted = True
                selected_effects[id] = values

            if not inserted:
                selected_effects[item] = {
                    "strength": default_strength_value,
                    "frames": default_frames_value,
                }

    else:
        tree.item(item, text=text.replace("✔", "☐"))
        if tree == effects_tree and item in selected_effects:
            del selected_effects[item]

    if tree == effects_tree:
        #print(f"Debug (handle_selection): Selected effects list after operation: {selected_effects}")
        return "break"


def handle_queue_selection(event):
    item = queue_tree.identify_row(event.y)
    text = queue_tree.item(item)["text"]

    # If the clicked item is "Effect Strength" or "Number of Frames"
    if "Effect Strength" in text:
        adjust_effect_strength(item, queue_tree)
    elif "Number of Frames" in text:
        adjust_num_frames(item, queue_tree)
    elif "Reverse" in text:
        if "✔" in text:
            queue_tree.item(item, text="☐ Reverse")
        else:
            queue_tree.item(item, text="✔ Reverse")

        # After making adjustments, let's print which effect was adjusted and its new values
        effect_id = queue_tree.parent(item)
        combined_effect_id = queue_tree.parent(effect_id)
        effect_unique_id = queue_tree.item(combined_effect_id)["values"][0]

        # Debug: Print the effect ID and unique ID
        effect_name = queue_tree.item(effect_id)["text"]
        #print(f"Debug (handle_queue_selection): effect_id: {effect_id}, effect_name: {effect_name}, unique_id: {effect_unique_id}")

        for idx, data in enumerate(queue_tree_data):
            uid, effects, frames, path = data
            if uid == effect_unique_id and effect_name in effects:
                print(
                    f"Debug (handle_queue_selection): Adjusted effect '{effect_name}' with details: {effects[effect_name]}"
                )
                reverse = effects[effect_name]["reverse"]
                effects[effect_name]["reverse"] = not reverse

    elif "Invert Colors" in text:
        if "✔" in text:
            queue_tree.item(item, text="☐ Invert Colors")
        else:
            queue_tree.item(item, text="✔ Invert Colors")

        effect_id = queue_tree.parent(item)
        combined_effect_id = queue_tree.parent(effect_id)
        effect_unique_id = queue_tree.item(combined_effect_id)["values"][0]
        effect_name = queue_tree.item(effect_id)["text"]

        for idx, data in enumerate(queue_tree_data):
            uid, effects, frames, path = data
            if uid == effect_unique_id and effect_name in effects:
                invert = effects[effect_name]["invert"]
                effects[effect_name]["invert"] = not invert

    # Print the updated queue_tree_data after any changes
    # print(f"Debug (handle_queue_selection): Updated queue_tree_data: {queue_tree_data}")


def adjust_effect_strength(item, tree):
    # Create a new top-level window for adjusting effect strength
    strength_window = tk.Toplevel(window)
    strength_window.title("Adjust Effect Strength")

    # Set theme and positioning
    current_theme = "dark" if window.cget("bg") == themes["dark"]["bg"] else "light"
    strength_window.configure(bg=themes[current_theme]["bg"])
    apply_positioning(strength_window, item, tree)

    # Entry for manual input
    strength_entry_label = tk.Label(
        strength_window,
        text="Enter Strength (%):",
        bg=themes[current_theme]["bg"],
        fg=themes[current_theme]["fg"],
    )
    strength_entry_label.pack(pady=(10, 0))
    strength_entry = tk.Entry(
        strength_window,
        textvariable=effect_strength_var,
        width=4,
        bg=themes[current_theme]["bg"],
        fg=themes[current_theme]["fg"],
    )
    strength_entry.pack(pady=5)

    # Slider
    local_strength_slider = tk.Scale(
        strength_window,
        from_=-100,
        to=100,
        orient=tk.HORIZONTAL,
        label="Effect Strength (%):",
        variable=effect_strength_var,
        length=300,
    )
    local_strength_slider.pack(pady=10)
    apply_theme_to_slider(local_strength_slider, themes[current_theme])

    # Button to confirm changes
    ttk.Button(
        strength_window,
        text="Confirm",
        command=lambda: update_item_text(
            strength_window,
            item,
            f"Effect Strength: {effect_strength_var.get()}%",
            tree,
        ),
    ).pack(pady=10)


def adjust_num_frames(item, tree):
    # Create a new top-level window for adjusting number of frames
    frames_window = tk.Toplevel(window)
    frames_window.title("Adjust Number of Frames")

    # Set theme and positioning
    current_theme = "dark" if window.cget("bg") == themes["dark"]["bg"] else "light"
    frames_window.configure(bg=themes[current_theme]["bg"])
    apply_positioning(frames_window, item, tree)

    # Entry for manual input
    frames_entry_label = tk.Label(
        frames_window,
        text="Enter Frames:",
        bg=themes[current_theme]["bg"],
        fg=themes[current_theme]["fg"],
    )
    frames_entry_label.pack(pady=(10, 0))
    frames_entry = tk.Entry(
        frames_window,
        textvariable=num_frames_var,
        width=4,
        bg=themes[current_theme]["bg"],
        fg=themes[current_theme]["fg"],
    )
    frames_entry.pack(pady=5)

    # Slider
    local_frames_slider = tk.Scale(
        frames_window,
        from_=1,
        to=999,
        orient=tk.HORIZONTAL,
        label="Number of Frames:",
        variable=num_frames_var,
        length=300,
    )
    local_frames_slider.pack(pady=10)
    apply_theme_to_slider(local_frames_slider, themes[current_theme])

    # Button to confirm changes
    ttk.Button(
        frames_window,
        text="Confirm",
        command=lambda: update_item_text(
            frames_window, item, f"Number of Frames: {num_frames_var.get()}", tree
        ),
    ).pack(pady=10)


def apply_positioning(settings_window, item, tree):
    # Get the bounding box of the clicked effect
    bbox = tree.bbox(item)

    # Check if bounding box coordinates are valid
    if bbox and len(bbox) == 4:
        x, y, _, _ = bbox
        abs_x = tree.winfo_rootx() + x
        abs_y = (
            tree.winfo_rooty() + y - settings_window.winfo_height() - 50
        )  # Subtract an additional 50 pixels as offset
        settings_window.geometry(f"+{abs_x}+{abs_y}")
    else:
        print(f"Warning: Could not get bounding box for item {item}.")
        # Default position (modify as needed)
        settings_window.geometry(f"+100+100")


def update_item_text(window, item, new_text, tree):
    tree.item(item, text=new_text)

    if tree == effects_tree:
        parent = tree.parent(item)
        effect_id = parent  # Use the parent directly as the effect ID

        # If the effect is not in the dictionary, add it with default values
        if effect_id not in selected_effects:
            selected_effects[effect_id] = {
                "strength": default_strength_value,
                "frames": default_frames_value,
            }

        # Check if it's a strength update or frame count update
        if "Effect Strength:" in new_text:
            selected_effects[effect_id]["strength"] = effect_strength_var.get()
        elif "Number of Frames:" in new_text:
            selected_effects[effect_id]["frames"] = num_frames_var.get()

    elif tree == queue_tree:
        effect_id = tree.parent(item)
        combined_effect_id = tree.parent(effect_id)
        effect_name = tree.item(effect_id)["text"]

        # If it's a combined effect, update only the specific effect's frame count
        if "Number of Frames:" in new_text:
            new_frames_value = int(new_text.split(":")[-1].strip())

            # Fetch the UUID of the effect being edited.
            effect_unique_id = tree.item(combined_effect_id)["values"][0]

            # Debug
            #print(f"Debug (update_item_text): Before update - queue_tree.queue_data: {queue_tree.queue_data}")

            for idx, data in enumerate(queue_tree.queue_data):
                uid, effects, frames, path = data
                if uid == effect_unique_id:
                    # Update the frame counts of all effects within the combined effect
                    for effect in effects:
                        effects[effect]["frames"] = new_frames_value

                    # Update the tree display for the "Number of Frames" child of each effect within the combined effect
                    for child_effect_id in tree.get_children(combined_effect_id):
                        for sub_child_id in tree.get_children(child_effect_id):
                            if "Number of Frames:" in tree.item(sub_child_id)["text"]:
                                tree.item(
                                    sub_child_id,
                                    text=f"Number of Frames: {new_frames_value}",
                                )
                                break

                    # Update the total frames for the combined effect
                    frames = new_frames_value
                    queue_tree.queue_data[idx] = (uid, effects, frames, path)
                    break

            # Debug
            #print(f"Debug (update_item_text): After update - queue_tree.queue_data: {queue_tree.queue_data}")

        # If it's effect strength, update the strength in queue_tree_data
        elif "Effect Strength:" in new_text:
            new_strength_value = int(new_text.split(":")[1].replace("%", "").strip())

            # Fetch the UUID of the effect being edited.
            effect_unique_id = tree.item(combined_effect_id)["values"][0]

            for idx, data in enumerate(queue_tree.queue_data):
                uid, effects, frames, path = data
                if (
                    uid == effect_unique_id and effect_name in effects
                ):  # Ensure both UUID and effect name match
                    effects[effect_name]["strength"] = new_strength_value
                    queue_tree.queue_data[idx] = (uid, effects, frames, path)
                    break

    window.destroy()


def apply_theme_to_slider(slider, theme):
    slider.configure(
        bg=theme["button_bg"],
        fg=theme["fg"],
        troughcolor=theme.get("listbox_bg", "#616161"),
        highlightbackground=theme.get("slider_border", "#616161"),
        highlightcolor=theme.get("slider_border", "#616161"),
        sliderrelief="raised",
        sliderlength=20,
        activebackground=theme.get("slider_handle", "#505050"),
    )


def play_queue_preview():
    frames = generate_queue_preview_frames()

    # Check if the frames list is empty
    if not frames:
        # Show a message to the user
        tk.messagebox.showinfo(
            "Queue Empty",
            "The queue is empty. Please add effects to the queue before previewing.",
        )
        return

    play_preview_animation(frames, preview_canvas)


def apply_theme_to_widgets(root_widget, theme):
    all_widgets = get_all_children(root_widget)
    for widget in all_widgets:
        if isinstance(widget, tk.Label):
            widget.configure(bg=theme["bg"], fg=theme["fg"])
        elif isinstance(widget, tk.Scale):
            widget.configure(
                bg=theme["button_bg"],
                fg=theme["fg"],
                troughcolor=theme.get("listbox_bg", "#616161"),
                highlightbackground=theme.get("slider_border", "#616161"),
                highlightcolor=theme.get("slider_border", "#616161"),
                sliderrelief="raised",
                sliderlength=20,
                activebackground=theme.get(
                    "slider_handle", widget.cget("activebackground")
                ),
            )


def apply_theme(theme_name):
    theme = themes[theme_name]
    style = ttk.Style()

    # Configure the main window and frames
    window.configure(bg=theme["bg"])
    left_frame.configure(bg=theme["bg"])
    preview_frame.configure(bg=theme["bg"])
    right_frame.configure(bg=theme["bg"])
    checkbox_frame.configure(bg=theme["bg"])
    toggle_frame.configure(bg=theme["bg"])
    preview_canvas.configure(bg=theme["button_bg"])

    # Update the custom menu bar's background
    create_custom_menu_bar(theme_name)
    menu_bar_frame = create_custom_menu_bar(
        theme_name
    )  # Capture the returned frame here
    menu_bar_frame.configure(bg=theme["bg"])

    # Configure theme for the Treeview
    style.theme_use(
        "clam"
    )  # Use the "clam" theme for full customization of the Treeview widget
    style.configure(
        "Custom.Treeview",
        background=theme["tree_bg"],
        foreground=theme["tree_fg"],
        fieldbackground=theme["tree_field_bg"],
        borderwidth=0,
        highlightthickness=0,
        bd=0,  # Remove borders
        rowheight=20,
        padding=0,  # Adjust row height and remove padding
    )
    selected_foreground = "#000000" if theme_name == "light" else "#ffffff"
    style.map(
        "Custom.Treeview",
        background=[("selected", theme["tree_selected_bg"])],
        foreground=[
            ("selected", selected_foreground)
        ],  # Setting the foreground color based on the theme
    )

    # Configure the Treeview heading
    style.configure(
        "Custom.Treeview.Heading",
        background=theme["tree_bg"],
        foreground=theme["tree_fg"],
        borderwidth=0,
        highlightthickness=0,
    )

    # Configure the Treeview's parent frame
    style.configure("Custom.TFrame", background=theme["tree_bg"])

    # Set the style for the treeview
    style.configure(
        "Treeview",
        background=theme["listbox_bg"],
        foreground=theme["fg"],
        fieldbackground=theme["listbox_bg"],
    )
    style.map("Treeview", background=[("selected", theme["selected_bg"])])

    # Adjust ttk.Button style
    style.configure("TButton", background=theme["button_bg"], foreground=theme["fg"])

    all_widgets = get_all_children(window)
    for widget in all_widgets:
        if isinstance(widget, tk.Label):
            widget.configure(bg=theme["bg"], fg=theme["fg"])


def create_custom_menu_bar(theme_name):
    theme = themes[theme_name]

    # Create a frame to represent the menu bar
    menu_bar_frame = tk.Frame(window, bg=theme["bg"], height=25)
    menu_bar_frame.grid(row=0, column=0, columnspan=3, sticky="new")

    # Create a thin frame to represent the bottom border of the menu bar
    border_thickness = 1  # Adjust this value if you want a thicker or thinner border
    border_frame = tk.Frame(window, bg=theme["fg"], height=border_thickness)
    border_frame.grid(row=1, column=0, columnspan=3, sticky="new")

    # Variables to track the currently open menu and its associated label
    currently_open_menu = None
    currently_open_label = None

    def on_menu_open(label, menu):
        nonlocal currently_open_menu, currently_open_label

        # Close any other open menus
        if currently_open_menu:
            currently_open_menu.unpost()
            on_menu_close(currently_open_label, theme_name)

        # Highlight the current label
        opposite_theme = "dark" if theme_name == "light" else "light"
        label.configure(
            bg=themes[opposite_theme]["bg"], fg=themes[opposite_theme]["fg"]
        )

        # Set the current menu and label
        currently_open_menu = menu
        currently_open_label = label

    def on_menu_close(label, theme_name):
        label.configure(bg=themes[theme_name]["bg"], fg=themes[theme_name]["fg"])

    def show_menu(event, label, menu):
        on_menu_open(label, menu)
        menu.post(label.winfo_rootx(), label.winfo_rooty() + label.winfo_height())
        window.bind("<Button-1>", lambda e: on_menu_close(label, theme_name))

    def handle_focus_out(_):
        # Check if a menu is currently open and close it
        if currently_open_menu:
            currently_open_menu.unpost()
            on_menu_close(currently_open_label, theme_name)


    # Bind the <FocusOut> event to handle when the window loses focus
    window.bind("<FocusOut>", handle_focus_out)

    # Create a label (or button) for the "File" menu
    file_label = tk.Label(
        menu_bar_frame, text="File", bg=theme["bg"], fg=theme["fg"], padx=5
    )
    file_label.pack(side=tk.LEFT, padx=5)

    # Create a menu for the "File" label
    file_menu = tk.Menu(window, tearoff=0, bg=theme["bg"], fg=theme["fg"])
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=window.quit)

    # Bind the label to show the menu when clicked
    file_label.bind("<Button-1>", lambda e: show_menu(e, file_label, file_menu))

    # Create a label (or button) for the "View" menu
    view_label = tk.Label(
        menu_bar_frame, text="View", bg=theme["bg"], fg=theme["fg"], padx=5
    )
    view_label.pack(side=tk.LEFT, padx=5)

    # Create a menu for the "View" label
    view_menu = tk.Menu(window, tearoff=0, bg=theme["bg"], fg=theme["fg"])
    view_menu.add_command(label="Toggle Theme", command=toggle_theme)

    # Bind the label to show the menu when clicked
    view_label.bind("<Button-1>", lambda e: show_menu(e, view_label, view_menu))
    
    # Add the "Text" menu
    text_label = tk.Label(
        menu_bar_frame, text="Custom Text", bg=theme["bg"], fg=theme["fg"], padx=5
    )
    text_label.pack(side=tk.LEFT, padx=5)

    text_menu = tk.Menu(window, tearoff=0, bg=theme["bg"], fg=theme["fg"])
    text_menu.add_command(label="Create Text Image", command=create_text_image_window)

    # Add the Help menu with an "Check for Updates" option
    help_label = tk.Label(
        menu_bar_frame, text="Help", bg=theme["bg"], fg=theme["fg"], padx=5
    )
    help_label.pack(side=tk.LEFT, padx=5)

    help_menu = tk.Menu(window, tearoff=0, bg=theme["bg"], fg=theme["fg"])
    help_menu.add_command(label="Check for Updates", command=check_for_updates)

    # Bind the label to show the menu when clicked
    help_label.bind("<Button-1>", lambda e: show_menu(e, help_label, help_menu))



    # Bind the label to show the menu when clicked
    text_label.bind("<Button-1>", lambda e: show_menu(e, text_label, text_menu))

    return menu_bar_frame


def create_text_image_window():
    global text_image_window

    # Retrieve the current theme based on the main window's background color
    current_theme = "dark" if window.cget("bg") == themes["dark"]["bg"] else "light"
    theme = themes[current_theme]
    
    # Check if the window is already open. If it is, destroy the existing window.
    if text_image_window:
        text_image_window.destroy()

    # Variable to hold the original background color for use with the transparency toggle
    original_bg_color = None

    # Default settings: black text on a white background, no transparency
    text_color = tk.StringVar(value="black")
    bg_color = tk.StringVar(value="white")
    bg_transparency = tk.BooleanVar(value=False)


    # Create a new window for the text image tool
    text_window = tk.Toplevel(window)
    text_image_window = text_window
    text_window.configure(bg=theme["bg"])
    text_window.resizable(False, False) 
    
    # Function to allow the user to pick a text color
    def pick_text_color():
        color = colorchooser.askcolor(parent=text_window)[1]
        if color:
            text_color.set(color)
            update_preview()
        # Ensure the text_image_window remains on top
        text_window.lift()

    # Function to allow the user to pick a background color
    def pick_bg_color():
        nonlocal original_bg_color
        color = colorchooser.askcolor(parent=text_window)[1]
        if color:
            # Store the selected color in case the user toggles transparency
            original_bg_color = color
            bg_color.set(color)
            # Turn off transparency if the user picks a new background color
            bg_transparency.set(False)
            update_preview()
        # Ensure the text_image_window remains on top
        text_window.lift()

    # Function to update the preview when settings are changed
    def update_preview(*_):
        image = generate_image()
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(256, 256, image=photo)
        canvas.image = photo

        # Update the appearance of the transparency checkbox based on its state
        if bg_transparency.get():
            bg_transparency_check["bg"] = theme["selected_bg"]
        else:
            bg_transparency_check["bg"] = theme["bg"]

    # Function to handle changes to the transparency setting
    def bg_transparency_changed():
        nonlocal original_bg_color
        if bg_transparency.get():
            # If turning transparency on, remember the current background color
            if not original_bg_color:
                original_bg_color = bg_color.get()
            # Set the background color to the theme color for visual clarity
            bg_color.set(theme["bg"])
        else:
            # If turning transparency off, restore the original background color
            if original_bg_color:
                bg_color.set(original_bg_color)
        update_preview()

    # Function to generate an image with the current settings
    def generate_image(export=False):
        desired_width = int(width_entry.get())
        desired_height = int(height_entry.get())
        scale_factor_w = desired_width / 512
        scale_factor_h = desired_height / 512
        if bg_transparency.get():
            # Use a fully transparent background for the preview if transparency is on
            image_color = (0, 0, 0, 0) if export else bg_color.get()
            image = Image.new("RGBA", (desired_width, desired_height), image_color)
        else:
            # Use the selected background color
            image = Image.new("RGB", (desired_width, desired_height), bg_color.get())
        draw = ImageDraw.Draw(image)
        chosen_font_path = FONTS[font_var.get()]
        try:
            # Load the selected font at the appropriate size
            font = ImageFont.truetype(
                chosen_font_path,
                int(font_size.get() * (scale_factor_w + scale_factor_h) / 2),
            )
        except IOError:
            # Fallback to the default font if the selected font can't be loaded
            font = ImageFont.load_default()
        text_content = input_box.get("1.0", tk.END).strip().upper()
        bbox = draw.textbbox((0, 0), text_content, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        # Calculate where to draw the text to center it in the image
        x = (image.width - text_width) / 2 - bbox[0]
        y = (image.height - text_height) / 2 - bbox[1]
        # Draw the text onto the image
        draw.text((x, y), text_content, font=font, fill=text_color.get())
        # If this is just for the preview, resize the image to fit the preview area
        if not export:
            image = image.resize((512, 512), Image.LANCZOS)
        return image

    # Function to save the generated image to a file
    def save_image():
        image = generate_image(export=True)
        # Ask where to save the image
        file_path = filedialog.asksaveasfilename(
            parent=text_window,
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )
        if file_path:
            image.save(file_path)
        # Ensure the text_image_window remains on top
        text_window.lift()

    # GUI elements for the text image tool
    # Input label for text entry
    input_label = tk.Label(
        text_window, text="Enter Text:", bg=theme["bg"], fg=theme["fg"]
    )
    input_label.pack(pady=(10, 5))

    # Text box for user to input the text
    input_box = tk.Text(text_window, height=3, width=25, bg="white", fg="black")
    input_box.pack(pady=(0, 10))
    input_box.bind("<KeyRelease>", update_preview)

    # Frame for font-related widgets
    font_frame = tk.Frame(text_window, bg=theme["bg"])
    font_frame.pack(pady=(10, 5))

    # Label for font dropdown
    font_label = tk.Label(font_frame, text="Font:", bg=theme["bg"], fg=theme["fg"])
    font_label.grid(row=0, column=0, padx=(0, 5))

    # Dropdown for font selection
    font_var = tk.StringVar(value="Arial")
    font_dropdown = ttk.Combobox(
        font_frame, textvariable=font_var, values=list(FONTS.keys())
    )
    font_dropdown.grid(row=0, column=1, padx=(0, 5))
    font_dropdown.bind("<<ComboboxSelected>>", update_preview)

    # Font size variable
    font_size = tk.IntVar(value=30)

    # Frame for font size slider and related widgets
    font_size_frame = tk.Frame(font_frame, bg=theme["bg"])
    font_size_frame.grid(row=0, column=2, padx=(20, 20), sticky=tk.W)

    # Label for font size
    font_size_label = tk.Label(font_size_frame, text="Font Size:", bg=theme["bg"], fg=theme["fg"])
    font_size_label.grid(row=0, column=0, padx=(0, 5))

    # Entry box for font size
    font_size_entry = tk.Entry(font_size_frame, width=5, textvariable=font_size)
    font_size_entry.grid(row=0, column=1, padx=(0, 10))
    font_size_entry.bind("<KeyRelease>", lambda _: update_preview())

    # Font size slider
    font_slider = tk.Scale(
        font_size_frame,
        from_=1,
        to=1000,
        orient=tk.HORIZONTAL,
        variable=font_size,
        bg=theme["bg"],
        fg=theme["fg"],
        command=lambda _: update_preview(),
    )
    font_slider.grid(row=1, columnspan=2, padx=(0, 0))

    # Checkbox for background transparency
    bg_transparency_check = tk.Checkbutton(
        font_frame,
        text="Transparent Background",
        variable=bg_transparency,
        command=bg_transparency_changed,
        bg=theme["bg"],
        fg=theme["fg"],
        selectcolor=theme["selected_bg"],
        activebackground=theme["bg"],
        activeforeground=theme["fg"],
    )
    bg_transparency_check.grid(row=0, column=3, padx=(20, 0))

    
    

    # Canvas for image preview
    canvas = tk.Canvas(text_window, width=512, height=512, bg="white")
    canvas.pack(pady=20, padx=20)

    # Frame for width and height entry boxes
    dimensions_frame = tk.Frame(text_window, bg=theme["bg"])
    dimensions_frame.pack(pady=(10, 5))

    # Label and entry box for width
    width_entry_label = tk.Label(
        dimensions_frame, text="Width:", bg=theme["bg"], fg=theme["fg"]
    )
    width_entry_label.grid(row=0, column=0)
    width_entry = tk.Entry(dimensions_frame, width=6)
    width_entry.insert(0, "512")
    width_entry.grid(row=0, column=1, padx=(0, 20))

    # Label and entry box for height
    height_entry_label = tk.Label(
        dimensions_frame, text="Height:", bg=theme["bg"], fg=theme["fg"]
    )
    height_entry_label.grid(row=0, column=2)
    height_entry = tk.Entry(dimensions_frame, width=6)
    height_entry.insert(0, "512")
    height_entry.grid(row=0, column=3)

    # Frame for color selection buttons
    color_btn_frame = tk.Frame(text_window, bg=theme["bg"])
    color_btn_frame.pack(pady=5)

    # Button to pick background color
    bg_color_btn = tk.Button(
        color_btn_frame,
        text="Pick Background Color",
        command=pick_bg_color,
        bg=theme["button_bg"],
        fg=theme["fg"],
    )
    bg_color_btn.grid(row=0, column=0, padx=5)

    # Button to pick text color
    text_color_btn = tk.Button(
        color_btn_frame,
        text="Pick Text Color",
        command=pick_text_color,
        bg=theme["button_bg"],
        fg=theme["fg"],
    )
    text_color_btn.grid(row=0, column=1, padx=5)

    # Button to save the generated image
    save_button = tk.Button(
        text_window,
        text="Save Image",
        command=save_image,
        bg=theme["button_bg"],
        fg=theme["fg"],
    )
    save_button.pack(pady=10)

    # Start the main event loop for the window
    text_window.mainloop()


def check_for_updates():
    try:
        repo_url = (
            "https://api.github.com/repos/Bassna/Ez-Image-Effects/releases/latest"
        )
        response = requests.get(repo_url)
        latest_version = response.json()["tag_name"]

        if version_tuple(latest_version) > version_tuple(CURRENT_VERSION):
            answer = messagebox.askyesno(
                "Update Available",
                f"A new version ({latest_version}) is available. Would you like to download it?",
            )
            if answer:
                webbrowser.open(
                    "https://github.com/Bassna/Ez-Image-Effects/releases/latest"
                )
        else:
            messagebox.showinfo(
                "No Update Available",
                "You are using the latest version of the application.",
            )
    except Exception as e:
        messagebox.showerror(
            "Update Check Failed",
            "Failed to check for updates. Please try again later.",
        )
        print(f"Error checking for updates: {e}")


def version_tuple(v):
    # Remove the 'v' prefix and split the version string into a tuple of integers
    return tuple(map(int, v[1:].split(".")))


def toggle_theme():
    global text_image_window

    current_theme = "dark" if window.cget("bg") == themes["dark"]["bg"] else "light"
    new_theme = "light" if current_theme == "dark" else "dark"
    apply_theme(new_theme)

    # Check if the text_image_window is alive
    was_text_window_open = False
    try:
        if text_image_window and text_image_window.winfo_exists():
            was_text_window_open = True
            text_image_window.destroy()
    except NameError:  # In case text_image_window is not defined yet
        pass

    # If the text_image_window was open, recreate it with the updated theme
    if was_text_window_open:
        create_text_image_window()




themes = {
    "dark": {
        "bg": "#2c2c2c",
        "fg": "#ffffff",
        "button_bg": "#3f3f3f",
        "listbox_bg": "#2c2c2c",
        "slider_handle": "#505050",
        "tree_bg": "#2c2c2c",
        "tree_fg": "#ffffff",
        "tree_field_bg": "#2c2c2c",
        "tree_selected_bg": "#505050",
        "selected_bg": "#505050",
    },
    "light": {
        "bg": "#f0f0f0",
        "fg": "#000000",
        "button_bg": "#f0f0f0",
        "listbox_bg": "#f5f5f5",
        "slider_border": "#000000",
        "tree_bg": "#f5f5f5",
        "tree_fg": "#000000",
        "tree_field_bg": "#f0f0f0",
        "tree_selected_bg": "#a0a0a0",
        "selected_bg": "#505050",
    },
}

# Initialize the main application window
window = tk.Tk()
window.resizable(False, False)
window.title("EZ Image Effects")
window.geometry("1180x640")
window.configure(bg="#2c2c2c")  # Set background color to dark gray

# Variables to store states for various toggle options
theme_var = tk.BooleanVar(value=True)  # Set dark theme as default
effect_strength_var = tk.IntVar(value=100)  # Default strength is 0%
num_frames_var = tk.IntVar(value=48)  # Default number of frames is 48


# Define standard fonts and colors for the application
default_font = ("Arial", 10)
header_font = ("Arial", 14, "bold")
button_bg_color = "#3f3f3f"  # Color for buttons (darker gray)
button_fg_color = "#ffffff"  # Text color for buttons (white)

# Organize available effects into distinct categories
effect_categories = {
    "Transformations": [
        "Grow Effect",
        "Radial Zoom Effect",
        "Distorted Mirror Effect",
        "Warp Effect",
        "Wave Effect",
        "No Effect",
    ],
    "Movements": [
        "Left to Right",
        "Top to Bottom",
        "Rotation Effect",
        "Spin Effect",
        "Random Jitter Effect",
    ],
    "Colors & Tones": [
        "Blur Effect",
        "Sharpen Effect",
        "Sepia Tone Effect",
        "Grayscale Effect",
        "Vignette Effect",
        "Color Overlay Effect",
        "Negative Effect",
        "Solarization Effect",
        "Retro Effect",
    ],
    "Special Effects": [
        "Pixelation Effect",
        "Emboss Effect",
        "Edge Highlight Effect",
        "Oil Painting Effect",
        "Watercolor Effect",
        "Mosaic Effect",
        "Lens Flare Effect",
        "Frosted Glass Effect",
        "Static Effect",
    ],
}


# Configure custom Treeview style
style = ttk.Style()
style.configure(
    "Custom.Treeview",
    background=themes["light"]["tree_bg"],
    foreground=themes["light"]["fg"],
)


# Frame for the left side of the main window (for selecting effects)
left_frame = tk.Frame(window, bg="#2c2c2c", padx=10, pady=10)
left_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)

# Frame in the center for displaying an animation preview
preview_frame = tk.Frame(window, bg="#2c2c2c", padx=20, pady=20)
preview_frame.grid(row=2, column=1, sticky="nsew", padx=10, pady=10)

# Canvas within the preview frame to show the actual animation
preview_canvas = tk.Canvas(
    preview_frame, width=512, height=512, bg="#3f3f3f", bd=1, relief="solid"
)
preview_canvas.grid(row=0, column=0, pady=10, sticky="nsew")

# Frame on the right side for the effects queue
right_frame = tk.Frame(window, bg="#2c2c2c", padx=20, pady=20)
right_frame.grid(row=2, column=2, sticky="nsew", padx=10, pady=10)


# Label to indicate the section for selecting effects
tk.Label(
    left_frame,
    text="Select Effects",
    font=header_font,
    bg="#2c2c2c",
    fg=button_fg_color,
).grid(row=0, column=0, sticky="w", pady=(10, 5))

# Treeview widget to display and select available effects
effects_tree = ttk.Treeview(
    left_frame, style="Custom.Treeview", selectmode="extended", height=21
)
effects_tree["show"] = "tree"
effects_tree.grid(row=1, column=0, padx=10, pady=(0, 5), sticky="nsew")

effects_tree.bind("<<TreeviewSelect>>", deselect_categories)


# Frames to group buttons and toggle switches together
checkbox_frame = tk.Frame(left_frame, bg="#2c2c2c")
checkbox_frame.grid(row=2, column=0, pady=(0, 5), sticky="nsew")
toggle_frame = tk.Frame(left_frame, bg="#2c2c2c")
toggle_frame.grid(row=3, column=0, pady=(0, 5), sticky="nsew")

# Add effects and preview buttons
add_effects_button = ttk.Button(
    checkbox_frame, text="Add Effects", command=add_to_queue, width=14
)
add_effects_button.grid(row=0, column=0, pady=5, padx=10, sticky="w")
preview_button = ttk.Button(
    checkbox_frame, text="Preview Effect", command=preview_selected_effect, width=14
)
preview_button.grid(row=0, column=2, pady=5, padx=10, sticky="w")


# Event bindings for the Treeview
effects_tree.bind("<<TreeviewSelect>>", deselect_categories)
effects_tree.bind("<ButtonRelease-1>", handle_selection)

# Label and tree view for the effects queue
tk.Label(
    right_frame,
    text="Effects Queue",
    font=header_font,
    bg="#2c2c2c",
    fg=button_fg_color,
).grid(row=0, column=0, columnspan=2, sticky="w")
queue_tree = DragDropTree(
    right_frame,
    columns=("UUID", "Effect", "Parameters"),
    height=21,
    selectmode="none",
    displaycolumns=("Effect", "Parameters"),
)

queue_tree.heading("#1", text="Effect")
queue_tree.heading("#2", text="Parameters")
# queue_tree.column("#1", width=0, stretch=tk.NO)  # Hide the UUID column
queue_tree.column("#1", width=40)  # Effect column
queue_tree.column("#2", width=20)  # Parameters column

# queue_tree.heading("#1", text="UUID")
queue_tree.heading("#1", text="Effect")
queue_tree.heading("#2", text="Parameters")

queue_tree.grid(row=1, column=0, columnspan=2, pady=10)
queue_tree_data = []  # Additional attribute to store data associated with the queue
queue_tree["show"] = "tree"  # Hide the headers
queue_tree.queue_data = queue_tree_data
queue_tree.bind("<ButtonRelease-1>", handle_queue_selection)

# Populate the Treeview with effects from the defined categories
for category, effects in effect_categories.items():
    if category == "No Effect":
        effects_tree.insert("", "end", text=category, open=False)
    else:
        category_id = effects_tree.insert("", "end", text=category, open=False)
        for effect in effects:
            effect_id = effects_tree.insert(category_id, "end", text="☐ " + effect)

            # Add sub-items for effect strength and number of frames
            effects_tree.insert(
                effect_id, "end", text=f"Effect Strength: {effect_strength_var.get()}%"
            )
            effects_tree.insert(
                effect_id, "end", text=f"Number of Frames: {num_frames_var.get()}"
            )

            # Existing checkboxes for Reverse and Invert Colors
            effects_tree.insert(
                effect_id, "end", text="☐ Reverse", tags=("non_selectable",)
            )
            effects_tree.insert(
                effect_id, "end", text="☐ Invert Colors", tags=("non_selectable",)
            )


# Buttons associated with the effects queue (process, remove, clear, and play preview)
process_queue_button = ttk.Button(
    right_frame, text="Process Queue", command=process_queued_effects, width=14
)
remove_effect_button = ttk.Button(
    right_frame, text="Remove Effect", command=remove_selected_effect, width=14
)
clear_queue_button = ttk.Button(
    right_frame, text="Clear Queue", command=clear_queue, width=14
)
play_preview_button = ttk.Button(
    right_frame, text="Play Preview", command=play_queue_preview, width=14
)

process_queue_button.grid(row=2, column=0, padx=5, pady=5)
remove_effect_button.grid(row=2, column=1, padx=5, pady=5)
clear_queue_button.grid(row=3, column=1, padx=5, pady=5)
play_preview_button.grid(row=3, column=0, padx=5, pady=5)

app.last_image_directory = None
app.last_output_directory = None

# Extract the icon from the .exe and set it as the window icon
icon_path = "EZIcon.ico"
if os.path.exists(icon_path):
    window.iconbitmap(default=icon_path)
elif getattr(sys, "frozen", False):
    icon_path = os.path.join(sys._MEIPASS, icon_path)
    window.iconbitmap(default=icon_path)


# Apply the default theme as Dark
apply_theme("dark")

# Create and style the initial custom menu bar according to the default theme
create_custom_menu_bar("dark")

# Start the application's event loop
window.mainloop()


# Created by Bassna
# Discord - Bassna
