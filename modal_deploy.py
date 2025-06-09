import modal
import discord
from discord.ext import commands
import os
import torch
from PIL import Image
import json
import io
import aiohttp
import clip
import logging
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = modal.App("DiscordBot")

# Create image with dependencies
image = (
    modal.Image.from_registry("pytorch/pytorch:latest")
    .apt_install("git")
    .pip_install(
        "discord.py",
        "Pillow",
        "ftfy",
        "regex",
        "tqdm",
        "git+https://github.com/openai/CLIP.git"
    )
)

# Add the file to the image
image = image.add_local_file("car_classifier/class_dict.json", "/root/car_classifier/class_dict.json")

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("discord-secret")],
    gpu="T4",
    timeout=600  # Increase timeout to 10 minutes
)
def main():
    logger.info("Starting Discord bot...")
    
    # Initialize bot
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix='!', intents=intents)
    
    # Rate limiting setup
    processing_queue = []
    user_last_request = defaultdict(lambda: datetime.min)
    USER_COOLDOWN = 10  # seconds between requests per user
    MAX_QUEUE_SIZE = 5  # maximum number of images in queue
    
    # Load CLIP model
    logger.info("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=True)
    logger.info(f"CLIP model loaded. Using device: {device}")
    
    # Load class dictionary
    logger.info("Loading class dictionary...")
    with open('/root/car_classifier/class_dict.json', 'r') as f:
        class_dict = json.load(f)
        car_classes = [' '.join(class_dict[str(i)].split()[:-1]) for i in range(len(class_dict))]
    logger.info("Class dictionary loaded")
    
    # Create text descriptions and tokens
    text_descriptions = [f"a photo of a {car_class}" for car_class in car_classes]
    text_tokens = clip.tokenize(text_descriptions).to(device)

    async def process_image_with_timeout(message, attachment):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(attachment.url) as resp:
                    if resp.status != 200:
                        await message.channel.send("Failed to download the image!")
                        return
                    image_data = await resp.read()
                    image = Image.open(io.BytesIO(image_data))

            # Process with timeout
            try:
                # Create a coroutine for the processing
                async def process_image():
                    # Preprocess image
                    image_input = preprocess(image).unsqueeze(0).to(device)
                    
                    # Get predictions
                    with torch.no_grad():
                        image_features = model.encode_image(image_input)
                        text_features = model.encode_text(text_tokens)
                        
                        # Normalize features
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        
                        # Calculate similarity
                        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                        
                        # Get top 5 predictions
                        top_5_probs, top_5_indices = torch.topk(similarity[0], 5)
                        
                        # Create response message
                        response = "**Car Classification Results:**\n"
                        for prob, idx in zip(top_5_probs.cpu().numpy(), top_5_indices.cpu().numpy()):
                            response += f"• {class_dict[str(idx)]}: {prob:.2%}\n"
                        
                        await message.channel.send(response)

                # Run with timeout
                await asyncio.wait_for(process_image(), timeout=30.0)  # 30 second timeout

            except asyncio.TimeoutError:
                await message.channel.send("Processing took too long! Please try again.")
                
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            await message.channel.send(f"An error occurred while processing the image. Please try again later.")
        finally:
            # Remove from processing queue
            if message.author.id in processing_queue:
                processing_queue.remove(message.author.id)

    @bot.event
    async def on_ready():
        logger.info(f'Bot is ready! Logged in as {bot.user.name}')
        logger.info(f'Bot ID: {bot.user.id}')
        logger.info(f'Connected to {len(bot.guilds)} guilds')

    @bot.event
    async def on_message(message):
        if message.author == bot.user:
            return

        if message.attachments:
            # Check if user is in cooldown
            last_request = user_last_request[message.author.id]
            time_since_last = datetime.now() - last_request
            if time_since_last.seconds < USER_COOLDOWN:
                await message.channel.send(f"Please wait {USER_COOLDOWN - time_since_last.seconds} seconds before sending another image.")
                return

            # Check if queue is full
            if len(processing_queue) >= MAX_QUEUE_SIZE:
                await message.channel.send("Too many images being processed. Please try again in a moment.")
                return

            # Check if user already has an image being processed
            if message.author.id in processing_queue:
                await message.channel.send("Please wait for your previous image to finish processing.")
                return

            # Process images
            for attachment in message.attachments:
                if attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    logger.info(f"Processing image: {attachment.filename}")
                    
                    # Add to processing queue and update last request time
                    processing_queue.append(message.author.id)
                    user_last_request[message.author.id] = datetime.now()
                    
                    # Process image
                    await process_image_with_timeout(message, attachment)
        
        await bot.process_commands(message)

    # Start the bot
    token = os.environ["DISCORD_TOKEN"]
    if not token:
        raise ValueError("Discord token is empty")
    logger.info("Discord token found, starting bot...")
    bot.run(token)

if __name__ == "__main__":
    modal.runner.deploy_stub(app)