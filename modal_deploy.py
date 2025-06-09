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

# Create Modal stub
stub = modal.Stub("discord-bot")

# Create image with dependencies
image = modal.Image.debian_slim().pip_install(
    "discord.py",
    "torch",
    "Pillow",
    "ftfy",
    "regex",
    "tqdm",
    "git+https://github.com/openai/CLIP.git"
)

@stub.cls(
    image=image,
    secret=modal.Secret.from_name("discord-secret"),
    gpu="T4",  # We'll keep the GPU since we're doing image processing
)
class DiscordBot:
    def __enter__(self):
        # Initialize bot
        intents = discord.Intents.default()
        intents.message_content = True
        self.bot = commands.Bot(command_prefix='!', intents=intents)
        
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=True)
        
        # Load class dictionary
        with open('car_classifier/class_dict.json', 'r') as f:
            self.class_dict = json.load(f)
            self.car_classes = [' '.join(self.class_dict[str(i)].split()[:-1]) for i in range(len(self.class_dict))]
        
        # Create text descriptions and tokens
        self.text_descriptions = [f"a photo of a {car_class}" for car_class in self.car_classes]
        self.text_tokens = clip.tokenize(self.text_descriptions).to(self.device)
        
        # Set up bot events
        @self.bot.event
        async def on_ready():
            print(f'Bot is ready! Logged in as {self.bot.user.name}')

        @self.bot.event
        async def on_message(message):
            if message.author == self.bot.user:
                return

            if message.attachments:
                for attachment in message.attachments:
                    if attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        await self.process_car_image(message, attachment)
            
            await self.bot.process_commands(message)

    async def process_car_image(self, message, attachment):
        try:
            # Download the image
            async with aiohttp.ClientSession() as session:
                async with session.get(attachment.url) as resp:
                    if resp.status != 200:
                        await message.channel.send("Failed to download the image!")
                        return
                    image_data = await resp.read()
                    image = Image.open(io.BytesIO(image_data))

            # Preprocess image
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(self.text_tokens)
                
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
                    response += f"• {self.class_dict[str(idx)]}: {prob:.2%}\n"
                
                await message.channel.send(response)

        except Exception as e:
            await message.channel.send(f"An error occurred: {str(e)}")

    @modal.method()
    def run(self):
        self.bot.run(os.environ["DISCORD_TOKEN"])

@stub.function()
def main():
    bot = DiscordBot()
    bot.run()

if __name__ == "__main__":
    stub.serve()