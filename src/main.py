import discord
import json
from helper_functions import get_emojis_from_member
from discord.ext import commands
from os import getenv
from dotenv import load_dotenv
from random import choice
from car_classifier.model import CarClassifier, predict_image
from PIL import Image
import torch
import requests
import io

# BOT SETUP VARS
#####################################
load_dotenv()
TOKEN = getenv('DISCORD_TOKEN')
GUILD = getenv('DISCORD_GUILD')
intents = discord.Intents.default()
intents.members = True
intents.message_content = True
bot = commands.Bot(intents = intents, command_prefix='!')
#####################################

#MODEL SETUP
#####################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CarClassifier(num_classes=196, train_resnet=False)
model.load_state_dict(torch.load('car_classifier/car_classifier.pth', map_location=device))
model.to(device)
#####################################

# JSON IMPORT
#####################################
membersJSON = []
complimentsJSON = []
with open("json/members.json") as f:
  membersJSON = json.load(f)
with open("json/compliments.json") as f:
  complimentsJSON = json.load(f)

with open('car_classifier/class_dict.json', 'r') as f:
  class_dict = json.load(f)
#####################################

# BOT FUNCTIONS
#####################################
@bot.event
async def on_message(message):
    if message.author == bot.user:
      return

    if message.author.id not in [member['id'] for member in membersJSON]:
      return

    emojis = get_emojis_from_member(membersJSON, message.author.id)

    for e in emojis:
      await message.add_reaction(e)

    await bot.process_commands(message)

@bot.event
async def on_ready():
    print(f"Bot connected as {bot.user.name}")

@bot.command()
async def loveme(ctx):
    compliment = choice(complimentsJSON)
    await ctx.send(compliment)

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.attachments:
        for attachment in message.attachments:
            if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                try:
                    # Download and process image
                    response = requests.get(attachment.url)
                    image = Image.open(io.BytesIO(response.content)).convert('RGB')
                    
                    # Get prediction
                    predicted_class = predict_image(model, image, device, class_dict)
                    
                    # Send response
                    await message.channel.send(f"This appears to be a {predicted_class}")
                except Exception as e:
                    await message.channel.send(f"Sorry, I couldn't process that image: {str(e)}")

    await bot.process_commands(message)
#####################################

bot.run(TOKEN)
