import os
import sys
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import discord
import json
from helper_functions import get_emojis_from_member
from discord.ext import commands
from os import getenv
from dotenv import load_dotenv
from random import choice
from car_classifier.model import CarClassifier, predict_image, transform
from PIL import Image
import requests
import io
import boto3
import os
from botocore.exceptions import ClientError
import logging
import traceback


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
def download_from_s3():
    bucket_name = getenv('AWS_BUCKET_NAME')
    model_key = 'car_classifier.pth'  # The path/name of file in S3
    local_path = 'car_classifier/car_classifier.pth'
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Initialize S3 client
        s3 = boto3.client(
            's3',
            aws_access_key_id=getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=getenv('AWS_SECRET_ACCESS_KEY')
        )
        
        # Download file
        s3.download_file(bucket_name, model_key, local_path)
        print("Model downloaded successfully")
    except ClientError as e:
        print(f"Error downloading model: {e}")
        raise

# Download model if not exists
if not os.path.exists('car_classifier/car_classifier.pth'):
    download_from_s3()

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
print("Bot is starting up...")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    force=True
)
logger = logging.getLogger('car_classifier_bot')

@bot.event
async def on_ready():
    print(f'Bot is ready! Logged in as {bot.user.name}')
    logger.info(f'Bot is ready! Logged in as {bot.user.name}')

@bot.event
async def on_command_error(ctx, error):
    print(f'Command error: {str(error)}')
    logger.error(f'Command error: {str(error)}')
    logger.error(traceback.format_exc())
    await ctx.send(f"An error occurred: {str(error)}")

@bot.command()
async def loveme(ctx):
    compliment = choice(complimentsJSON)
    await ctx.send(compliment)

@bot.event
async def on_message(message):
    try:
        # Ignore messages from the bot itself
        if message.author == bot.user:
            return

        print(f"Raw message received: {message}")
        logger.info(f"Raw message received: {message}")

        # Immediately log attachment info
        if message.attachments:
            for idx, attachment in enumerate(message.attachments):
                print(f"Attachment {idx} details:")
                print(f"  Filename: {attachment.filename}")
                print(f"  Size: {attachment.size}")
                print(f"  URL: {attachment.url}")
                print(f"  Content type: {attachment.content_type}")
                
                logger.info(f"Attachment {idx} details:")
                logger.info(f"  Filename: {attachment.filename}")
                logger.info(f"  Size: {attachment.size}")
                logger.info(f"  URL: {attachment.url}")
                logger.info(f"  Content type: {attachment.content_type}")

            # Try to process the first image attachment
            try:
                if message.attachments[0].filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    print("Starting image processing...")
                    logger.info("Starting image processing...")
                    await process_car_image(message)
                else:
                    print(f"Invalid file type: {message.attachments[0].filename}")
                    logger.info(f"Invalid file type: {message.attachments[0].filename}")
                    await message.channel.send("Please upload a PNG or JPG image!")
            except Exception as e:
                print(f"Error in attachment processing: {str(e)}")
                logger.error(f"Error in attachment processing: {str(e)}")
                logger.error(traceback.format_exc())
                await message.channel.send(f"Error processing attachment: {str(e)}")

    except Exception as e:
        print(f"Error in message handling: {str(e)}")
        logger.error(f"Error in message handling: {str(e)}")
        logger.error(traceback.format_exc())
        await message.channel.send(f"An error occurred: {str(e)}")

    # Make sure to process commands as well
    await bot.process_commands(message)

@bot.command()
async def identify(ctx):
    print(f"Command received from {ctx.author}")
    logger.info(f"Command received from {ctx.author}")
    
    try:
        if not ctx.message.attachments:
            print("No attachment found in message")
            logger.info("No attachment found in message")
            await ctx.send("Please attach an image!")
            return
            
        attachment = ctx.message.attachments[0]
        print(f"Attachment received: {attachment.filename}")
        logger.info(f"Attachment received: {attachment.filename} ({attachment.size} bytes)")
        
        if not attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Invalid file type: {attachment.filename}")
            logger.info(f"Invalid file type: {attachment.filename}")
            await ctx.send("Please upload a PNG or JPG image!")
            return
            
        await process_car_image(ctx)
        
    except Exception as e:
        print(f"Error in identify command: {str(e)}")
        logger.error(f"Error in identify command: {str(e)}")
        logger.error(traceback.format_exc())
        await ctx.send(f"Sorry, an error occurred: {str(e)}")

async def process_car_image(message):
    attachment = message.attachments[0]
    logger.info(f"Processing image: {attachment.filename}")
    logger.info(f"Image size: {attachment.size} bytes")
    
    # Create temp directory if it doesn't exist
    os.makedirs('temp', exist_ok=True)
    temp_path = os.path.join('temp', 'temp_image.jpg')
    
    try:
        # Download image
        logger.info("Downloading image...")
        await attachment.save(temp_path)
        logger.info(f"Image saved to {temp_path}")
        
        # Validate file
        if not os.path.exists(temp_path):
            raise FileNotFoundError("Temp file was not created")
        
        file_size = os.path.getsize(temp_path)
        logger.info(f"Saved file size: {file_size} bytes")
        
        if file_size == 0:
            raise ValueError("Downloaded file is empty")

        # Process image
        logger.info("Running prediction...")
        result = predict_image(model, temp_path, device, class_dict)
        logger.info(f"Prediction result: {result}")
        
        await message.channel.send(f"This car appears to be a {result}")

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"File error: {str(e)}")
        await message.channel.send(f"Sorry, I couldn't process that image: {str(e)}")
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        await message.channel.send("Sorry, something went wrong while processing your request.")
    
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info("Cleaned up temporary file")
            except Exception as e:
                logger.error(f"Cleanup error: {str(e)}")
#####################################

bot.run(TOKEN)
