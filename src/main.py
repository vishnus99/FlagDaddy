import discord
import json
from helper_functions import get_emojis_from_member
from discord.ext import commands
from os import getenv
from dotenv import load_dotenv
from random import choice

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

# JSON IMPORT
#####################################
membersJSON = []
complimentsJSON = []
with open("json/members.json") as f:
  membersJSON = json.load(f)
with open("json/compliments.json") as f:
  complimentsJSON = json.load(f)
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
#####################################

bot.run(TOKEN)
