import discord
import json
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
membersJSON = 0
complimentsJSON = 0
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

    if message.author.id not in [member.['id'] for member in membersJSON]:
      return

    emojis = filter(lambda member: member['id'] == message.author.id, membersJSON)

    for e in emojis:
      await message.add_reaction(e)

    await bot.process_commands(message)

@bot.event
async def on_ready():
    print(f"Bot connected as {bot.user.name}")

@bot.command()
async def loveme(ctx):
    compliment = choice(compliments)
    await ctx.send(compliment)
#####################################

bot.run(TOKEN)
