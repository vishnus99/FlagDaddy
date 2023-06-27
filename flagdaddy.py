import discord
from discord.ext import commands
import os
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')
intents = discord.Intents.default()
intents.members = True
client = discord.Client(intents=intents)

bot = commands.Bot(intents = intents, command_prefix='!')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Hard-coded user nationality: 
    user_nationality = {
        '135863950045741056': 'India',  
        '203034834530992129': 'India',   
        '208997009275748352': 'USA',  
        '264556234802200578': 'Vietnam',
        '306229200740810753': 'Ecuador',
        '323860438750068737': 'Korea',
        '327175813902499840': 'USA',
        '444574782554636288': 'India',
        '472646061563969537': 'Puerto Rico',
        '605287052497518609': 'India',
        '728136669848535071': 'Philippines',
        '826255462767788082': 'India'
    }

    # Get the nationality based on the user ID
    nationality = user_nationality.get(str(message.author.id))

    if nationality:
        flag_emoji = get_flag_emoji(nationality)
        await message.add_reaction(flag_emoji)

    await bot.process_commands(message)

def get_flag_emoji(nationality):
    flags = {
        'USA': '🇺🇸',
        'India': '🇮🇳',
        'Vietnam': '🇻🇳',
        'Ecuador': '🇪🇨',
        'Korea': '🇰🇷',
        'Puerto Rico': '🇵🇷',
        'Philippines': '🇵🇭' }
    
    return flags.get(nationality, '')

bot.run(TOKEN)
