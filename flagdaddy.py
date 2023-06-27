import discord
from discord.ext import commands
import os
from dotenv import load_dotenv
import random
from discord.ext import commands

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

    
compliments = [
"You are amazing!",
"You are beautiful inside and out!",
"You are incredibly talented!",
"You brighten up everyone's day!",
"You have a heart of gold!",
"You are doing great things!",
"You're wise beyond your years, always making thoughtful decisions.",
"You have a bright future ahead of you. Your determination is inspiring!",
"You're incredibly talented and have a unique perspective on things.",
"You have a remarkable work ethic. Your dedication is commendable!",
"You're a natural leader, always stepping up to the plate.",
"You bring a positive energy to every room you enter. It's infectious!",
"You have a strong sense of empathy and always show kindness to others.",
"You're a great problem solver, finding innovative solutions to challenges.",
"You have a genuine passion for learning and self-improvement.",
"You have a creative mind and constantly inspire those around you.",
"You're not afraid to take risks and embrace new opportunities.",
"You're adaptable and handle change with grace and resilience.",
"You have an incredible sense of style. Your fashion choices are always on point!",
"You have a magnetic personality. People are naturally drawn to you!",
"You have a strong sense of integrity. Your honesty and values are admirable.",
"You're a fantastic communicator. You express yourself with clarity and confidence.",
"You have an adventurous spirit. Your willingness to explore and try new things is inspiring!",
"You have a great sense of humor. Your jokes and wit never fail to bring joy to others.",
"You have a genuine curiosity about the world. Your thirst for knowledge is commendable!",
"You're a reliable friend. Your loyalty and support mean the world to those around you.",
"You have a nurturing nature. You always make others feel cared for and loved.",
"You have an entrepreneurial mindset. Your ambition and drive set you up for success!",
"You have a strong sense of social responsibility. Your dedication to making a difference is inspiring!",
"You have a magnetic charisma. People are naturally drawn to your presence!"
"Damn girl, you thiccer than a bowl of oatmeal"
"You? Would."
"How you fit all that in them pants"
"You smell good, I like that."
""
]

@bot.command
async def loveme(ctx):
    compliment = random.choice(compliments)
    await ctx.send(compliment)

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
