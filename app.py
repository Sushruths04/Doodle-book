"""
DoodleBook — HF ZeroGPU Version

Free T4 GPU on Hugging Face Spaces!
No Modal needed.
"""

import gradio as gr
import os
import sys
import torch
try:
    import spaces
except ModuleNotFoundError:
    # `spaces` only exists on HF ZeroGPU. Off-HF (local/dev) provide a no-op so
    # the app still runs; generation then uses whatever local GPU/CPU exists.
    class _SpacesShim:
        @staticmethod
        def GPU(*args, **kwargs):
            if args and callable(args[0]):      # bare @spaces.GPU
                return args[0]
            def deco(fn):                       # @spaces.GPU(duration=...)
                return fn
            return deco
    spaces = _SpacesShim()
import json
import time
import tempfile
import logging
import struct
import re

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    FLUX_MODEL, STORY_MODEL, TTS_MODEL,
    GENERATION_PARAMS, SAMPLE_BOOK_PATH, BASE_SEED, page_seed,
    DEFAULT_VOICE, voice_design,
)
from book_builder import (
    build_book_html, export_pdf, magic_loader_html,
    build_coloring_html, export_coloring_pdf,
)
from ui.layout import create_layout

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ZeroGPU sets SPACES_ZERO_GPU. On the Space we load models on cuda at IMPORT
# (a CUDA-emulation layer makes that work without a real GPU); lazy-loading
# inside @spaces.GPU is explicitly discouraged and was why FLUX kept failing
# → sketch. Guarded so a local/dev import doesn't try to pull ~20GB of weights.
ON_ZEROGPU = bool(os.environ.get("SPACES_ZERO_GPU"))

_FLUX_PIPE = None
_STORY_MODEL = None
_STORY_TOKENIZER = None
_TTS_MODEL = None
_LOAD_ERRORS = {}


def load_flux():
    """FLUX image pipeline placed on cuda at module scope (the ZeroGPU pattern).
    No enable_model_cpu_offload() — that fights ZeroGPU's device management."""
    global _FLUX_PIPE
    if _FLUX_PIPE is None:
        from diffusers import Flux2KleinPipeline
        logger.info(f"Loading image model: {FLUX_MODEL.hub_id}")
        pipe = Flux2KleinPipeline.from_pretrained(
            FLUX_MODEL.hub_id, torch_dtype=torch.bfloat16,
        )
        pipe.to("cuda")
        _FLUX_PIPE = pipe
    return _FLUX_PIPE


def load_story():
    global _STORY_MODEL, _STORY_TOKENIZER
    if _STORY_MODEL is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        logger.info(f"Loading story model: {STORY_MODEL.hub_id}")
        _STORY_TOKENIZER = AutoTokenizer.from_pretrained(
            STORY_MODEL.hub_id, trust_remote_code=True,
        )
        _STORY_MODEL = AutoModelForCausalLM.from_pretrained(
            STORY_MODEL.hub_id, torch_dtype=torch.float16, trust_remote_code=True,
        ).to("cuda").eval()
    return _STORY_MODEL, _STORY_TOKENIZER


def load_tts():
    global _TTS_MODEL
    if _TTS_MODEL is None:
        from voxcpm import VoxCPM
        logger.info(f"Loading TTS model: {TTS_MODEL.hub_id}")
        # load_denoiser=True enables voice cloning (Custom Voice option)
        _TTS_MODEL = VoxCPM.from_pretrained(TTS_MODEL.hub_id, load_denoiser=True)
    return _TTS_MODEL


if ON_ZEROGPU:
    for _name, _loader in (
        ("flux", load_flux), ("story", load_story), ("tts", load_tts),
    ):
        try:
            _loader()
        except Exception as _e:
            _LOAD_ERRORS[_name] = repr(_e)
            logger.exception(f"Module-level load failed for {_name}")

COLOR_ART_STYLE = (
    "hand-drawn crayon children's storybook illustration, soft waxy crayon "
    "texture and visible crayon strokes, warm colorful crayon shading, "
    "simple friendly shapes, looks drawn by hand with crayons"
)
COLOR_PAGE_SUFFIX = "full colorful background scene, the character clearly visible."
LINE_ART_STYLE = (
    "children's coloring book page, pure black ink outlines on pure white paper, "
    "clean contour lines, no color, no gray, no shading, no texture, "
    "no hatching, no pencil marks, open spaces to color"
)
LINE_ART_SUFFIX = (
    "simple clean background shapes, same composition, thick readable outlines, "
    "no filled black areas, no extra sketch marks."
)

import random as _random

# Multiple story arcs per theme (3 variants each). build_story_locally picks one at random.
THEME_TEMPLATES = {
    "brave adventure": [
        [   # arc A: overcoming fear of the dark
            ("{hero} loved golden daytime more than anything, but when the sun went down, a cold shiver always crept up their back. The shadows in the forest path twisted into strange shapes, and {hero}'s heart would beat very fast. {hero} dreamed of being brave — but brave felt so very far away.", "{hero} standing nervously at the edge of a dark forest path at dusk, a tiny lantern in hand"),
            ("One night, a tiny bird cried out from deep inside the dark trees — a small, lost sound. {hero} stood very still at the forest's edge, listening hard. The bird needed help, and {hero} was the only one close enough to hear it.", "{hero} listening with wide eyes at the forest entrance, a faint bird cry coming from the dark trees"),
            ("{hero} took one shaky breath and stepped onto the dark path, one careful foot at a time. The lantern cast a warm golden circle around {hero}, pushing back the shadows just enough. 'One step at a time,' {hero} whispered to the lantern.", "{hero} stepping carefully down the dark forest path, holding a small glowing lantern"),
            ("Inside the trees, something surprising happened — the dark was full of beauty, not danger. Moonlight turned the leaves silver and made the dewdrops sparkle like tiny stars. {hero} felt the fear beginning to melt, replaced by wide-eyed wonder.", "{hero} standing in moonlit forest, looking around in amazement at glowing leaves and sparkling dewdrops"),
            ("{hero} found the baby bird on a low branch, ruffled and trembling. Gently, gently, {hero} cupped both hands around the tiny creature and felt its quick little heartbeat. Walking home through the moonlit trees, {hero} felt ten times taller than before.", "{hero} carefully cradling a tiny bird in cupped hands, walking home through the glowing moonlit forest"),
            ("Back in the warm light of home, {hero} placed the bird safely in a nest of soft cloth. Brave hearts don't need the fear to disappear — they just need to start walking, {hero} had discovered. {hero} looked out at the dark night and, for the very first time, thought it looked beautiful.", "{hero} smiling proudly in the warm doorway light, holding the safe little bird, with the moonlit forest behind"),
        ],
        [   # arc B: the broken bridge
            ("{hero} set off to bring bread to grandmother across the river.", "{hero} walking cheerfully with a basket of bread"),
            ("But the bridge had a big crack — too dangerous to cross!", "{hero} staring at a broken bridge over a rushing river"),
            ("{hero} looked around and thought very hard.", "{hero} standing still, eyes closed, thinking carefully"),
            ("Nearby, fallen logs could make a safe path!", "{hero} spotting logs and having a bright idea"),
            ("{hero} rolled the logs into place, one by one.", "{hero} working hard rolling logs to build a crossing"),
            ("Grandmother hugged {hero} tight — cleverness gets you there!", "{hero} and grandmother hugging warmly on the other side"),
        ],
        [   # arc C: the lost kite
            ("{hero}'s favourite kite flew away in a big gust of wind.", "{hero} watching the kite soar away into the sky"),
            ("It landed at the very top of the tallest oak tree.", "{hero} staring up at the kite caught high in a tree"),
            ("{hero} asked the squirrels if they could help reach it.", "{hero} talking to two friendly squirrels on a branch"),
            ("The squirrels agreed — if {hero} would share some acorns.", "{hero} nodding happily and handing over a bag of acorns"),
            ("Together they climbed and tugged the kite free.", "{hero} and the squirrels cheering as the kite comes loose"),
            ("{hero} learned that sharing always brings something back.", "{hero} flying the kite high with the squirrels watching"),
        ],
    ],
    "making a new friend": [
        [   # arc A: the new kid
            ("{hero} stood at the entrance of the new place with a tight feeling in their chest. Everyone inside already seemed to know each other — laughing, running, belonging. {hero} didn't know a single name, and the morning felt very long.", "{hero} standing shyly at a doorway watching other children play happily together"),
            ("Then {hero} noticed someone else — sitting alone by the fence, watching the others too. The child had the same tight look on their face that {hero} had. Something about that made {hero} feel a little less alone.", "{hero} noticing a lonely child sitting quietly by a fence, looking the same way {hero} feels"),
            ("{hero} took a deep breath, walked over, and said the five most important words: 'Hi. Do you want to play?' The words came out a bit wobbly, but they came out. The other child looked up slowly.", "{hero} walking towards the lonely child with a warm, nervous smile, hand raised in greeting"),
            ("The child's whole face changed — like sunshine breaking through clouds. 'Yes,' they said, and both of them stood up at exactly the same time. {hero} had never been so glad to ask a question.", "{hero} and the new friend laughing together as they stand up at the same time"),
            ("By lunchtime, {hero} and the new friend were making up the rules of an entirely new game. They had invented their own handshake and a secret name for their team. The morning that had felt so long was suddenly over.", "{hero} and the new friend playing their new game together, both laughing"),
            ("{hero} waved goodbye at the end of the day with a full, happy heart. One small hello had turned a stranger into a friend, and a lonely morning into the best day in a long time. {hero} had learned that the bravest thing can be the simplest thing.", "{hero} and the new friend waving goodbye happily, both tired and smiling"),
        ],
        [   # arc B: the robot who didn't know how
            ("{hero} met a little robot who had never played before.", "{hero} meeting a small friendly robot in a sunny park"),
            ("The robot could count to a million but didn't know how to laugh.", "{hero} looking curiously at the robot"),
            ("{hero} taught the robot how to blow bubbles.", "{hero} showing the robot how to blow a big soapy bubble"),
            ("The first bubble the robot made was huge — they both gasped!", "{hero} and the robot staring at a giant rainbow bubble"),
            ("Then {hero} taught the robot the best knock-knock joke.", "{hero} whispering a joke to the robot, who looks confused"),
            ("The robot laughed for the very first time, and they both felt warm.", "{hero} and the robot laughing together in golden light"),
        ],
        [   # arc C: opposite neighbours
            ("{hero} loved loud music; the neighbour's cat loved total silence.", "{hero} dancing loudly while a cat glares from the fence"),
            ("They looked at each other across the fence every day.", "{hero} and the cat eyeing each other from opposite sides"),
            ("One rainy afternoon, the cat got stuck in a puddle.", "{hero} seeing the cat splashing helplessly in the rain"),
            ("{hero} rushed over with an umbrella and warm towel.", "{hero} wrapping the soggy cat in a big fluffy towel"),
            ("The cat purred loudly — which sounded like music to {hero}.", "{hero} grinning as the cat purrs on their lap"),
            ("{hero} learned that kindness finds harmony in differences.", "{hero} and the cat sharing headphones by the window"),
        ],
    ],
    "overcoming a fear": [
        [   # arc A: scared of thunder
            ("Every time thunder rumbled, {hero} dove under the blankets and held on very tight. The sound was so big and so sudden — it seemed to fill the whole world. {hero} had tried to be brave, but the thunder was always louder than the bravery.", "{hero} burrowed completely under a pile of colourful blankets while a storm rages outside"),
            ("One stormy afternoon, Grandpa came and sat quietly beside {hero}'s blanket fort. He didn't say 'don't be scared' or 'it's nothing.' He just sat there, calm and warm, until {hero} peeked out. 'Want to know a secret?' he whispered.", "{hero} peeking out from under the blankets to see Grandpa sitting calmly and smiling"),
            ("'Thunder,' said Grandpa softly, 'is just two big clouds bumping into each other by accident — and then saying sorry very loudly.' {hero} blinked. That was not what {hero} had expected thunder to be. A tiny smile appeared.", "{hero} listening with wide eyes to Grandpa's explanation, a small surprised smile on their face"),
            ("{hero} closed both eyes and tried to imagine it — two enormous fluffy clouds bonking heads and making embarrassed faces. A giggle escaped before {hero} could stop it. The next rumble came and for the first time, it sounded almost funny.", "{hero} grinning with eyes closed, imagining two cartoon clouds bumping together and looking sheepish"),
            ("BOOM! The biggest thunder of the night shook the windows — and {hero} burst out laughing. Grandpa laughed too, and together they ran to the window to watch the lightning put on its show. The storm looked completely different now.", "{hero} and Grandpa laughing at the window together as dramatic lightning flashes outside in the dark"),
            ("When the storm finally passed, {hero} and Grandpa stepped outside and breathed in the clean, rain-washed air. {hero} had learned something that night that no book could teach — understanding a thing is the first step to not being afraid of it. {hero} looked up at the clearing sky and felt proud.", "{hero} and Grandpa standing outside after the storm, looking up at a clearing sky with smiles"),
        ],
        [   # arc B: the deep swimming pool
            ("{hero} loved watching others swim but wouldn't go past the steps.", "{hero} sitting on the pool steps, looking at the water"),
            ("'What if I sink?' {hero} thought every single time.", "{hero} staring nervously at the deep end"),
            ("A kind teacher floated a pool noodle close to {hero}.", "{hero} eyeing a colourful foam noodle in the water"),
            ("{hero} grabbed it tightly and floated for the first time.", "{hero} floating on the noodle, surprised and delighted"),
            ("Slowly {hero} kicked — and actually moved through the water!", "{hero} kicking their legs, moving through the pool"),
            ("{hero} found that courage grows one tiny step at a time.", "{hero} smiling proudly in the water, arms wide open"),
        ],
        [   # arc C: the big dog next door
            ("{hero} ran past the gate whenever the big dog barked.", "{hero} sprinting past a big dog behind a garden gate"),
            ("One day the gate was open — and the dog sat very still.", "{hero} stopping in surprise to see the gate open"),
            ("The dog had a thorn in its paw and looked at {hero} sadly.", "{hero} noticing the dog holding up a hurt paw"),
            ("{hero} knelt down slowly and carefully pulled out the thorn.", "{hero} gently helping the dog with a very careful hand"),
            ("The dog licked {hero}'s hand with a big warm tongue.", "{hero} giggling as the dog licks their face"),
            ("{hero} learned that courage and kindness are the same thing.", "{hero} and the big dog walking happily side by side"),
        ],
    ],
    "helping someone": [
        [   # arc A: the broken toy shop
            ("{hero} was walking past the old toy shop when the sound of quiet crying stopped them in their tracks. Through the window, {hero} could see the toy-maker sitting on the floor with her head in her hands. A long wooden shelf had fallen and scattered twenty carefully painted toys across the floorboards.", "{hero} stopping outside the toy shop window, seeing the sad toy-maker among broken toys"),
            ("The toy-maker looked up at {hero} with red eyes. 'Years of work,' she said softly, 'all broken in a moment.' {hero} looked at the jumble of wings, wheels, and tiny painted faces. Without being asked, {hero} stepped inside and knelt down.", "{hero} kneeling on the toy-shop floor, gently picking up pieces of broken toys"),
            ("{hero} sorted the pieces carefully — wings in one pile, wheels in another, tiny hats and buttons in a third. The toy-maker watched for a moment, then picked up the glue and sat down beside {hero}. They didn't need to say anything. They just began.", "{hero} and the toy-maker sitting together on the shop floor, sorting toy pieces into careful groups"),
            ("Piece by careful piece, the toys came back together. {hero} held the pieces steady while the toy-maker's practiced hands glued and pressed. Some were wonky; some had chips missing. But each one was whole again, and that was what mattered.", "{hero} and the toy-maker carefully gluing a toy together, their hands working side by side"),
            ("The very last toy was a small wooden bird with a key in its back. The toy-maker wound it gently, set it on the counter, and they both held their breath. Then — the most perfect little song rang through the quiet shop. {hero} felt the hairs on their arms stand up.", "{hero} and the toy-maker listening with soft smiles as a small wooden bird on the counter plays its song"),
            ("{hero} and the toy-maker stood at the door as the afternoon light turned golden. 'You didn't have to stop,' said the toy-maker quietly. 'I know,' said {hero}. {hero} had learned that helping someone on their worst day costs nothing — but can change everything.", "{hero} and the toy-maker standing at the sunny shop door, shaking hands warmly with smiles"),
        ],
        [   # arc B: the garden in winter
            ("{hero} noticed old Mrs. Patel's garden was bare and sad.", "{hero} looking over a fence at a grey winter garden"),
            ("Mrs. Patel was too sick to plant her spring seeds.", "{hero} seeing Mrs. Patel watching sadly from her window"),
            ("{hero} made a plan and came back with a trowel.", "{hero} arriving at the garden gate with a basket of seeds"),
            ("Row by row, {hero} pressed seeds into the cold soil.", "{hero} carefully planting seeds in the dark garden earth"),
            ("Weeks later, the garden burst into pink and yellow flowers.", "{hero} and Mrs. Patel gasping at the blooming garden"),
            ("{hero} learned that small acts of kindness bloom in their own time.", "{hero} and Mrs. Patel sitting together among the flowers"),
        ],
        [   # arc C: lost dog returns home
            ("{hero} found a shaggy dog trembling alone in the rain.", "{hero} spotting a scared wet dog in the rain"),
            ("The dog had a tag — but the address was smudged.", "{hero} squinting at a blurry metal tag on the dog's collar"),
            ("{hero} drew the dog's face on a poster and put it everywhere.", "{hero} sticking handmade posters on poles and windows"),
            ("A little girl called — she'd been crying for two days!", "{hero} on the phone, looking relieved and happy"),
            ("{hero} walked the dog right to the girl's front door.", "{hero} arriving at a door with the shaggy dog on a lead"),
            ("{hero} learned that caring for others feels better than anything.", "{hero} watching the girl and dog reunite with happy tears"),
        ],
    ],
    "lost and found": [
        [   # arc A: the missing story book
            ("{hero}'s most treasured storybook had vanished — completely, utterly gone. {hero} checked under the bed, behind the sofa cushions, and inside every drawer twice. The whole morning felt grey without it.", "{hero} searching through a pile of scattered books with a worried, determined expression"),
            ("It wasn't in the kitchen, the garden shed, or the coat-pocket where lost things often hid. {hero} stood in the middle of the room with hands on hips, thinking hard. 'Where,' said {hero} very firmly, 'did you go?'", "{hero} standing hands-on-hips in the middle of the room, looking puzzled"),
            ("{hero} closed both eyes and retraced every single step of the day before. Morning — the bench by the door. Then the walk. Then... the library. {hero}'s eyes flew open. The library!", "{hero} eyes suddenly wide with an idea, finger pointing up"),
            ("Sure enough, there it was — sitting on a library shelf between two books that didn't belong to it. But it wasn't alone. A child was holding it carefully, almost at the last page. {hero} stopped and watched.", "{hero} spotting the beloved book on a library shelf, a child sitting nearby reading it"),
            ("{hero} waited until the child finished the last page, and the child's face went soft and quiet the way it does at the end of a perfect story. {hero} smiled and sat down beside the child. 'That's my favourite book,' {hero} said. 'I know all the best bits by heart.' And they read it again, together.", "{hero} and a new friend sitting side-by-side at the library, reading the book together"),
            ("{hero} walked home with the book tucked under one arm and a warm feeling tucked inside their chest. Some things are better when they're shared — stories most of all. {hero} had lost a book and found a friend, which was a much better trade.", "{hero} walking home smiling with the book under their arm, waving goodbye to the new friend"),
        ],
        [   # arc B: the lost recipe
            ("Granny had lost her secret recipe for the best cookies ever.", "{hero} watching granny search through messy recipe cards"),
            ("{hero} promised to help find it before the big bake.", "{hero} putting on an apron and looking determined"),
            ("They searched the kitchen, the attic, and every drawer.", "{hero} and granny checking every corner together"),
            ("Inside an old hat box — a folded, faded piece of paper!", "{hero} gasping and holding up a crumpled piece of paper"),
            ("The recipe was there: flour, love, and a pinch of laughter.", "{hero} and granny reading the recipe together with big smiles"),
            ("{hero} learned that the best things are worth looking for.", "{hero} and granny pulling cookies from the oven together"),
        ],
        [   # arc C: lost in the market
            ("{hero} looked up and all the grown-ups had disappeared.", "{hero} standing alone in a busy colourful market"),
            ("The market seemed bigger and louder than ever before.", "{hero} turning around looking worried among tall market stalls"),
            ("{hero} remembered: stay calm, find someone who helps.", "{hero} taking a deep breath with eyes closed"),
            ("{hero} found a stall keeper with a kind face and asked for help.", "{hero} tapping a friendly shop-keeper on the shoulder"),
            ("The family came running — laughing with relief.", "{hero} being scooped up in a big family hug"),
            ("{hero} learned that staying calm helps you find the way home.", "{hero} and family walking home hand in hand"),
        ],
    ],
    "learning something new": [
        [   # arc A: first time on a bicycle
            ("{hero} had wanted to ride the bicycle since the very first day of summer. But every single time, the wheels wobbled and {hero} tumbled sideways onto the grass. By the fourth fall, {hero}'s knees were scraped and the bicycle looked very pleased with itself.", "{hero} sitting on the grass beside the fallen bicycle, looking frustrated but determined"),
            ("Every try ended the same way — a moment of hope, then a wobble, then the ground. {hero} sat up, brushed off the mud, and stared at the bicycle. The bicycle, {hero} decided, was not winning today.", "{hero} wobbling dramatically on the bicycle, trying hard not to fall"),
            ("Papa crouched beside {hero} and said something that didn't make sense at first. 'Don't look at the ground. Look at where you want to go.' {hero} frowned at the gate at the end of the path. It felt very far away. But {hero} nodded.", "{hero} listening carefully as Papa points towards the garden gate ahead"),
            ("{hero} climbed back on, fixed both eyes on the gate, and pedalled. The wheels still wobbled — but just a little. {hero} kept looking forward. The wobble got smaller. The gate got closer.", "{hero} cycling carefully with eyes fixed ahead, wheels wobbling less and less"),
            ("Zoom! {hero} shot past the gate with a shout that startled every bird in the garden. The bicycle was going fast and straight and perfectly. {hero} had done it — had actually, genuinely, completely done it.", "{hero} flying past the garden gate on the bicycle, arms raised in triumph"),
            ("{hero} rode up and down the lane until the sun went low. That evening, {hero} understood something new: looking forward isn't just how you ride a bicycle. It's how you do almost everything hard. {hero} couldn't wait to see what else was possible.", "{hero} cycling confidently down a golden sunny lane, a huge proud smile on their face"),
        ],
        [   # arc B: cooking for the first time
            ("{hero} decided to make soup all by themselves.", "{hero} standing on a step stool by the big kitchen pot"),
            ("The first batch was way too salty — even the cat sneezed.", "{hero} and a sneezing cat looking at a steaming pot"),
            ("{hero} didn't give up — just added less salt next time.", "{hero} carefully adding a tiny pinch of salt to the pot"),
            ("The second batch smelled amazing — warm and rich.", "{hero} inhaling deeply with a big happy smile"),
            ("The whole family asked for a second bowl.", "{hero} ladling soup into happy family bowls"),
            ("{hero} learned that mistakes are just steps on the way to great.", "{hero} sitting proudly at the head of the dinner table"),
        ],
        [   # arc C: learning to read music
            ("{hero} wanted to play piano but the notes looked like dots and sticks.", "{hero} staring at a sheet of music with a puzzled frown"),
            ("Teacher showed that each line on the page had its own note.", "{hero} and a teacher pointing at a music book together"),
            ("{hero} practised one tiny tune — just five notes — every day.", "{hero} pressing piano keys slowly and carefully"),
            ("After a week, those five notes sounded like a real song!", "{hero} playing piano with eyes closed, smiling"),
            ("At the recital, {hero} played for everyone — hands barely shaking.", "{hero} performing on stage in front of a small happy crowd"),
            ("{hero} learned that big skills start with five tiny notes.", "{hero} taking a bow as the crowd claps"),
        ],
    ],
    "kindness to animals": [
        [   # arc A: the sparrow with a broken wing
            ("{hero} found a small sparrow under the garden hedge, one wing folded oddly. The bird looked up with big trusting eyes, and {hero}'s heart squeezed with worry. Very carefully, {hero} scooped the sparrow into both warm palms.", "{hero} kneeling under a garden hedge, gently cupping a small sparrow with a drooping wing"),
            ("{hero} made the sparrow a cosy nest from a shoebox, a soft scarf, and a little bowl of water. The sparrow settled in slowly, its feathers ruffling then flattening in a sign of calm. {hero} watched over it all afternoon like a tiny doctor.", "{hero} leaning over a shoebox nest watching the sparrow settle in, a small bowl of water beside it"),
            ("Every morning, {hero} brought the sparrow berries and tiny seeds, counting the days. On the ninth morning, the wing looked different — puffed out and strong. {hero}'s hands trembled as the lid of the box came off.", "{hero} holding open the shoebox in morning light, the sparrow inside looking alert and ready"),
            ("The sparrow hopped to the edge of the box and blinked. Then it flew — just a short flutter to the window ledge — and {hero} gasped with pure delight. 'You did it!' {hero} whispered.", "{hero} with wide joyful eyes watching the sparrow flutter to a sunlit window ledge"),
            ("Day by day the sparrow's flights grew longer, until one afternoon it didn't come back. {hero} stood in the garden for a long time, looking at the bright empty sky. The feeling was sad and proud and right all at once.", "{hero} standing alone in the garden, looking up at a bright clear sky"),
            ("That evening, soft twittering came from the oak tree — the sparrow, singing. {hero} smiled and understood: helping someone doesn't mean keeping them. {hero} had learned that the kindest love lets go.", "{hero} smiling up at a sparrow singing in a golden sunlit oak tree"),
        ],
        [   # arc B: the cold kitten
            ("{hero} heard a tiny mew from under the porch steps on a cold rainy night.", "{hero} kneeling by the porch steps, ear to the ground, listening"),
            ("A small wet kitten was shivering there, eyes barely open.", "{hero} discovering a tiny shivering kitten under the steps"),
            ("{hero} wrapped the kitten in an old soft towel and held it very close.", "{hero} cradling a towel-wrapped kitten against their chest in the lamplight"),
            ("The kitten's trembling slowly stopped — and then it purred for the very first time.", "{hero} feeling the kitten purr, an expression of wonder and happiness"),
            ("{hero} fed it warm milk from a tiny spoon, drop by drop, all through the night.", "{hero} gently spooning warm milk to the kitten under a lamp"),
            ("By morning the kitten was playing with {hero}'s shoelace — and life was completely different.", "{hero} and the kitten playing happily in morning sunlight"),
        ],
        [   # arc C: the turtle on the path
            ("In the park behind the houses, {hero} spotted a turtle sitting in the middle of the path.", "{hero} stopping in surprise, spotting a turtle in the centre of a busy path"),
            ("Cars could come — the turtle was in real danger and didn't know it.", "{hero} looking worried, pointing at the turtle on the path"),
            ("{hero} knelt down and lifted the turtle very, very slowly and gently.", "{hero} carefully lifting a turtle with both hands, moving very slowly"),
            ("The turtle tucked its head inside its shell — but then peeked one eye out at {hero}.", "{hero} smiling as the turtle peeks one cautious eye from its shell"),
            ("{hero} carried it all the way to the pond's edge and set it softly in the reeds.", "{hero} gently lowering the turtle to the water's edge among green reeds"),
            ("It slid into the water and swam away strong and free. {hero} had learned that every life matters — even a very slow-moving one.", "{hero} watching the turtle swim away into the glinting pond with a proud smile"),
        ],
    ],
    "the magic of imagination": [
        [   # arc A: the chalk spaceship
            ("{hero} had a big box of chalk, an empty pavement, and a whole sunny Saturday. {hero} drew a long silver rocket with the word HERO painted on the side in bold yellow letters. Then the strangest thing happened — the chalk rocket shimmered like it was breathing.", "{hero} crouched over a colourful chalk spaceship drawing on the pavement, eyes wide with wonder"),
            ("{hero} pressed one hand flat on the rocket's hatch — and with a whoooosh that nobody else could hear, {hero} was inside. The walls were silver, the windows were perfectly round, and the stars outside were very, very close. {hero}'s heart pounded with pure happiness.", "{hero} inside a gleaming silver rocket cockpit with round porthole windows full of bright stars"),
            ("The rocket glided past ringed planets and fizzing green nebulas. {hero} steered with a big joystick and called out every planet's name — and then spotted one nobody had ever named. It glowed pink and had three wobbling moons.", "{hero} at the rocket controls grinning, with ringed planets and a glowing pink unknown planet outside"),
            ("On the pink planet, everything was made of soft, springy cloud. {hero} jumped and bounced and did a wobbly somersault and landed laughing. Three moon-creatures with gentle faces waved all four of their arms.", "{hero} bouncing joyfully on a cloud surface while three friendly moon-creatures wave"),
            ("Too soon, the rocket beeped: FUEL LOW. {hero} waved goodbye to the moon-creatures — who waved back with every single arm — and aimed the rocket toward the pale blue dot that was home.", "{hero} waving out the porthole at the moon-creatures as the rocket turns toward a small blue planet"),
            ("The whoooosh reversed and {hero} was on the pavement again, chalk dust on both knees. The drawing looked ordinary — but {hero} knew imagination isn't in the pavement. It's in whoever holds the chalk.", "{hero} sitting back on the pavement looking at the chalk drawing, smiling with dusty knees"),
        ],
        [   # arc B: the painting that moved
            ("{hero} painted a jungle on a big white paper using every colour in the box.", "{hero} painting an enormous colourful jungle with a big brush, tongue out in concentration"),
            ("When {hero} looked away and looked back, a painted parrot had definitely moved.", "{hero} staring at the painting with a surprised expression, the parrot in a different spot"),
            ("{hero} stepped through the paper — the air inside smelled of flowers and warm bark.", "{hero} stepping into a lush painted jungle, eyes wide with wonder"),
            ("The painted animals talked and played music on instruments made of leaves.", "{hero} dancing with painted animals playing leaf-drums and twig-flutes"),
            ("The parrot showed {hero} the only way back: a painted door at the edge of the world.", "{hero} and the parrot standing before a bright painted door in the jungle"),
            ("{hero} stepped through, picked up the brush again, and knew that art is always a door.", "{hero} back in the real world holding the paintbrush, smiling at the painting"),
        ],
        [   # arc C: the cardboard kingdom
            ("{hero} found twelve big cardboard boxes and had an enormous, unstoppable idea.", "{hero} surrounded by cardboard boxes with a huge grin and a marker pen"),
            ("The boxes became towers, a drawbridge, a throne, and a very impressive cardboard dragon.", "{hero} building an elaborate cardboard castle with towers and a dragon"),
            ("{hero} was King and General and Chef of the Cardboard Kingdom all at once.", "{hero} wearing a cardboard crown, ruling from a cardboard throne"),
            ("The cardboard dragon came alive and roared — it needed a brave and worthy friend.", "{hero} facing the cardboard dragon bravely, both looking at each other with interest"),
            ("{hero} and the dragon ruled fairly together, sharing cookies with every subject.", "{hero} and the dragon handing out cookies to a line of stuffed animals"),
            ("When the boxes got soggy in the rain, the kingdom lived on — inside {hero}'s mind forever.", "{hero} looking at the damp collapsed boxes with a proud, happy smile"),
        ],
    ],
    "celebrating who you are": [
        [   # arc A: the one who was different
            ("In the whole classroom, {hero} did everything differently — coloured outside the lines, loved books more than balls, and always had mud on one shoe. Some days {hero} wished hard to be exactly like everyone else. Most days, though, the mud was the best part.", "{hero} sitting slightly apart at school, colouring enthusiastically outside the lines"),
            ("One afternoon, the art teacher set a challenge: paint something nobody else could paint. {hero} looked at the blank paper for a quiet moment, then began. The brush moved in swirling loops and a map-like pattern that only {hero} fully understood.", "{hero} painting an enormous swirling colourful map-pattern on a big piece of paper, looking absorbed"),
            ("When the paintings were displayed, everyone kept stopping at {hero}'s. 'What is it?' they asked. {hero} explained: it was the path home, drawn in the language of memory and feeling. The room went very quiet.", "{hero} standing beside their large painting, with other children gathered around it looking interested"),
            ("A boy who always seemed perfectly ordinary said, 'Could you teach me?' Nobody had ever asked {hero} to teach anyone anything before. {hero} felt something warm light up behind the ribs.", "{hero} and the boy sitting together with brushes, looking at the colourful painting"),
            ("From that week on, the classroom had more mud, more unusual loops, and more paintings that asked 'what does this mean?' {hero} kept being exactly the same {hero} — and the whole room had changed around that.", "{hero} in a colourful classroom full of different unusual paintings, everyone comparing their work"),
            ("{hero} had learned something important: you don't need to change who you are to be the most interesting person in the room. Being exactly, completely, wonderfully yourself is enough — and might be exactly what the world needs.", "{hero} holding up their painting proudly in the school corridor, smiling in afternoon light"),
        ],
        [   # arc B: the tallest in class
            ("{hero} was always in the back row of every class photo — the tallest by far.", "{hero} standing head-and-shoulders above everyone else in a class photo"),
            ("{hero} bumped on doorways and could never hide in hide-and-seek.", "{hero} ducking through a doorway, knees bent, trying to be smaller"),
            ("One day the library ladder broke — and only {hero} could reach the top shelf.", "{hero} easily reaching the highest library shelf with one stretched arm"),
            ("{hero} found the lost book the whole class had been waiting weeks to read.", "{hero} pulling a dusty book from the highest shelf, looking triumphant"),
            ("Everyone cheered and {hero} laughed — the biggest, loudest laugh in the whole room.", "{hero} laughing with the class, their laugh the biggest of all"),
            ("{hero} learned that what sometimes feels too much is very often exactly enough.", "{hero} walking tall and proud down the school corridor, head held high"),
        ],
        [   # arc C: the quiet one
            ("In a loud and laughing classroom, {hero} was always quiet — watching, listening, noticing.", "{hero} sitting calmly while a noisy classroom swirls around them"),
            ("When a strange crackling sound came from the kitchen, only {hero} heard it.", "{hero} looking up sharply, alert, while everyone else stays busy"),
            ("{hero} quietly told the teacher: 'Something is burning in the kitchen.'", "{hero} raising one calm hand and speaking softly to the teacher"),
            ("It was just toast — caught quickly because {hero} had been paying attention.", "A teacher pulling smoke-free toast from the toaster, looking relieved"),
            ("The whole class said: 'Good thing {hero} was listening!'", "The whole class turning to look at {hero} with grateful, impressed faces"),
            ("{hero} smiled and understood: being quiet is not the same as being invisible. Quiet is its own kind of power.", "{hero} smiling serenely amid a cheerful classroom, feeling completely at home"),
        ],
    ],
    "a rainy day adventure": [
        [   # arc A: the indoor expedition
            ("Rain drummed on the windows and {hero}'s big outdoor plans were cancelled. {hero} pressed both palms against the cold glass and watched the street turn into a silver river. The day stretched ahead, long and damp — or so it seemed.", "{hero} pressing hands to a rain-streaked window, watching puddles form outside"),
            ("Then {hero} had a thought: what if inside was a country nobody had ever properly explored? {hero} pulled on the rain hat and the old adventure boots and declared the hallway the Valley of the Long Rug. The journey had begun.", "{hero} wearing a rain hat and boots at the start of a long hallway, looking determined"),
            ("The sofa became Mount Cushion — shaky to climb, magnificent from the top. The kitchen table became a cave for thinking. The bookshelf was Shelf-Everest and {hero} read aloud to an imaginary expedition team.", "{hero} sitting triumphantly on top of a pile of sofa cushions, arms raised like a mountain climber"),
            ("{hero} drew a map as the journey went: squiggly lines, starred locations, important notes like 'biscuit found here' and 'very good echo.' By afternoon the map covered four pages and had a legend with six entries.", "{hero} drawing a large map on multiple sheets of paper spread across the floor"),
            ("The rain slowed to a drizzle. {hero} stood at the window again and looked out at the wet, sparkling garden. It had been the best day — not despite the rain, but because of it.", "{hero} looking out at a clearing rain-washed garden with a warm, contented smile"),
            ("{hero} pinned the map to the bedroom wall, where it would stay forever. The Valley of the Long Rug was right there at the top with a red X where the biscuit was found. {hero} had learned: adventure begins wherever you decide it does.", "{hero} pinning a colourful hand-drawn map to the bedroom wall, looking proud"),
        ],
        [   # arc B: the puddle scientist
            ("{hero} wasn't allowed out in the heavy rain — but was allowed to watch through the window.", "{hero} pressing nose to the window glass watching rain fall outside"),
            ("{hero} got a notebook and began recording every puddle forming on the garden path.", "{hero} sitting by the window sketching puddle shapes in a notebook"),
            ("When the rain stopped, {hero} raced out with a ruler, a magnifying glass, and big boots.", "{hero} stomping outside in big boots carrying a ruler and magnifying glass"),
            ("The puddles held entire worlds: a floating leaf, a tiny snail, a rainbow swirl in a patch of oil.", "{hero} kneeling beside a puddle, magnifying glass held over a rainbow reflection"),
            ("{hero} sketched everything carefully and wrote at the top: 'Rainy Day Study — Field Notes.'", "{hero} writing in the notebook surrounded by puddle sketches and measurements"),
            ("{hero} learned that science was not far away at all — it was right outside the front door.", "{hero} closing the notebook with satisfaction, boots muddy, face glowing"),
        ],
        [   # arc C: the baking storm
            ("All the cousins were stuck indoors at Grandma's house because of a big thunderstorm.", "A group of children looking out at stormy rain from a cosy kitchen window"),
            ("Grandma pulled out flour, butter, eggs, and a mysterious tin of something wonderful.", "{hero} and the cousins gathering around Grandma as she opens the baking cupboard"),
            ("{hero} was made head baker — the only one patient enough to measure carefully.", "{hero} very seriously measuring flour with a large spoon while cousins watch"),
            ("The kitchen filled with warm smell and accidental flour clouds that made everyone sneeze.", "{hero} and Grandma laughing in a cloud of flour, the bowl half-mixed"),
            ("By the time the thunder faded, six perfect golden scones sat cooling on the rack.", "{hero} and cousins staring at perfectly golden scones with wide, hungry eyes"),
            ("Storms, {hero} decided, should probably happen more often. Grandma agreed.", "{hero} and Grandma sharing a warm scone at the kitchen table, rain soft outside"),
        ],
    ],
}

FEW_SHOT_EXEMPLAR = """
Write a 6-page children's storybook for age 5 about Finn the red fox with theme: overcoming a fear.

Rules:
- Finn must appear by name in EVERY page text.
- No new characters introduced after page 3.
- Each page is 2–3 sentences — vivid, warm, and emotionally engaging.
- Use sensory details: colours, sounds, textures, and feelings.
- Pages 1–2 introduce the character and problem. Pages 3–4 build the challenge. Pages 5–6 resolve and teach.
- Page 6 ends with a warm lesson Finn has learned and a feeling of pride.
- The scene describes only what can be drawn in one illustration.
- Return ONLY valid JSON (no extra text).

{
  "title": "Finn and the Thunderstorm",
  "character_description": "A small red fox named Finn with big amber eyes, white-tipped ears, and a fluffy striped tail",
  "pages": [
    {"page": 1, "text": "Finn the red fox loved sunny mornings more than anything in the world, but the very first rumble of thunder made Finn's whole body tremble. Whenever dark clouds rolled in, Finn would crawl under the sofa and squeeze both eyes tightly shut. Finn wanted to be brave — but brave seemed like something meant for bigger foxes.", "scene": "Finn the red fox curled tightly under a blue sofa, eyes shut, while dark storm clouds fill the window behind the curtains"},
    {"page": 2, "text": "One stormy afternoon, thunder shook the whole cottage and rattled the teacups on the shelf. Finn buried deep under the blankets, heart going thump-thump-thump. Grandma Fox came quietly and sat on the edge of the bed without saying a word.", "scene": "Finn the red fox burrowed under a patchwork blanket on a bed, Grandma Fox sitting calmly beside in warm lamplight"},
    {"page": 3, "text": "Grandma Fox leaned in close and whispered a secret into Finn's ear. 'Thunder is just two clouds bumping into each other and saying sorry,' she said softly. Finn's ears perked up, and one eye opened just a little.", "scene": "Grandma Fox leaning close to whisper to Finn the red fox peeking out from under the blanket"},
    {"page": 4, "text": "Finn closed both eyes again and tried very hard to imagine it — two big fluffy clouds bumping heads and making an oops face. A tiny giggle bubbled up before Finn could stop it. The next rumble came, and this time it sounded almost funny instead of frightening.", "scene": "Finn the red fox grinning with closed eyes, imagining two round cartoon clouds bumping together with surprised faces"},
    {"page": 5, "text": "The next crack of thunder boomed across the sky — and Finn laughed out loud. Grandma Fox laughed too, and together they ran to the window to watch the lightning flash and dance. The storm looked completely different now, like a wild and wonderful show put on just for them.", "scene": "Finn the red fox and Grandma Fox laughing at a rain-streaked window as bright lightning flashes outside in the dark sky"},
    {"page": 6, "text": "When the storm finally passed, Finn stood in the doorway and breathed in the clean, rain-washed air. Grandma Fox held Finn's paw quietly — some lessons don't need extra words. Finn had learned that understanding why something happens is the very first step to not being afraid of it anymore.", "scene": "Finn the red fox and Grandma Fox standing in the open doorway watching a golden rainbow stretch across the sky after the storm"}
  ]
}
"""


def build_story_prompt(hero_name: str, theme: str, age: int,
                       num_pages: int = 6, story_idea: str = "") -> str:
    mid = num_pages - 2
    spark_line = (f"\n- Story spark from the child: {story_idea.strip()}" if story_idea and story_idea.strip() else "")
    return f"""{FEW_SHOT_EXEMPLAR}

Write a {num_pages}-page children's storybook for age {age} about {hero_name} with theme: {theme}.

Rules:
- {hero_name} must appear by name in EVERY page text. Every single page.
- Keep all characters consistent — do NOT introduce random new characters after page 3.
- Each page: 2–3 vivid, emotionally warm sentences a {age}-year-old can follow and feel.
- Use sensory details — colours, sounds, textures, emotions — to bring each moment alive.
- Pages 1–2 introduce {hero_name} and the challenge. Pages 3–{mid} deepen the adventure. Pages {mid+1}–{num_pages} resolve it and teach.
- Page {num_pages} ends with a clear, warm lesson {hero_name} has learned and a feeling of pride or joy.
- Scene describes exactly what would appear in ONE illustration.{spark_line}
- Return ONLY valid JSON (no explanation, no markdown fences):
"""


def _validate_story_structure(story: dict) -> bool:
    required_keys = ["title", "character_description", "pages"]
    if not all(k in story for k in required_keys):
        return False
    pages = story.get("pages", [])
    if not isinstance(pages, list) or len(pages) < 1:
        return False
    first_page = pages[0]
    return all(k in first_page for k in ["page", "text", "scene"])


def _repair_json(json_str: str) -> str:
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
    json_str = re.sub(r'/\*[\s\S]*?\*/', '', json_str)
    json_str = re.sub(r'(?<=")\n(?=")', '\\n', json_str)
    json_str = re.sub(r'(\s)(\w+)(\s*:)', r'\1"\2"\3', json_str)
    return json_str


def parse_story_json(raw_output: str) -> dict | None:
    match = re.search(r'\{[\s\S]*\}', raw_output or "")
    if not match:
        return None
    raw_json = match.group(0)
    for candidate in (raw_json, _repair_json(raw_json)):
        try:
            story = json.loads(candidate)
            if _validate_story_structure(story):
                return story
        except Exception:
            continue
    return None


def _normalize_story(story: dict, num_pages: int = 6) -> dict:
    pages = list(story.get("pages", []))[:num_pages]
    while len(pages) < num_pages:
        pages.append({
            "page": len(pages) + 1,
            "text": "And the adventure continued happily.",
            "scene": "Continuing adventure",
        })
    story["pages"] = pages
    story.setdefault("title", "A Wonderful Adventure")
    story.setdefault(
        "character_description",
        "A friendly children's storybook hero with bright colors and cheerful features",
    )
    return story


def build_story_locally(hero_name: str, theme: str) -> dict:
    """Fast fallback — picks a random arc variant so kids get a different story each run."""
    hero = (hero_name or "Little Hero").strip() or "Little Hero"
    arcs = THEME_TEMPLATES.get(theme, THEME_TEMPLATES["brave adventure"])
    beats = _random.choice(arcs)   # one of 3 arc variants, chosen at random
    pages = [
        {"page": i + 1, "text": text.format(hero=hero), "scene": scene.format(hero=hero)}
        for i, (text, scene) in enumerate(beats)
    ]
    arc_titles = {
        "brave adventure": [f"{hero}'s Big Brave Day", f"{hero} and the Broken Bridge", f"{hero}'s Flying Kite"],
        "making a new friend": [f"{hero}'s First Hello", f"{hero} and the Laughing Robot", f"{hero} and the Quiet Cat"],
        "overcoming a fear": [f"{hero} and the Thunder Secret", f"{hero} Takes the Plunge", f"{hero} and the Big Dog"],
        "helping someone": [f"{hero} and the Broken Toys", f"{hero}'s Secret Garden", f"{hero} Finds the Way Home"],
        "lost and found": [f"{hero}'s Missing Book", f"{hero} Finds Granny's Recipe", f"{hero} and the Big Market"],
        "learning something new": [f"{hero} Rides!", f"{hero}'s Salty Soup", f"{hero}'s Five Notes"],
    }
    arc_index = THEME_TEMPLATES.get(theme, THEME_TEMPLATES["brave adventure"]).index(beats)
    titles = arc_titles.get(theme, [f"{hero}'s Storybook Adventure"])
    title = titles[arc_index] if arc_index < len(titles) else f"{hero}'s Storybook Adventure"
    return {
        "title": title,
        "character_description": (
            f"{hero}, a friendly children's storybook hero with bright colours, "
            "bold outlines, and a cheerful expressive face"
        ),
        "pages": pages,
    }


def silent_wav_bytes(duration_seconds: int = 2, sample_rate: int = 24000) -> bytes:
    """Return a short silent WAV so the UI remains stable if TTS is unavailable."""
    num_samples = sample_rate * duration_seconds
    data_size = num_samples * 2
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b"data", data_size,
    )
    return header + (b"\x00" * data_size)


def _with_heartbeat(blocking_fn, frame_fn, poll=4.0):
    import threading

    box = {}

    def _run():
        try:
            box["val"] = blocking_fn()
        except BaseException as e:
            box["err"] = e

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    t0 = time.time()
    while th.is_alive():
        th.join(timeout=poll)
        if th.is_alive():
            yield ("hb", frame_fn(int(time.time() - t0)))
    if "err" in box:
        raise box["err"]
    yield ("done", box["val"])


# ============================================================================
# SAMPLE BOOK (loads instantly, no GPU needed)
# ============================================================================

SAMPLE_BOOK_HTML = None

def load_sample_book() -> str:
    """Load pre-generated sample book (C3: always ship sample)."""
    global SAMPLE_BOOK_HTML
    if SAMPLE_BOOK_HTML:
        return SAMPLE_BOOK_HTML
    
    sample_path = os.path.join(SAMPLE_BOOK_PATH, "sample.html")
    if os.path.exists(sample_path):
        with open(sample_path, "r", encoding="utf-8") as f:
            SAMPLE_BOOK_HTML = f.read()
        return SAMPLE_BOOK_HTML
    
    return "<div class='page-loading'>Loading sample book...</div>"


# ============================================================================
# ZEROGPU INFERENCE FUNCTIONS
# ============================================================================

@spaces.GPU(duration=90)
def generate_story_gpu(hero_name: str, theme: str, age: int = 5,
                       num_pages: int = 6, story_idea: str = "") -> dict:
    """Generate a story on ZeroGPU, falling back to a deterministic local story."""
    try:
        model, tok = load_story()
        prompt = build_story_prompt(hero_name, theme, age, num_pages, story_idea)
        inputs = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=200 * num_pages,
                do_sample=True,
                temperature=0.85,
                top_p=0.92,
                repetition_penalty=1.1,
            )
        response = tok.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        parsed = parse_story_json(response)
        if parsed:
            return _normalize_story(parsed, num_pages)
        logger.warning("Story parser failed; using random-arc local fallback")
    except Exception as e:
        logger.warning(f"ZeroGPU story generation failed: {e}")
    return _normalize_story(build_story_locally(hero_name, theme), num_pages)


@spaces.GPU(duration=200)
def generate_images_gpu(
    character_desc: str,
    scenes: list,
    doodle_bytes: bytes = None,
    seed: int = 42,
) -> list:
    """Generate all story pages with FLUX on ZeroGPU (two-stage: canonical
    character from the doodle, then the same character in each scene)."""
    import io
    from PIL import Image

    pipe = load_flux()
    num_steps, guidance = 6, 1.0

    canonical = None
    if doodle_bytes:
        try:
            ref = Image.open(io.BytesIO(doodle_bytes)).convert("RGB")
            canonical = pipe(
                prompt=(f"Turn this child's drawing into a clean, friendly, full-body cartoon "
                        f"character for a children's storybook. Keep the EXACT same creature, "
                        f"face, and features as the drawing. {COLOR_ART_STYLE}, "
                        f"plain white background, full character visible, centered."),
                image=ref, height=768, width=768, guidance_scale=guidance,
                num_inference_steps=num_steps,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]
            logger.info("Canonical character built from doodle")
        except Exception as e:
            logger.warning(f"Canonical build failed ({e}); text2img fallback")
            canonical = None

    images = []
    for i, scene in enumerate(scenes):
        if canonical is not None:
            prompt = f"The same character. {scene}. {COLOR_ART_STYLE}, {COLOR_PAGE_SUFFIX}"
            kw = dict(image=canonical, prompt=prompt)
        else:
            prompt = (f"{character_desc}. Scene: {scene}. {COLOR_ART_STYLE}, "
                      f"white background, centered, full character visible")
            kw = dict(prompt=prompt)
        kw.update(height=768, width=768, guidance_scale=guidance,
                  num_inference_steps=num_steps,
                  generator=torch.Generator("cuda").manual_seed(seed + i + 1))
        images.append(pipe(**kw).images[0])
        logger.info(f"Generated page {i+1}/{len(scenes)}")
    return images


@spaces.GPU(duration=150)
def generate_coloring_images_gpu(color_pngs: list, seed: int = 7) -> list:
    """Coloring pages = FLUX redraws each finished COLOR page as clean line art
    (img2img). This MATCHES the storybook composition and avoids the speckly
    look of tracing crayon texture. Caller crispens the result to black-on-white."""
    import io
    from PIL import Image

    pipe = load_flux()
    prompt = f"{LINE_ART_STYLE}, {LINE_ART_SUFFIX}"
    outs = []
    for i, png in enumerate(color_pngs):
        ref = Image.open(io.BytesIO(png)).convert("RGB")
        base = dict(prompt=prompt, image=ref, height=768, width=768,
                    guidance_scale=1.0, num_inference_steps=6,
                    generator=torch.Generator("cuda").manual_seed(seed + i))
        try:                                  # strength may not be accepted
            img = pipe(**base, strength=0.85).images[0]
        except TypeError:
            img = pipe(**base).images[0]
        outs.append(img)
        logger.info(f"Generated coloring page {i+1}/{len(color_pngs)}")
    return outs


@spaces.GPU(duration=180)
def generate_tts_gpu(text: str, voice: str = DEFAULT_VOICE,
                     ref_wav: str | None = None) -> bytes:
    """Narrate the book with VoxCPM2.
    When voice=='my_voice' and ref_wav is provided, clones the caller's voice.
    Raises on failure so the caller can surface the real reason."""
    import io
    import numpy as np

    try:
        model = load_tts()
        design = voice_design(voice)
        is_cloned = (voice == "my_voice"
                     and bool(ref_wav and os.path.exists(str(ref_wav))))

        chunks = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not chunks:
            chunks = [text.strip() or "The end."]
        if is_cloned:
            chunks = chunks[:15]  # voice cloning ~6-8s/sentence; cap for 180s budget

        sr = model.tts_model.sample_rate
        pause = np.zeros(int(sr * 0.35), dtype=np.float32)
        pieces = []

        for i, sentence in enumerate(chunks):
            kw = dict(text=f"{design} {sentence}",
                      cfg_value=2.0, inference_timesteps=10)
            if is_cloned:
                kw["reference_wav_path"] = ref_wav
            wav = model.generate(**kw)
            pieces.append(np.asarray(wav, dtype=np.float32))
            if i < len(chunks) - 1:
                pieces.append(pause)

        audio = np.concatenate(pieces)
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        return buf.getvalue()

    except Exception:
        logger.exception("TTS failed")
        raise


# ============================================================================
# MAIN BOOK CREATION (Generator for streaming)
# ============================================================================

def create_book(doodle_image, character_name, theme, hero_name,
                voice=DEFAULT_VOICE, make_coloring=False,
                num_pages=6, story_idea=None, custom_voice_wav=None):
    """ZeroGPU book flow: story → images → narration → PDFs → coloring book,
    each a sequential @spaces.GPU call (ZeroGPU has one GPU per request)."""
    t_total = time.perf_counter()
    character_name = (character_name or "").strip() or "Little Hero"
    hero_name = (hero_name or "").strip() or character_name
    num_pages = max(6, min(10, int(num_pages or 6)))

    trace_data = {
        "backend": "zerogpu",
        "hero_name": hero_name,
        "theme": theme,
        "voice": voice,
        "num_pages": num_pages,
        "make_coloring": make_coloring,
        "seed": BASE_SEED,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if _LOAD_ERRORS:
        trace_data["model_load_errors"] = _LOAD_ERRORS
    
    _no = gr.update(visible=False)
    _keep = gr.update()

    yield (
        magic_loader_html("story", hero_name),
        "Writing the story…",
        None, _keep, {}, "", json.dumps(trace_data, indent=2),
        _no, _keep,
    )

    t_story = time.perf_counter()
    try:
        story = generate_story_gpu(hero_name, theme, num_pages=num_pages,
                                   story_idea=story_idea or "")
    except Exception as e:
        logger.error(f"Story generation failed: {e}")
        yield (
            f"<div class='page-loading'>Error: {e}</div>",
            f"Error: {e}",
            None, _keep, {}, "", "",
            _no, _keep,
        )
        return
    trace_data["story_sec"] = round(time.perf_counter() - t_story, 2)

    pages = story.get("pages", [])
    char_desc = story.get("character_description", "")
    title = story.get("title", "Untitled Story")
    page_texts = [p.get("text", "") for p in pages]
    scenes = [p.get("scene", "") for p in pages]
    
    trace_data["title"] = title
    trace_data["character_description"] = char_desc
    
    yield (
        magic_loader_html("images", hero_name),
        f"{title} — illustrating on ZeroGPU…",
        None, _keep, story, "", json.dumps(trace_data, indent=2),
        _no, _keep,
    )

    doodle_bytes = None
    if doodle_image is not None:
        import io
        from PIL import Image
        img = Image.fromarray(doodle_image)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        doodle_bytes = buf.getvalue()

    full_text = f"{title}. {' '.join(page_texts)}"

    # ---- NARRATION starts NOW, in PARALLEL with the images (it only needs the
    #      story text). The audio is surfaced the moment it's ready — usually
    #      before the illustrations finish — so narration appears first. ----
    import threading
    voice_box = {}
    t_tts = time.perf_counter()

    def _do_voice():
        try:
            voice_box["bytes"] = generate_tts_gpu(
                full_text, voice,
                ref_wav=custom_voice_wav if voice == "my_voice" else None,
            )
        except Exception as e:
            voice_box["err"] = e

    voice_thread = threading.Thread(target=_do_voice, daemon=True)
    voice_thread.start()

    def _audio_now():
        """Write the narration to a temp wav once it's ready; return its path."""
        if voice_box.get("bytes") and not voice_box.get("path"):
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(voice_box["bytes"])
                    voice_box["path"] = tmp.name
            except Exception as e:
                logger.warning(f"writing audio failed: {e}")
        return voice_box.get("path")

    # ---- IMAGES (FLUX on ZeroGPU), surfacing the narration as soon as it lands ----
    img_bytes, engine, images = None, "sketch", None
    t_images = time.perf_counter()
    try:
        for kind, payload in _with_heartbeat(
            lambda: generate_images_gpu(char_desc, scenes, doodle_bytes, BASE_SEED),
            lambda s: (
                magic_loader_html("images", hero_name),
                f"{title} — illustrating… {s}s"
                + ("  ·  narration ready ▶" if _audio_now() else "  ·  recording narration…"),
                _audio_now(), _keep, story, "", json.dumps(trace_data, indent=2), _no, _keep,
            ),
        ):
            if kind == "hb":
                yield payload
            else:
                images = payload
        import io
        img_bytes = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes.append(buf.getvalue())
        engine = "flux"
    except Exception as e:
        logger.exception("Image generation failed")
        trace_data["image_error"] = repr(e)
        from services.images import generate_placeholder_images
        img_bytes = generate_placeholder_images(char_desc, scenes, doodle_bytes)
        engine = "sketch"
    trace_data["images_sec"] = round(time.perf_counter() - t_images, 2)
    trace_data["engine"] = engine

    book_html = build_book_html(img_bytes, page_texts, title, engine)

    # ---- collect the parallel narration (usually already finished) ----
    while voice_thread.is_alive():
        voice_thread.join(timeout=4)
        if voice_thread.is_alive():
            yield (book_html, f"{title} — finishing narration…",
                   _audio_now(), _keep, story, "", json.dumps(trace_data, indent=2),
                   _no, _keep)
    audio_path = _audio_now()
    if voice_box.get("err"):
        logger.warning(f"TTS failed: {voice_box['err']}")
        trace_data["tts_error"] = repr(voice_box["err"])
    trace_data["tts_sec"] = round(time.perf_counter() - t_tts, 2)

    pdf_path = None
    t_pdf = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            pdf_path = export_pdf(img_bytes, page_texts, title, tmp.name)
    except Exception as e:
        logger.warning(f"PDF failed: {e}")
    trace_data["pdf_sec"] = round(time.perf_counter() - t_pdf, 2)

    coloring_html = ""
    coloring_pdf_path = None
    if make_coloring:
        t_coloring = time.perf_counter()
        try:
            from services.coloring import _crispen
            for kind, payload in _with_heartbeat(
                lambda: generate_coloring_images_gpu(img_bytes, 7),
                lambda s: (
                    book_html,
                    f"{title} — building coloring book… {s}s",
                    audio_path,
                    _keep,
                    story,
                    "",
                    json.dumps(trace_data, indent=2),
                    _no,
                    _keep,
                ),
            ):
                if kind == "hb":
                    yield payload
                else:
                    coloring_images = payload
            import io
            outlines = []
            for img in coloring_images:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                outlines.append(_crispen(buf.getvalue()))
            coloring_html = build_coloring_html(outlines, page_texts, title)
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                coloring_pdf_path = export_coloring_pdf(outlines, page_texts, title, tmp.name)
            trace_data["coloring_book"] = True
            trace_data["coloring_engine"] = "flux-direct-lineart"
        except Exception as e:
            logger.exception("FLUX line-art coloring failed; using local trace fallback")
            trace_data["coloring_error"] = repr(e)
            try:
                from services.coloring import _to_line_art_opencv
                outlines = [_to_line_art_opencv(b) for b in img_bytes]
                coloring_html = build_coloring_html(outlines, page_texts, title)
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    coloring_pdf_path = export_coloring_pdf(outlines, page_texts, title, tmp.name)
                trace_data["coloring_book"] = True
                trace_data["coloring_engine"] = "opencv-trace-fallback"
            except Exception as e2:
                logger.exception("Coloring fallback also failed")
                trace_data["coloring_error2"] = repr(e2)
        trace_data["coloring_sec"] = round(time.perf_counter() - t_coloring, 2)

    trace_data["completed"] = True
    trace_data["pages_generated"] = len(img_bytes)
    trace_data["total_sec"] = round(time.perf_counter() - t_total, 2)

    # Reveal the download buttons (they start visible=False) — value alone left
    # them hidden, which is why there was "no download option" (incl. on mobile).
    pdf_update = gr.update(value=pdf_path, visible=True) if pdf_path else _keep
    coloring_pdf_update = (gr.update(value=coloring_pdf_path, visible=True)
                           if coloring_pdf_path else _keep)
    coloring_display_update = (gr.update(visible=True, value=coloring_html) if coloring_html
                               else _no)

    yield (
        book_html,
        f"Complete: {title} — {len(img_bytes)} pages · {'FLUX (ZeroGPU)' if engine == 'flux' else 'local sketch fallback'} · voice: {voice} · total {trace_data['total_sec']}s",
        audio_path,
        pdf_update,
        story,
        f"Pages: {len(img_bytes)} | Seed: {BASE_SEED} | Engine: {engine} | Story {trace_data.get('story_sec', 0)}s | Images {trace_data.get('images_sec', 0)}s | PDF {trace_data.get('pdf_sec', 0)}s | Coloring {trace_data.get('coloring_sec', 0)}s",
        json.dumps(trace_data, indent=2),
        coloring_display_update,
        coloring_pdf_update,
    )


if __name__ == "__main__":
    demo = create_layout(
        load_sample_fn=load_sample_book,
        create_book_fn=create_book,
    )
    demo.queue(default_concurrency_limit=2, max_size=8)
    # design_kwargs (theme/css/js/head) is non-empty on gradio 6 (moved to launch)
    demo.launch(share=False, allowed_paths=[tempfile.gettempdir()], **demo.design_kwargs)
