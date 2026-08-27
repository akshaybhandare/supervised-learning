import os
import json
import random

random.seed(42)

# ======================================================================
# 1. RAW KNOWLEDGE CORPUS (For Stage 1: Pre-training Base Model)
# Notice: NO "User:", NO "Assistant:", NO "Q:", NO "A:".
# Just pure continuous paragraphs of human knowledge and facts!
# ======================================================================
PRETRAIN_PARAGRAPHS = [
    # Geography & Travel
    "Paris is the capital city of France, located along the Seine river. Visitors in Paris often stay in boutique hotels overlooking the historic Eiffel Tower and explore the art collections at the Louvre museum.",
    "Tokyo is the bustling capital of Japan, famous for ancient Kyoto temples, high-speed bullet trains, and modern technology districts. Travelers often take bullet trains across Japan to experience sushi and ramen culture.",
    "Rome is the historic capital of Italy, renowned for the ancient Colosseum, the Pantheon, and walking through classical Roman ruins.",
    "Iceland is a volcanic island in the North Atlantic known for dramatic landscapes, glacial waterfalls, and vibrant green Northern Lights appearing in clear night skies.",
    "Chennai is a major coastal city in South India, easily accessible via the Chennai Express railway connecting Southern states.",
    "Bali is a tropical island in Indonesia popular for scenic rice terraces, sacred Hindu temples, and relaxing coastal beach resorts.",
    "Goa is a popular coastal destination in Western India known for sandy beaches, seafood, and historic Portuguese architecture.",
    "Machu Picchu is a fifteenth-century Inca citadel set high in the Andes mountains of Peru along the historic Inca Trail.",
    "The Thar Desert near Jaisalmer in India is famous for golden sand dunes, camel safaris, and overnight camping under desert stars.",
    "Kenya in East Africa offers world-renowned safari wildlife parks where visitors can observe elephants, lions, and giraffes in natural savannah habitats.",

    # Food & Culinary
    "Butter chicken is a classic North Indian curry made with tender chicken simmered in a rich tomato, butter, and cream sauce, traditionally served with warm garlic naan.",
    "Masala dosa is a crispy fermented rice and lentil crepe originating in South India, filled with spiced potatoes and served with coconut chutney and hot sambar.",
    "Pasta carbonara is an authentic Italian pasta dish prepared with spaghetti, fresh egg yolks, aged pecorino cheese, and cured pork without heavy cream.",
    "Sourdough bread is made through natural fermentation of wild yeast, resulting in a crispy golden crust and an open airy crumb structure.",
    "Neapolitan pizza is baked in a wood-fired oven at high heat, topped with San Marzano tomatoes, fresh mozzarella, and fresh green basil leaves.",
    "Tonkotsu ramen is a hearty Japanese noodle soup made by boiling pork bones for hours into a creamy broth, topped with soft-boiled eggs and scallions.",
    "Guacamole is a traditional Mexican dip made by mashing ripe avocados with lime juice, fresh cilantro, diced red onions, and sea salt.",
    "Old Delhi is famous for authentic Indian street food including hot crispy jalebis, spicy aloo tikki, and stuffed parathas.",

    # Health & Fitness
    "Progressive resistance training with barbell squats and bench press builds muscular strength, bone density, and total body power.",
    "A barbell bench press of 225 pounds represents a solid milestone of upper body pressing strength for strength athletes.",
    "Running ten kilometers builds cardiovascular endurance, improves resting heart rate, and strengthens leg muscles when trained consistently.",
    "Human muscle recovery requires seven to eight hours of quality uninterrupted sleep each night to facilitate tissue repair and growth hormone release.",
    "Consuming adequate dietary protein, typically 1.6 to 2.2 grams per kilogram of body weight, optimizes muscle protein synthesis after resistance workouts.",
    "Drinking three liters of clean water daily maintains optimal cellular hydration, aids digestion, and prevents exercise-induced muscle fatigue.",
    "Dynamic stretching and foam rolling after intense exercise help reduce muscle tightness, promote blood flow, and accelerate recovery.",

    # Technology & Engineering
    "FastAPI is a modern high-performance Python web framework used to build RESTful APIs with automatic OpenAPI documentation and asynchronous support.",
    "Docker containers package application code and dependencies into standardized lightweight units that run consistently across development and production environments.",
    "Kubernetes is an open-source container orchestration system that automates deployment, scaling, and operational management of containerized services.",
    "PostgreSQL is a powerful open-source relational database supporting advanced SQL queries, JSON storage, and high transaction concurrency.",
    "Redis is an in-memory data structure store used as a distributed cache and message broker to achieve sub-millisecond database query latency.",
    "Apache Kafka is a distributed event streaming platform used by modern high-throughput architectures for real-time messaging and data pipelines.",
    "Next.js is a React web framework that enables server-side rendering, static site generation, and optimized frontend performance.",

    # Personal Finance
    "The S&P 500 index represents five hundred leading publicly traded American companies and has delivered long-term historical returns around ten percent annually.",
    "A Roth IRA is an individual retirement account in the United States allowing contributions of post-tax money to grow and be withdrawn completely tax-free in retirement.",
    "An emergency fund consisting of three to six months of living expenses should be kept in a liquid high-yield savings account earning competitive interest.",
    "Investing consistently in low-cost index funds minimizes management fees and captures broad market returns over long time horizons.",
    "Treasury bills are short-term United States government debt obligations backed by the full faith and credit of the government, offering secure risk-free yield.",
    "Maintaining a personal savings rate above thirty percent accelerates progress toward financial independence by widening the gap between income and spending."
]

# ======================================================================
# 2. INSTRUCTION / CHAT CONVERSATIONS (For Stage 2: Post-training / SFT)
# Formatted explicitly as conversations: User <-> Assistant
# ======================================================================
SFT_DIALOGUES = [
    # Travel
    ("Where did you stay in Paris?", "I stayed at a boutique hotel with a balcony overlooking the Eiffel Tower."),
    ("What can you see in Tokyo?", "You can see ancient Kyoto temples, ride bullet trains, and explore modern tech districts."),
    ("What are the main sights in Rome?", "The ancient Colosseum, the Pantheon, and historical Roman ruins."),
    ("What makes Iceland unique?", "Iceland is famous for dramatic waterfalls, volcanic landscapes, and the green Northern Lights."),
    ("How do people travel to Chennai?", "Many travelers take the Chennai Express train connecting Southern India."),
    ("What is Bali famous for?", "Bali is famous for scenic rice terraces, Hindu temples, and tropical beach resorts."),
    ("What can you do in Goa?", "Goa is great for relaxing on sandy beaches and exploring Portuguese architecture."),
    ("Where is Machu Picchu located?", "Machu Picchu is set high in the Andes mountains of Peru along the Inca Trail."),
    ("What can you experience in Jaisalmer?", "You can enjoy camel safaris and overnight camping in the Thar Desert."),
    ("What wildlife can you see in Kenya?", "You can observe wild elephants, lions, and giraffes on African savannah safaris."),

    # Food
    ("What is butter chicken served with?", "Butter chicken is traditionally served with warm garlic naan and steamed rice."),
    ("How is masala dosa prepared?", "It is a crispy rice and lentil crepe filled with spiced potatoes, served with coconut chutney."),
    ("How do you make authentic pasta carbonara?", "Use spaghetti with fresh egg yolks, aged pecorino cheese, and cured pork without cream."),
    ("What gives sourdough bread its texture?", "Natural fermentation gives sourdough its golden crispy crust and open airy crumb."),
    ("What are the key toppings for Neapolitan pizza?", "Fresh mozzarella cheese, San Marzano tomatoes, and fresh basil leaves."),
    ("What is in tonkotsu ramen?", "Tonkotsu ramen features rich pork bone broth with noodles, soft-boiled eggs, and scallions."),
    ("How do you make fresh guacamole?", "Mash ripe avocados with lime juice, fresh cilantro, diced red onions, and sea salt."),
    ("What street food is popular in Old Delhi?", "Crispy jalebis, spicy aloo tikki, and stuffed parathas."),

    # Health
    ("Why is progressive resistance training important?", "It builds muscular strength, bone density, and total body power."),
    ("What is a milestone for bench press?", "A bench press of 225 pounds is a classic strength milestone."),
    ("What are the benefits of running ten kilometers?", "It builds cardiovascular endurance, lowers resting heart rate, and strengthens leg muscles."),
    ("How much sleep is needed for muscle recovery?", "Seven to eight hours of quality uninterrupted sleep per night."),
    ("How much protein should an active person eat?", "Around 1.6 to 2.2 grams of protein per kilogram of body weight daily."),
    ("Why should you drink three liters of water daily?", "It maintains cellular hydration, aids digestion, and prevents workout fatigue."),
    ("How do you accelerate post-workout recovery?", "Use dynamic stretching, foam rolling, and adequate sleep to reduce muscle tightness."),

    # Tech
    ("What is FastAPI used for?", "FastAPI is a Python web framework used to build high-performance REST APIs."),
    ("What are Docker containers?", "Docker packages application code and dependencies into standardized portable units."),
    ("What does Kubernetes do?", "Kubernetes automates the deployment, scaling, and management of containerized applications."),
    ("Why choose PostgreSQL?", "PostgreSQL is a reliable relational database with great support for complex queries and JSON."),
    ("What is Redis used for?", "Redis is an in-memory database used for caching and low-latency message broker systems."),
    ("What is Apache Kafka?", "Kafka is a distributed event streaming platform used for high-throughput messaging."),
    ("What does Next.js offer?", "Next.js offers server-side rendering and static site generation for React frontends."),

    # Finance
    ("What is the S&P 500?", "The S&P 500 tracks 500 leading US companies with an average annual historical return around ten percent."),
    ("What is a Roth IRA?", "A Roth IRA is an account where post-tax money grows and is withdrawn tax-free in retirement."),
    ("How much should be in an emergency fund?", "Three to six months of essential living expenses kept in a high-yield savings account."),
    ("Why invest in low-cost index funds?", "Index funds minimize management fees and provide broad, diversified market exposure."),
    ("What are US Treasury bills?", "Treasury bills are short-term government debt securities backed by the full faith of the government."),
    ("How does a high savings rate help?", "A savings rate above thirty percent accelerates progress toward financial independence.")
]


def create_all_datasets():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # 1. Write Pretraining Corpus (Raw text paragraphs)
    pretrain_file = os.path.join(data_dir, "pretrain_corpus.txt")
    with open(pretrain_file, "w") as f:
        for p in PRETRAIN_PARAGRAPHS:
            f.write(p.strip() + "\n")
    print(f"✓ Created Stage 1 Pre-training Corpus: {len(PRETRAIN_PARAGRAPHS)} raw knowledge paragraphs in '{pretrain_file}'")

    # 2. Write Post-training / SFT Dialogues (Chat format)
    sft_file = os.path.join(data_dir, "sft_dialogues.jsonl")
    variations = ["", "Hey, ", "Can you tell me, ", "Quick question: ", "Please explain, "]
    dialogues = []
    for _ in range(30):
        for q, a in SFT_DIALOGUES:
            v = random.choice(variations)
            user_msg = v + q
            formatted = f"<bos> User: {user_msg} Assistant: {a} <eos>"
            dialogues.append({
                "prompt": f"<bos> User: {user_msg} Assistant:",
                "response": f" {a} <eos>",
                "full_text": formatted
            })

    with open(sft_file, "w") as f:
        for d in dialogues:
            f.write(json.dumps(d) + "\n")
    print(f"✓ Created Stage 2 Post-training SFT Dataset: {len(dialogues)} conversation pairs in '{sft_file}'")


if __name__ == "__main__":
    create_all_datasets()
