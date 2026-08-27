import json
import random

random.seed(42)

QA_SEEDS = [
    # Travel
    ("Where did you travel in Europe?", "I took a train from Paris to Amsterdam and explored the canals."),
    ("Where did you stay in Paris?", "At a boutique hotel with a balcony overlooking the Eiffel Tower."),
    ("How did you travel to Chennai?", "I took the Chennai Express train for a quick weekend trip."),
    ("What did you see in Iceland?", "The Northern Lights appeared at night and we drove past frozen waterfalls."),
    ("What did you do in Japan?", "Explored temples in Kyoto and took the high-speed bullet train to Tokyo."),
    ("Where did you go for a beach vacation?", "Took a flight to Goa for a relaxing beach holiday."),
    ("Where did you stay in Costa Rica?", "Spent a night in a treehouse hotel in the Costa Rican rainforest."),
    ("What was your road trip route?", "Drove along Route 66 across America with friends."),
    ("What did you visit in Rome?", "Visited the Colosseum and walked around ancient historic ruins."),
    ("How was the flight to London?", "Flew into London Heathrow and took the tube to central London."),
    ("What did you do in Bali?", "Rented a scooter in Bali and visited rice terraces and beaches."),
    ("Where did you go for a desert safari?", "Camped under the stars in the Thar Desert near Jaisalmer."),
    ("How did you travel from London to Paris?", "Took the Eurostar high-speed train in under three hours."),
    ("What did you see in Kenya?", "Went on a safari in Kenya and saw elephants and lions in the wild."),
    ("Where did you hike in Peru?", "Hiked the Inca Trail to Machu Picchu with breathtaking views."),
    
    # Food
    ("What did you cook for Sunday dinner?", "Homemade butter chicken with garlic naan for the whole family."),
    ("What did you have for breakfast?", "Masala dosa with fresh coconut chutney and hot filter coffee."),
    ("What is your favorite Italian pasta?", "Homemade pasta carbonara with pecorino cheese and egg yolks."),
    ("What street food did you try in Delhi?", "Crispy jalebis, aloo tikki, and parathas in Old Delhi."),
    ("How do you bake sourdough bread?", "Baked fresh sourdough with a crispy crust and soft airy crumb."),
    ("What did you order at the Indian restaurant?", "Authentic chicken biryani with spicy salan and cooling raita."),
    ("What did you make on the barbecue?", "Grilled marinated chicken tikka and paneer skewers over hot charcoal."),
    ("What did you have for lunch?", "Meal prepped grilled chicken breast with brown rice and steamed broccoli."),
    ("What did you eat at the Japanese restaurant?", "A hot bowl of tonkotsu ramen with soft-boiled egg and fresh sashimi."),
    ("What coffee did you brew this morning?", "Fresh pour-over coffee using Ethiopian single-origin medium roast beans."),
    ("What pizza did you make?", "Wood-fired Neapolitan pizza with fresh mozzarella and fresh basil leaves."),
    ("How do you make fresh guacamole?", "Mashed ripe avocados with lime juice, fresh cilantro, and diced red onions."),
    ("What dessert did you bake?", "Warm chocolate chip cookies with gooey centers and sea salt flakes."),
    ("What soup did you simmer?", "Homemade chicken bone broth with fresh ginger, garlic, and scallions."),
    
    # Health & Fitness
    ("What is your bench press record?", "Hit a new personal record of 225 lbs for 5 clean reps."),
    ("What is your barbell squat record?", "Squatted 315 lbs for 3 solid reps with full depth."),
    ("What was your morning run distance?", "Completed a 10 km morning run in 52 minutes at a steady pace."),
    ("How much water do you drink daily?", "I drink at least 3 liters of water daily to stay hydrated."),
    ("What is your deadlift record?", "Deadlifted 405 lbs with solid form and a neutral back."),
    ("How many hours do you sleep?", "I prioritize 8 hours of quality sleep every night for muscle recovery."),
    ("What workout did you do today?", "Did a 45-minute high-intensity interval training HIIT session at the gym."),
    ("What supplements do you take?", "Taking 5g of Creatine Monohydrate daily to support strength and power."),
    ("What is your post-workout stretch routine?", "15 minutes of dynamic stretching targeting hamstrings and hips."),
    ("What is your daily protein target?", "Hitting 160g of protein daily from clean whole food sources."),
    ("What yoga routine do you practice?", "60 minutes of Vinyasa flow yoga focusing on mobility and balance."),
    ("How do you set your circadian rhythm?", "Morning walk in outdoor sunlight for 30 minutes right after waking up."),
    
    # Technology
    ("What framework did you use for the API?", "Built a high-performance REST API using Python and FastAPI."),
    ("How is the application deployed?", "Deployed as Docker containers on a local Kubernetes cluster."),
    ("What database are you using?", "Upgraded from MySQL to PostgreSQL for better performance and reliability."),
    ("What programming languages do you use?", "I write backend services in Python, Go, and TypeScript."),
    ("How do you automate testing and deployment?", "Wrote automated CI/CD pipelines using GitHub Actions."),
    ("How do you monitor server metrics?", "Using Prometheus and Grafana dashboards for real-time monitoring."),
    ("How do you speed up database queries?", "Configured a Redis caching layer to reduce query latency to 5ms."),
    ("What reverse proxy do you configure?", "Configured Nginx reverse proxy with SSL certificates from Let's Encrypt."),
    ("How do you handle background message queues?", "Built an event-driven architecture using Apache Kafka message queues."),
    ("What frontend framework are you using?", "Built a modern web frontend with Next.js and React server components."),
    ("How do you secure API endpoints?", "Implemented OAuth2 authentication with secure JWT tokens."),
    
    # Finance
    ("What index fund do you invest in?", "Invested in the low-cost S&P 500 index fund VOO every month."),
    ("Did you max out your Roth IRA?", "Yes, maxed out my annual Roth IRA contribution limit of $7,000."),
    ("Where do you keep your emergency savings?", "In a high-yield savings account earning 4.5% annual interest."),
    ("What is your current savings rate?", "Reached a 40% personal savings rate by cutting unnecessary expenses."),
    ("Are you debt free?", "Paid off my remaining car loan and high-interest debt, officially debt-free."),
    ("How do you allocate your investments?", "Allocated 80% to total stock market index funds and 20% to bonds."),
    ("How do you track monthly cash flow?", "Using an automated spreadsheet to track income, expenses, and savings."),
    ("What short-term investments do you buy?", "Opened an account to buy 4-week US Treasury bills yielding 5.2%."),
    ("How much do you contribute to your 401k?", "Contributed 15% of my salary to maximize the employer company match."),
    ("How did you reduce monthly expenses?", "Audited credit card statements and canceled four unused recurring subscriptions.")
]

QUESTION_PREFIXES = [
    "",
    "Hey, ",
    "Quick question: ",
    "Can you tell me, ",
    "Please tell me: "
]

def generate_qa_dataset(num_samples=1200):
    dataset = []
    for _ in range(num_samples):
        q, a = random.choice(QA_SEEDS)
        prefix = random.choice(QUESTION_PREFIXES)
        clean_q = prefix + q
        # Format: <bos> Q: <question> A: <answer> <eos>
        formatted_text = f"<bos> Q: {clean_q} A: {a} <eos>"
        dataset.append({
            "question": clean_q,
            "answer": a,
            "text": formatted_text
        })
    return dataset

if __name__ == "__main__":
    dataset = generate_qa_dataset(1500)
    with open("qa_data.jsonl", "w") as f:
        for ex in dataset:
            f.write(json.dumps(ex) + "\n")
    print(f"✓ Generated {len(dataset)} Q&A pairs in 'baby_gpt/qa_data.jsonl'")
