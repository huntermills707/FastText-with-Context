import random
import os

# Configuration
authors = ["alice", "bob", "charlie", "diana", "eve", "frank", "grace", "john"]
domains = ["tech", "finance", "health", "politics", "science", "sports", "arts"]
years = [2020, 2021, 2022, 2023, 2024]
sentiments = ["positive", "neutral", "negative"]
locations = ["NYC", "London", "Tokyo", "Berlin", "Paris", "Sydney"]

# Sentence templates to ensure variety
templates = [
    "The {topic} market is showing {trend} signs today.",
    "{author} argues that {technology} will change the future of {industry}.",
    "Recent studies in {location} suggest a {trend} in {topic} adoption.",
    "The {event} highlighted the importance of {topic} in modern {industry}.",
    "Experts predict a {trend} shift in {topic} by {year}.",
    "New regulations in {location} affect how {topic} is managed.",
    "The {sentiment} response to the {event} was unexpected.",
    "Data from {year} shows a clear {trend} in {topic} metrics.",
    "{author} believes that {technology} is the key to solving {problem}.",
    "The intersection of {topic} and {industry} is creating new opportunities.",
    "Critics argue that the current {topic} strategy is {sentiment}.",
    "In {location}, the {topic} sector is experiencing rapid growth.",
    "The report released by {author} details the {trend} of {topic}.",
    "Advances in {technology} are driving changes in {industry}.",
    "Public opinion on {topic} has shifted {sentiment} since {year}.",
    "The {event} served as a turning point for {topic} enthusiasts.",
    "Analysts warn of potential risks in the {topic} market.",
    "Innovation in {technology} is reshaping the landscape of {industry}.",
    "The debate over {topic} continues to divide experts in {location}.",
    "A new study reveals {trend} patterns in {topic} consumption.",
]

topics = ["AI", "crypto", "renewable energy", "quantum computing", "biotech", "blockchain", "cloud", "cybersecurity"]
technologies = ["neural networks", "smart contracts", "CRISPR", "5G", "IoT", "edge computing", "LLMs"]
industries = ["healthcare", "banking", "retail", "manufacturing", "education", "logistics"]
events = ["summit", "conference", "election", "crisis", "launch", "report"]
problems = ["climate change", "inequality", "security breaches", "data privacy", "efficiency"]
trends = ["upward", "downward", "volatile", "stable", "exponential", "linear"]

def generate_training_file(filename="training_data_with_context.txt", num_rows=1_000_000):

    with open(filename, "w", encoding="utf-8") as f:
        for i in range(num_rows):
            # Select random metadata
            author = random.choice(authors)
            domain = random.choice(domains)
            year = random.choice(years)
            sentiment = random.choice(sentiments)
            location = random.choice(locations)
            
            # Select random sentence components
            template = random.choice(templates)
            topic = random.choice(topics)
            tech = random.choice(technologies)
            industry = random.choice(industries)
            event = random.choice(events)
            problem = random.choice(problems)
            trend = random.choice(trends)
            
            # Fill template
            sentence = template.format(
                author=author.capitalize(),
                domain=domain,
                year=year,
                sentiment=sentiment,
                location=location,
                topic=topic,
                technology=tech,
                industry=industry,
                event=event,
                problem=problem,
                trend=trend
            )
            
            # Construct the row: context1|||context2|||sentence
            n = random.randint(0, 3)
            elements = random.sample([author, year, location], n)
            ctx1 = " ".join(map(str, elements))
            n = random.randint(0, 2)
            elements = random.sample([domain, sentiment], n)
            ctx2 = " ".join(map(str, elements))
            row = " ||| ".join([ctx1, ctx2, sentence])
            f.write(row + "\n")

    print(f"Generated {num_rows} rows in '{filename}'")

if __name__ == "__main__":
    generate_training_file()
