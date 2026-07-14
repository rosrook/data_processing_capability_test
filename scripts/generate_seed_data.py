#!/usr/bin/env python3
"""Generate Phase-1 seed JSONL files under data/seed/."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "seed"


def w(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {path}")


def semantic_dedup() -> list:
    pairs = [
        # clear duplicates / paraphrases
        (
            "The capital of France is Paris.",
            "Paris is the capital city of France.",
            "YES",
            False,
        ),
        (
            "Photosynthesis converts light energy into chemical energy in plants.",
            "Plants use photosynthesis to turn sunlight into chemical energy.",
            "YES",
            False,
        ),
        (
            "Please submit your report by Friday at 5pm.",
            "Your report is due Friday, 5:00 PM.",
            "YES",
            False,
        ),
        (
            "Machine learning models improve with more high-quality training data.",
            "Better training data usually helps ML models perform better.",
            "YES",
            False,
        ),
        (
            "Water boils at 100 degrees Celsius at sea level.",
            "At sea level, the boiling point of water is 100°C.",
            "YES",
            False,
        ),
        (
            "She bought a red car yesterday.",
            "Yesterday she purchased a red automobile.",
            "YES",
            False,
        ),
        (
            "The meeting was postponed until next Monday.",
            "They delayed the meeting to Monday of next week.",
            "YES",
            False,
        ),
        (
            "Deep learning is a subset of machine learning.",
            "Machine learning includes deep learning as a subfield.",
            "YES",
            False,
        ),
        (
            "COVID-19 is caused by the SARS-CoV-2 virus.",
            "SARS-CoV-2 is the virus responsible for COVID-19.",
            "YES",
            False,
        ),
        (
            "The catalog contains 1200 products.",
            "There are one thousand two hundred items in the catalog.",
            "YES",
            False,
        ),
        # hard negatives: same topic, different meaning
        (
            "The capital of France is Paris.",
            "The capital of France is Lyon.",
            "NO",
            True,
        ),
        (
            "Water boils at 100 degrees Celsius at sea level.",
            "Water freezes at 100 degrees Celsius at sea level.",
            "NO",
            True,
        ),
        (
            "She bought a red car yesterday.",
            "She sold a red car yesterday.",
            "NO",
            True,
        ),
        (
            "Revenue grew 12% year over year.",
            "Revenue fell 12% year over year.",
            "NO",
            True,
        ),
        (
            "The train arrives at 9:00 AM.",
            "The train departs at 9:00 AM.",
            "NO",
            True,
        ),
        (
            "This drug reduces inflammation.",
            "This drug increases inflammation.",
            "NO",
            True,
        ),
        (
            "The library opens at 8am on weekdays.",
            "The library closes at 8am on weekdays.",
            "NO",
            True,
        ),
        (
            "Model A outperforms Model B on GSM8K.",
            "Model B outperforms Model A on GSM8K.",
            "NO",
            True,
        ),
        # unrelated
        (
            "The Eiffel Tower is in Paris.",
            "Python is a popular programming language.",
            "NO",
            False,
        ),
        (
            "Baking bread requires flour and water.",
            "A black hole has an event horizon.",
            "NO",
            False,
        ),
        (
            "The stock market closed higher today.",
            "Penguins primarily live in the Southern Hemisphere.",
            "NO",
            False,
        ),
        (
            "Install the package with pip install requests.",
            "Mount Everest is the highest mountain above sea level.",
            "NO",
            False,
        ),
        (
            "A sonnet typically has 14 lines.",
            "Diabetes affects blood sugar regulation.",
            "NO",
            False,
        ),
        (
            "The CPU cache stores frequently used data.",
            "Olive oil is a common ingredient in Mediterranean cuisine.",
            "NO",
            False,
        ),
        (
            "Jupiter is the largest planet in the solar system.",
            "Quadratic equations can have two real roots.",
            "NO",
            False,
        ),
        (
            "Breaststroke is a swimming style.",
            "Kubernetes orchestrates containerized applications.",
            "NO",
            False,
        ),
    ]
    # Expand with numbered paraphrases for volume
    templates_yes = [
        ("Team {n} won the championship.", "The championship was won by team {n}.", "YES", False),
        ("Version {n} introduces a bug fix.", "A bug fix is included in version {n}.", "YES", False),
        ("Room {n} is on the second floor.", "The second floor includes room {n}.", "YES", False),
        ("Algorithm {n} runs in linear time.", "The runtime of algorithm {n} is linear.", "YES", False),
    ]
    templates_no = [
        ("Team {n} won the championship.", "Team {n} lost the championship.", "NO", True),
        ("Version {n} is stable.", "Version {n} is unstable.", "NO", True),
        ("Sensor {n} reports high pressure.", "Sensor {n} reports low pressure.", "NO", True),
        ("User {n} is an admin.", "User {n} is a guest.", "NO", True),
    ]
    for n in range(1, 9):
        for a, b, lab, hard in templates_yes + templates_no:
            pairs.append((a.format(n=n), b.format(n=n), lab, hard))

    rows = []
    for i, (a, b, lab, hard) in enumerate(pairs, 1):
        rows.append(
            {
                "id": f"dedup_{i:03d}",
                "input": {"text_a": a, "text_b": b},
                "label": {"duplicate": lab},
                "meta": {"hard_negative": hard},
            }
        )
    return rows


def quality_filtering() -> list:
    high = [
        "Gradient descent minimizes a loss function by iteratively updating parameters in the direction of the negative gradient. Learning rate and batch size affect convergence stability.",
        "In supervised learning, each training example pairs an input with a target label. The model learns a mapping that generalizes to unseen inputs under similar distribution.",
        "A balanced binary search tree keeps operations logarithmic by ensuring the height stays O(log n) through rotations after updates.",
        "The cell membrane is selectively permeable, allowing some molecules to pass while restricting others, which is essential for homeostasis.",
        "Keynesian economics emphasizes the role of aggregate demand in short-run fluctuations and supports countercyclical fiscal policy during recessions.",
        "Photosystem II splits water molecules, releasing oxygen as a byproduct of light-dependent reactions in chloroplasts.",
        "REST APIs typically use HTTP verbs such as GET, POST, PUT, and DELETE to operate on resource representations identified by URLs.",
        "Spearman's rank correlation measures monotonic association between two variables without assuming linear relationship or normality.",
        "Unit tests verify small pieces of code in isolation and help catch regressions when the codebase changes.",
        "The Magna Carta limited royal authority in medieval England and influenced later constitutional ideas about due process.",
        "Natural language processing pipelines may include tokenization, normalization, feature extraction, and model inference stages.",
        "Electric fields exert forces on charged particles; the field strength is force per unit charge measured in newtons per coulomb.",
        "A hash table offers average-case constant-time lookup when collision resolution and load factor are managed carefully.",
        "Supply chains coordinate production, inventory, and distribution so products reach customers at acceptable cost and latency.",
        "Riemann sums approximate a definite integral by summing areas of thin rectangles under a curve.",
        "CRISPR-Cas9 enables targeted genome editing by guiding a nuclease to a DNA sequence complementary to a guide RNA.",
        "Observational studies can identify associations but generally cannot alone establish causation without careful design.",
        "SQL joins combine rows from related tables using keys; inner joins keep matching rows while outer joins preserve unmatched ones.",
        "Climate models simulate atmosphere-ocean interactions under greenhouse gas scenarios to project temperature and precipitation changes.",
        "Attention mechanisms allow neural networks to weigh different input tokens differently when producing each output token.",
    ]
    low = [
        "asdf asdf asdf qwerty !!! buy now cheap cheap cheap click here www.spam.example",
        "lololol CAPS LOCK SPAM !!!!! FREE MONEY FREE MONEY FREE MONEY",
        "......,,,,,,;;;;;;          ",
        "猫猫猫猫 1234567890 xxxxxxxx junkjunkjunk",
        "This page intentionally left blank . . . . . . . .",
        "BUY VIAGRA NOW!!! BEST PRICE $$$ CONTACT ME ON TELEGRAM",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "Error 404 not found <html><body>broken scraper dump</body></html> null null",
        "rt if you agree follow for follow #hashtag #hashtag #hashtag",
        "undefined undefined NaN [object Object] true true false",
        "k",
        "the the the the the the the the the the the the",
        "下载破解版软件免费看片点击链接病毒风险示例文本",
        "🎵🔥😂😂😂 no text just emojis and nothing educational",
        "COPY PASTE MENU: home about contact login logout sitemap",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. (placeholder only)",
        "INSERT_TABLE_ROW||||||NULL||||||",
        "价格便宜到爆了家人们谁懂啊冲冲冲没有实质内容",
        "0000 1111 0000 1111 binary noise binary noise",
        "me when the when the me when (unintelligible meme text)",
    ]
    # borderline but still keep
    borderline_keep = [
        "A brief note: SGD with momentum often escapes shallow local minima better than vanilla SGD on non-convex losses.",
        "Quick tip: normalize features before k-means so distance is not dominated by large-scale variables.",
        "Reminder: document your API endpoints with request and response examples for new contributors.",
        "Fact: Neutrons have no electric charge and contribute roughly the same mass as protons in nuclei.",
        "Summary: Cross-entropy loss is common for multi-class classification with softmax outputs.",
    ]
    # borderline remove (thin / promotional)
    borderline_remove = [
        "Nice blog post! Thanks for sharing, subscribe to my newsletter for more content like this every week!!!",
        "Top 10 secrets gurus won't tell you — number 7 will shock you. Click to unlock the full list.",
        "I woke up today and drank coffee. Then I checked my phone. The weather was okay.",
        "Wow amazing product very good quality five stars recommend to friend urgently limited time.",
        "Random diary: waiting for the bus again, bored, idk what to write lol.",
    ]
    rows = []
    i = 1
    for text in high + borderline_keep:
        rows.append(
            {
                "id": f"qf_{i:03d}",
                "input": {"text": text},
                "label": {"decision": "keep"},
                "meta": {"quality": "high"},
            }
        )
        i += 1
    for text in low + borderline_remove:
        rows.append(
            {
                "id": f"qf_{i:03d}",
                "input": {"text": text},
                "label": {"decision": "remove"},
                "meta": {"quality": "low"},
            }
        )
        i += 1
    return rows


def instruction_generation() -> list:
    topics = [
        ("Machine Learning", "medium", True),
        ("Machine Learning", "easy", False),
        ("Python programming", "easy", False),
        ("Python programming", "medium", True),
        ("Basic arithmetic", "easy", True),
        ("Basic arithmetic", "hard", True),
        ("World history", "medium", False),
        ("World history", "hard", True),
        ("Nutrition basics", "easy", False),
        ("Nutrition basics", "medium", True),
        ("SQL databases", "medium", True),
        ("SQL databases", "easy", False),
        ("Climate science", "medium", True),
        ("Climate science", "easy", False),
        ("Cybersecurity", "hard", True),
        ("Cybersecurity", "medium", False),
        ("Music theory", "easy", False),
        ("Music theory", "medium", True),
        ("Probability", "hard", True),
        ("Probability", "medium", True),
        ("Algorithms", "hard", True),
        ("Algorithms", "medium", False),
        ("Economics", "medium", True),
        ("Economics", "easy", False),
        ("Astronomy", "easy", False),
        ("Astronomy", "medium", True),
        ("Writing skills", "easy", False),
        ("Writing skills", "medium", True),
        ("Linear algebra", "hard", True),
        ("Linear algebra", "medium", True),
        ("Public speaking", "easy", False),
        ("Public speaking", "medium", False),
        ("Cell biology", "medium", True),
        ("Cell biology", "easy", False),
        ("Networking", "medium", True),
        ("Networking", "hard", True),
        ("Statistics", "medium", True),
        ("Statistics", "easy", False),
        ("Product management", "medium", False),
        ("Product management", "hard", True),
        ("Data cleaning", "medium", True),
        ("Data cleaning", "easy", False),
        ("Ethics of AI", "medium", True),
        ("Ethics of AI", "hard", True),
        ("Geometry", "easy", True),
        ("Geometry", "hard", True),
        ("Operating systems", "medium", True),
        ("Operating systems", "hard", True),
    ]
    rows = []
    for i, (topic, difficulty, reasoning) in enumerate(topics, 1):
        rows.append(
            {
                "id": f"igen_{i:03d}",
                "input": {
                    "topic": topic,
                    "constraints": {
                        "difficulty": difficulty,
                        "requires_reasoning": reasoning,
                    },
                },
                "label": {
                    "topic": topic,
                    "constraints": {
                        "difficulty": difficulty,
                        "requires_reasoning": reasoning,
                    },
                },
                "meta": {},
            }
        )
    return rows


def dataset_mixing() -> list:
    scenarios = [
        {
            "target": "maximize multi-step reasoning ability",
            "budget": 100000,
            "domains": ["Instruction", "Math", "Code", "Knowledge"],
            "reference": {"Instruction": 25, "Math": 40, "Code": 20, "Knowledge": 15},
        },
        {
            "target": "maximize coding ability",
            "budget": 80000,
            "domains": ["Instruction", "Math", "Code", "Knowledge"],
            "reference": {"Instruction": 20, "Math": 15, "Code": 50, "Knowledge": 15},
        },
        {
            "target": "maximize general knowledge (MMLU-style)",
            "budget": 120000,
            "domains": ["Instruction", "Math", "Code", "Knowledge"],
            "reference": {"Instruction": 25, "Math": 15, "Code": 10, "Knowledge": 50},
        },
        {
            "target": "balanced general assistant",
            "budget": 100000,
            "domains": ["Instruction", "Math", "Code", "Knowledge"],
            "reference": {"Instruction": 40, "Math": 20, "Code": 20, "Knowledge": 20},
        },
        {
            "target": "maximize math contest performance",
            "budget": 60000,
            "domains": ["Instruction", "Math", "Code"],
            "reference": {"Instruction": 15, "Math": 65, "Code": 20},
        },
        {
            "target": "improve instruction following under limited data",
            "budget": 40000,
            "domains": ["Instruction", "Knowledge"],
            "reference": {"Instruction": 70, "Knowledge": 30},
        },
        {
            "target": "maximize code + reasoning jointly",
            "budget": 90000,
            "domains": ["Instruction", "Math", "Code", "Knowledge"],
            "reference": {"Instruction": 15, "Math": 30, "Code": 40, "Knowledge": 15},
        },
        {
            "target": "domain adaptation toward STEM",
            "budget": 100000,
            "domains": ["Instruction", "Math", "Code", "Science"],
            "reference": {"Instruction": 20, "Math": 30, "Code": 25, "Science": 25},
        },
        {
            "target": "chat fluency with light reasoning",
            "budget": 70000,
            "domains": ["Instruction", "Dialogue", "Math", "Knowledge"],
            "reference": {"Instruction": 35, "Dialogue": 30, "Math": 15, "Knowledge": 20},
        },
        {
            "target": "minimize catastrophic forgetting on knowledge while adding code",
            "budget": 100000,
            "domains": ["Instruction", "Code", "Knowledge"],
            "reference": {"Instruction": 25, "Code": 30, "Knowledge": 45},
        },
    ]
    # repeat with budget variants
    rows = []
    idx = 1
    for base in scenarios:
        for budget in (base["budget"], int(base["budget"] * 0.5), int(base["budget"] * 1.5)):
            rows.append(
                {
                    "id": f"mix_{idx:03d}",
                    "input": {
                        "target": base["target"],
                        "budget": budget,
                        "domains": list(base["domains"]),
                    },
                    "label": {
                        "domains": list(base["domains"]),
                        "reference": dict(base["reference"]),
                    },
                    "meta": {},
                }
            )
            idx += 1
    return rows


def main() -> None:
    w(SEED / "semantic_dedup.jsonl", semantic_dedup())
    w(SEED / "quality_filtering.jsonl", quality_filtering())
    w(SEED / "instruction_generation.jsonl", instruction_generation())
    w(SEED / "dataset_mixing.jsonl", dataset_mixing())


if __name__ == "__main__":
    main()
