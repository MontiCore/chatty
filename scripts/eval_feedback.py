import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import json


if __name__ == "__main__":
    df = pd.read_csv("../data/feedback/feedback-wimis-first.csv")

    ratings = df["rating"].tolist()

    rating_values = {i: 0 for i in range(1, 6)}
    for r in ratings:
        rating_values[r] += 1
    print(rating_values.keys(), rating_values.values())
    plt.barh(rating_values.keys(), rating_values.values(), color=[(1, 0, 0), (1, 1/2, 0), (1, 1, 0), (1/2, 1, 0), (0, 1, 0)])
    plt.xlabel("Count")
    plt.ylabel("Rating")
    plt.savefig("../data/feedback/rating_plot_wimis.svg", format="svg", transparent=True)

    # Export feedback

    comments = df["comment"].dropna().tolist()
    print(len(comments))
    feedback_dict = []
    for c, r in zip(comments, ratings):
        feedback_dict.append({"comment": c, "rating": r})

    with open("../data/feedback/feedback_wimis.json", "w", encoding="utf-8") as f:
        json.dump(feedback_dict, f, ensure_ascii=False, indent=2)
