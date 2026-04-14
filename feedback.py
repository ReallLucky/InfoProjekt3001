def calculate_score(features):
    score = 50

    if abs(features["avg_velocity"]) > 0.03:
        score += 30
    else:
        score -= 10

    return max(0, min(100, score))


def generate_feedback(score, features):
    feedback = []

    if score > 80:
        feedback.append("Sehr gut! Dein Schlag ist schnell und kontrolliert 👍")
    elif score > 50:
        feedback.append("Gut, aber du kannst noch mehr Geschwindigkeit aufbauen.")
    else:
        feedback.append("Versuche, den Schlag flüssiger und schneller auszuführen.")

    if abs(features["avg_velocity"]) < 0.02:
        feedback.append("Nutze mehr Hüftrotation für mehr Kraft.")

    return feedback