def compute_engagement_score(hot_ratio, participation_diversity, positive_emotion_pct, discipline_index, teacher_student_balance):
    # all inputs expected 0..1
    score = (
        0.3 * hot_ratio +
        0.25 * participation_diversity +
        0.2 * positive_emotion_pct +
        0.15 * discipline_index +
        0.1 * teacher_student_balance
    )
    # scale to 0-100
    return float(max(0.0, min(100.0, score * 100.0)))


def classify_engagement(score):
    if score >= 70:
        return 'High'
    elif score >= 40:
        return 'Medium'
    else:
        return 'Low'


if __name__ == '__main__':
    print(compute_engagement_score(0.5, 0.6, 0.7, 0.8, 0.4))
