def mmlu_prompt(doc):
    question = doc.get("question", "")
    choices = doc.get("choices") or []
    subject = doc.get("subject")

    labels = "ABCD"
    option_lines = []
    for i, choice in enumerate(choices):
        lab = labels[i] if i < len(labels) else chr(65 + i)
        option_lines.append(f"{lab}. {choice}")

    subject_line = f"Subject: {subject}\n" if subject else ""

    return (
        "Answer the following multiple-choice question.\n"
        "Respond with a single letter: A, B, C, or D.\n"
        "Do not include any explanation.\n\n"
        f"{subject_line}"
        f"Question: {question}\n"
        "Options:\n"
        + "\n".join(option_lines)
        + "\nAnswer:"
    )
