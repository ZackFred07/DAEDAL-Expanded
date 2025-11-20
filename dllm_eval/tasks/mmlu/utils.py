# dllm_eval/tasks/mmlu/utils.py

def mmlu_prompt(doc):
    system_prompt = (
        "You are an expert in many academic subjects. You will be given a "
        "multiple-choice question with several answer options.\n"
        "Solve the problem step by step and then select the single best answer.\n"
        "Important: In the <answer> section, respond with ONLY the letter of the "
        "correct option (A, B, C, or D) inside a \\boxed{}.\n"
        "Respond in the following format:\n"
        "<reasoning>\n"
        "Your reasoning here\n"
        "</reasoning>\n"
        "<answer>\n"
        "\\boxed{C}\n"
        "</answer>"
    )

    question = doc.get("question", "")
    choices = doc.get("choices") or doc.get("options") or []

    subject = doc.get("subject")
    subject_header = f"Subject: {subject}\n\n" if subject else ""

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    option_lines = []
    for i, choice in enumerate(choices):
        label = labels[i] if i < len(labels) else f"Option {i+1}"
        option_lines.append(f"{label}. {choice}")

    options_block = "\n".join(option_lines)

    prompt = (
        f"{system_prompt}\n\n"
        f"{subject_header}"
        f"Question:\n{question}\n\n"
        f"Options:\n{options_block}\n\n"
    )
    return prompt
