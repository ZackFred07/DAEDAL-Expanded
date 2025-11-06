def svamp_prompt(doc):
  system_prompt = (
      "You are a math expert. You will be given a question to solve. "
      "Solve it step by step. Wrap the final answer in a \\boxed{}. \n"
      "Respond in the following format:\n"
      "<reasoning>\n"
      "Your reasoning here\n"
      "</reasoning>\n"
      "<answer>\n"
      "\\boxed{...}\n"
      "</answer>"
  )

  # Prefer the pre-concatenated field if present; otherwise join Body + Question.
  q = (
      doc.get("question_concat")
      or " ".join(x for x in [doc.get("Body"), doc.get("Question")] if x)
      or doc.get("question")  # fallback for any alternative mirrors
      or ""
  )

  return f"{system_prompt}\n\n{q}\n\n"
