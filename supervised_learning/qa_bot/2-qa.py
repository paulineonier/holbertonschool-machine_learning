#!/usr/bin/env python3
"""Interactive Q&A loop using BERT Question Answering."""
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """Answers questions from a reference text in an interactive loop.

    Args:
        reference (str): The reference document containing answers.
    """
    exit_words = {"exit", "quit", "goodbye", "bye"}

    while True:
        try:
            user_input = input("Q: ").strip()

            if user_input.lower() in exit_words:
                print("A: Goodbye")
                break

            if not user_input:
                continue

            answer = question_answer(user_input, reference)

            if answer is None:
                print("A: Sorry, I do not understand your question.")
            else:
                print(f"A: {answer}")

        except (KeyboardInterrupt, EOFError):
            print("\nA: Goodbye")
            break