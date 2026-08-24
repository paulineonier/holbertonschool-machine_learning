#!/usr/bin/env python3
"""Interactive Q&A loop module."""


def main():
    """Runs a loop that prompts the user with Q: and responds with A:."""
    exit_words = {"exit", "quit", "goodbye", "bye"}

    while True:
        try:
            user_input = input("Q: ").strip()
            if user_input.lower() in exit_words:
                print("A: Goodbye")
                break
            print("A:")
        except (KeyboardInterrupt, EOFError):
            print("\nA: Goodbye")
            break


if __name__ == "__main__":
    main()