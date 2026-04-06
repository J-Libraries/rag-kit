import re


def clean_response(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def is_bad_rewrite(text: str) -> bool:
    if not text:
        return True

    text = text.strip()

    if len(text.split()) > 20:
        return True

    if "\n" in text:
        return True

    return False