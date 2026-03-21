import random

def inject_typos(text, prob=0.08):
    chars = list(text)
    for i in range(len(chars)-1):
        if random.random() < prob:
            chars[i], chars[i+1] = chars[i+1], chars[i]
    return "".join(chars)