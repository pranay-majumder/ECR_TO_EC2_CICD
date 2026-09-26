import re
import string
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger("text_processing")

lemmatizer = WordNetLemmatizer()

# Load English stop words
stop_words = set(stopwords.words("english"))

# Keep negation words because they are important for sentiment analysis
stop_words -= {"not", "no", "never", "neither", "nor"}


def lower_case(text):
    return " ".join(word.lower() for word in text.split())


def remove_stop_words(text):
    return " ".join(
        word for word in text.split()
        if word not in stop_words
    )


def removing_numbers(text):
    return "".join(
        char for char in text
        if not char.isdigit()
    )


def removing_punctuations(text):
    text = re.sub(
        f"[{re.escape(string.punctuation)}]",
        " ",
        text
    )
    text = text.replace("؛", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def removing_urls(text):
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub("", text)


def lemmatization(text):
    return " ".join(
        lemmatizer.lemmatize(word)
        for word in text.split()
    )


def normalize_text(text):
    try:
        text = lower_case(text)
        text = remove_stop_words(text)
        text = removing_numbers(text)
        text = removing_punctuations(text)
        text = removing_urls(text)
        text = lemmatization(text)

        logger.debug("Text normalization completed")
        return text

    except Exception as e:
        logger.error("Error during text normalization: %s", e)
        raise