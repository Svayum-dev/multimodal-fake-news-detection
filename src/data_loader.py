
def inspect_news_post(text, image_path, label=None):
    """Display the information available for one news post."""

    print("News text:", text)
    print("Image path:", image_path)
    print("Label:", label)


if __name__ == "__main__":
    inspect_news_post(
        text="A major earthquake occurred in Delhi today.",
        image_path="data/sample_image.jpg",
        label="unverified"
    )
