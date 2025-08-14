from content_creator import create_posts_with_image

if __name__ == "__main__":
    topic = input("Enter topic: ")
    linkedin_caption, linkedin_image_path, twitter_post = create_posts_with_image(topic)

    print("\n✅ LinkedIn post saved to linkedin_post.md")
    print("✅ Twitter post saved to twitter_post.md")
    print(f"✅ LinkedIn image saved at: {linkedin_image_path}")
