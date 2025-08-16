from content_creator import linkedin_agent, twitter_agent, research_agent

if __name__ == "__main__":
    print("\n🔍 Researching trending topic for KLYRR.....")
    topic = research_agent()
    print(f"✅ Trending topic found: {topic}\n")

    print("🔹 Generating LinkedIn content...")
    linkedin_post, linkedin_md, linkedin_image = linkedin_agent(topic)
    print(f"✅ LinkedIn post saved: {linkedin_md}")
    print(f"✅ LinkedIn image saved: {linkedin_image}")

    print("\n🔹 Generating Twitter content...")
    twitter_post, twitter_md = twitter_agent(topic)
    print(f"✅ Twitter post saved: {twitter_md}")
