from content_creator import linkedin_agent, twitter_agent

if __name__ == "__main__":
    topic = input("Enter topic: ")
    
    linkedin_post, linkedin_md, linkedin_img = linkedin_agent(topic)
    print(f"✅ LinkedIn post saved at {linkedin_md}")
    print(f"✅ LinkedIn image saved at {linkedin_img}")
    
    twitter_post, twitter_md = twitter_agent(topic)
    print(f"✅ Twitter post saved at {twitter_md}")
