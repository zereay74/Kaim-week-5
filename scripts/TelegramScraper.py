from telethon import TelegramClient
import csv
import os
from dotenv import load_dotenv
import asyncio
import nest_asyncio
from datetime import datetime

# Allow nested event loops (e.g., for Jupyter Notebook)
nest_asyncio.apply()

class TelegramScraper:
    def __init__(self, api_id, api_hash, phone_number, session_name='scraping_session'):
        """
        Initializes the Telegram client and scraper configuration.

        :param api_id: Telegram API ID
        :param api_hash: Telegram API Hash
        :param phone_number: Phone number for authentication
        :param session_name: Name of the session
        """
        self.client = TelegramClient(session_name, api_id, api_hash)
        self.phone_number = phone_number

    async def scrape_channel(self, channel_username, writer, media_dir):
        """
        Scrapes messages and media from a single Telegram channel.

        :param channel_username: Username of the Telegram channel
        :param writer: CSV writer object
        :param media_dir: Directory to save downloaded media
        """
        print(f"[{datetime.now()}] Starting to scrape channel: {channel_username}")
        try:
            # Get channel entity
            entity = await self.client.get_entity(channel_username)
            channel_title = entity.title
            print(f"[{datetime.now()}] Connected to channel: {channel_title}")

            # Iterate over messages
            message_count = 0
            async for message in self.client.iter_messages(entity, limit=70000):
                media_path = None
                if message.media and hasattr(message.media, 'photo'):
                    # Save photos with unique filenames
                    filename = f"{channel_username}_{message.id}.jpg"
                    media_path = os.path.join(media_dir, filename)
                    await self.client.download_media(message.media, media_path)
                    print(f"[{datetime.now()}] Downloaded media: {filename}")

                # Write message data to the CSV file
                writer.writerow([
                    channel_title, 
                    channel_username, 
                    message.id, 
                    message.message, 
                    message.date, 
                    media_path
                ])
                message_count += 1

            print(f"[{datetime.now()}] Finished scraping {channel_username}: {message_count} messages processed.")
        except Exception as e:
            print(f"[{datetime.now()}] Error scraping channel {channel_username}: {e}")

    async def scrape_channels(self, channels, csv_file, media_dir):
        """
        Scrapes data from multiple channels and saves it to a CSV file.

        :param channels: List of Telegram channel usernames
        :param csv_file: Path to the CSV file
        :param media_dir: Directory to save downloaded media
        """
        os.makedirs(media_dir, exist_ok=True)

        with open(csv_file, 'w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow(['Channel Title', 'Channel Username', 'ID', 'Message', 'Date', 'Media Path'])

            for channel in channels:
                print(f"[{datetime.now()}] Starting processing for channel: {channel}")
                await self.scrape_channel(channel, writer, media_dir)
                print(f"[{datetime.now()}] Completed processing for channel: {channel}")

    async def run(self, channels, csv_file, media_dir):
        """
        Runs the scraper for the specified channels, CSV file, and media directory.

        :param channels: List of Telegram channel usernames
        :param csv_file: Path to the CSV file
        :param media_dir: Directory to save downloaded media
        """
        async with self.client:
            await self.client.start(phone=self.phone_number)
            start_time = datetime.now()
            print(f"[{start_time}] Scraping started.")
            await self.scrape_channels(channels, csv_file, media_dir)
            end_time = datetime.now()
            print(f"[{end_time}] Scraping completed. Duration: {end_time - start_time}")

'''
if __name__ == '__main__':
    # Load environment variables
    load_dotenv()
    api_id = os.getenv('TG_API_ID')
    api_hash = os.getenv('TG_API_HASH')
    phone_number = os.getenv('PHONE_NUMBER')

    # Define input parameters
    channels = ['@Shageronlinestore', '@ExampleChannel1', '@ExampleChannel2']  # Add your target channels here
    csv_file = 'telegram_data.csv'  # Path to save the CSV file
    media_dir = 'photos'  # Directory for media files

    # Initialize the scraper
    scraper = TelegramScraper(api_id, api_hash, phone_number)

    # Run the scraper
    try:
        asyncio.run(scraper.run(channels, csv_file, media_dir))
    except RuntimeError:
        # Handle active event loop (e.g., in Jupyter Notebook)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(scraper.run(channels, csv_file, media_dir))
''' 