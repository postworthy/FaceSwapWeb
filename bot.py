import os
import fsw_util
import cv2
import threading
import py7zr
os.environ['OMP_NUM_THREADS'] = '1'
from io import BytesIO
from PIL import Image
from telegram import InputFile
from telegram.ext import ApplicationBuilder, MessageHandler, filters, CommandHandler
from queue import Queue
from fsw_util import push_action, export_as_gif, export_as_jpg, get_face_single, get_face_swapper, process_frames, get_fps, get_face_analyser

THREAD_LOCK_UPLOAD = threading.Lock()

CURRENT_TARGET = "1.jpg"

VIDEO_DIR = './static/videos/'
IMAGE_DIR = './static/images/'

AUTHENTICATED_CHATS = {}

def find_img(target):
    image_path = IMAGE_DIR + str(target)
    if os.path.exists(image_path+".jpg"):
        image_path = os.path.join(image_path+".jpg")
    elif os.path.exists(image_path+".jpeg"):
        image_path = os.path.join(image_path+".jpeg")
    elif os.path.exists(image_path+".png"):
        image_path = os.path.join(image_path+".png")
    else:
        return None
    
    return image_path

def get_base(target):
    image_path = find_img(target)
    if image_path:
        source_face = get_face_single(cv2.imread(image_path))
        print(source_face.gender)
        
        frame = cv2.imread('./static/base.jpg')
        face = get_face_single(frame)
        
        result = get_face_swapper().get(frame, face, source_face, paste_back=True)
        jpg_img = cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)
        image = Image.fromarray(jpg_img)
        
        image_stream = BytesIO()
        image.save(image_stream, format='PNG')
        image_stream.seek(0)

        return image_stream
    else:
        return None
    
def is_authenticated(chat_id):
    return chat_id in AUTHENTICATED_CHATS

async def authenticate(update, context):
    global AUTHENTICATED_CHATS
    if len(context.args) > 0 and context.args[0] == os.environ.get('TELEGRAM_BOT_TOKEN'):
        AUTHENTICATED_CHATS[update.effective_chat.id] = True
        app.add_handler(MessageHandler(filters.Document.ALL & filters.Chat(chat_id=update.effective_chat.id), handle_files))
        app.add_handler(CommandHandler("set_target", set_target, filters.Chat(chat_id=update.effective_chat.id)))
        app.add_handler(CommandHandler("list_features", list_features, filters.Chat(chat_id=update.effective_chat.id)))
        await list_features(update, context)
    else:
        await update.message.reply_text(f"Nope")

async def list_features(update, context):
    features_list = [
        "1. Process files and send processed GIFs",
        "2. Change target using /set_target command"
        # Add more features here as needed
    ]
    features_text = "\n".join(features_list)
    await update.message.reply_text(f"Available features:\n{features_text}")

async def set_target(update, context):
    global CURRENT_TARGET
    if len(context.args) > 0 and context.args[0].isdigit():
        target = int(context.args[0])
        target_path = find_img(target)
        if target_path:
            CURRENT_TARGET = target
            await update.message.reply_text(f"Target changed to {CURRENT_TARGET}")
            processed_image = get_base(target)
            message = update.message
            # Send the processed image as a response
            await context.bot.send_photo(chat_id=message.chat_id, photo=InputFile(processed_image), write_timeout=120)

        else:
            await update.message.reply_text(f"Target {target} does not exist.")
    else:
        await update.message.reply_text(f"Target must be an integer.")

def save_file(downloaded_file_path):
    with THREAD_LOCK_UPLOAD:
        static_file_path = ""
        continue_Processing = False

        with open(downloaded_file_path, 'rb') as downloaded_file:
            downloaded_file_content = downloaded_file.read()

        _, file_extension = os.path.splitext(os.path.basename(downloaded_file_path))
        if file_extension in ['.gif', '.mp4']:
            files = sorted([filename for filename in os.listdir(VIDEO_DIR) if filename.split('.')[0].isdigit()], key=lambda filename: int(filename.split('.')[0]))
            file_id = str(len(files) + 1)
            file_name = file_id + file_extension
            file_path = 'videos/'
            static_file_path = 'static/' + file_path + file_name
            with open(static_file_path, 'wb') as new_file:
                new_file.write(downloaded_file_content)
            continue_Processing = True
        elif file_extension in ['.jpg', '.jpeg', '.png']:
            files = sorted([filename for filename in os.listdir(IMAGE_DIR) if filename.split('.')[0].isdigit()], key=lambda filename: int(filename.split('.')[0]))
            file_id = str(len(files) + 1)
            file_name = file_id + file_extension
            file_path = 'images/'
            static_file_path = 'static/' + file_path + file_name
            with open(static_file_path, 'wb') as new_file:
                new_file.write(downloaded_file_content)
            
        #with py7zr.SevenZipFile(zip_path, 'a', password=zip_pass) as archive:
        #    archive.write(f'static/{file_path}{file_name}', f'{file_path}{file_name}')

    os.remove(downloaded_file_path)
    return static_file_path, continue_Processing

async def handle_files(update, context):
    context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')

    message = update.message
    if message.document and any(extension in message.document.file_name.lower() for extension in ['.mp4', '.gif', '.png', '.jpg', '.jpeg']):
        file = await context.bot.get_file(message.document.file_id)
        file_extension = os.path.splitext(message.document.file_name)[1].lower()
        file_path = f"/tmp/{message.chat_id}{file_extension}"
        await file.download_to_drive(file_path)

        static_file_path, continue_Processing = save_file(file_path)
        static_file_name = os.path.basename(static_file_path)
        if continue_Processing:
            frames = [frame for frame in process_frames(str(CURRENT_TARGET), static_file_name, "gif", 1, [], 1, False)]
            gif_output = export_as_gif(frames, get_fps(static_file_name))
            gif_output.name = "output.gif"
            await context.bot.send_animation(chat_id=message.chat_id, animation=InputFile(gif_output), write_timeout=120*5)
        else: 
            last_file = sorted(os.listdir(IMAGE_DIR), key=lambda filename: int(filename.split('.')[0]))[-1]
            target = os.path.basename(last_file)
            context.args = [ target ]
            await set_target(update, context)
    elif message.document:
        await update.message.reply_text(f"File {message.document.file_name} not processed!")
    else:
        await update.message.reply_text(f"No file!")

async def start(update, context):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="You will want to /authenticate")

async def nope_handler(update, context):
    await update.message.reply_text(f"Nope")

if __name__ == '__main__':
    # Get your Telegram bot token from an environment variable
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')

    if bot_token is None:
        print("Please set the TELEGRAM_BOT_TOKEN environment variable.")
        exit(1)
    else:
        print(f"******************************************")
        print(f"TOKEN: {bot_token}")
        print(f"******************************************")

    # Create the Telegram Updater with your bot token
    app = ApplicationBuilder().token(bot_token).build()
    
    app.add_handler(CommandHandler('start', start))
    
    app.add_handler(CommandHandler("authenticate", authenticate))
    
    

    # Start the bot
    app.run_polling()
