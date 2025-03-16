import cv2, time, os
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
from collections import deque
from PIL import Image, ImageDraw, ImageFont

print("Starting program...")

yolo11_raasld = 'runs/train/yolov11_raasld/weights/best.pt'

ARABIC_LETTERS_RAASLD = {
    'Ain': 'ع',
    'Al': 'ال',
    'Alef': 'ا',
    'Beh': 'ب',
    'Dad': 'ض',
    'Dal': 'د',
    'Feh': 'ف',
    'Ghain': 'غ',
    'Hah': 'ح',
    'Heh': 'ه',
    'Jeem': 'ج',
    'Kaf': 'ك',
    'Khah': 'خ',
    'Laa': 'لا',
    'Lam': 'ل',
    'Meem': 'م',
    'Noon': 'ن',
    'Qaf': 'ق',
    'Reh': 'ر',
    'Sad': 'ص',
    'Seen': 'س',
    'Sheen': 'ش',
    'Tah': 'ط',
    'Teh': 'ت',
    'Teh_Marbuta': 'ة',
    'Thal': 'ذ',
    'Theh': 'ث',
    'Waw': 'و',
    'Yeh': 'ي',
    'Zah': 'ظ',
    'Zain': 'ز'
}

model_type = [{'name': 'yolo11_raasld', 'model': yolo11_raasld, 'letters': ARABIC_LETTERS_RAASLD}]

model_index = 0

def initialize_model():
    print("Initializing Model...")
    model = YOLO(model_type[model_index]['model'])
    model.conf = 0.60
    return model

def initialize_state():
    return {
        'model': initialize_model(),
        'detected_letters': [],
        'captured_frames': [],
        'detection_buffer': deque(maxlen=10),  # Increased buffer size
        'last_detection_time': time.time(),
        'last_letter': None,
        'cooldown': 1.5,  # Increased detection time
        'current_detection': None,
        'detection_start_time': 0,
        'english_letters': []
    }

# Draw text on frame in both Arabic and English
def draw_text(frame, text, position, scale=0.7, color=(0, 255, 0), align='left', max_width=None):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size=int(30 * scale))
    except:
        font = ImageFont.truetype("C:/Windows/Fonts/arialuni.ttf", size=int(30 * scale))

    # Handle text wrapping
    if max_width:
        lines = []
        words = text.split(' ')
        line = []
        for word in words:
            test_line = ' '.join(line + [word])
            text_size = draw.textbbox((0, 0), test_line, font=font)
            text_width = text_size[2] - text_size[0]
            if text_width <= max_width:
                line.append(word)
            else:
                lines.append(' '.join(line))
                line = [word]
        lines.append(' '.join(line))
    else:
        lines = [text]

    x, y = position
    for line in lines:
        text_size = draw.textbbox((0, 0), line, font=font)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]

        if align == 'right':
            x = position[0] - text_width
        elif align == 'center':
            x = position[0] - text_width // 2

        # Draw background with padding
        cv2.rectangle(frame,
                     (x-5, y - text_height - 5),
                     (x + text_width+5, y + 5),
                     (0, 0, 0),
                     -1)

        draw.text((x, y), line, font=font, fill=color[::-1])
        y += text_height + 10  # Move to the next line

    frame[:] = np.array(img_pil)

# Setup window for displaying frames
def setup_window():
    print("Setting up window...")
    cv2.namedWindow('ArSL Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ArSL Detection', 1280, 920)
    print("Window setup complete.")

# Process detection results and return highest confidence detection
def process_detection(frame, results, model):
    highest_conf_detection = None
    highest_conf = 0

    for det in results.boxes.data:
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        if conf >= model.conf and conf > highest_conf:
            highest_conf = conf
            highest_conf_detection = (x1, y1, x2, y2, conf, cls)

    return highest_conf_detection

# Handle letter detection and update state
def handle_letter_detection(frame, detection_info, state):
    x1, y1, x2, y2, conf, cls = detection_info
    letter = state['model'].names[int(cls)]
    current_time = time.time()

    # If new letter detected or detection stops, reset timer
    if letter != state['current_detection']:
        state['current_detection'] = letter
        state['detection_start_time'] = current_time
        state['detection_buffer'].clear()  # Clear buffer if detection stops
        return frame, state['cooldown']

    detection_duration = current_time - state['detection_start_time']
    time_left = max(0, state['cooldown'] - detection_duration)

    # Check if the confidence is above the threshold for the entire duration
    if conf < state['model'].conf:
        state['detection_start_time'] = current_time  # Reset detection start time
        state['detection_buffer'].clear()  # Clear buffer if confidence drops
        return frame, state['cooldown']

    if (detection_duration >= state['cooldown']):

        height, width = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 255, 0), -1)
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

        # Update state - append for both Arabic and English
        arabic_letter = model_type[model_index]['letters'].get(letter, letter)
        state['detected_letters'].append(arabic_letter)  # Append for Arabic
        state['english_letters'].append(letter)  # Append for English
        state['last_letter'] = letter
        state['last_detection_time'] = current_time
        state['captured_frames'].append(frame.copy())

        state['current_detection'] = None
        state['detection_start_time'] = 0

    return frame, time_left

# Draw detection box with pulsing effect
def draw_detection_box(frame, detection_info, model):
    x1, y1, x2, y2, conf, cls = detection_info
    box_color = (0, int(255 * conf), 0)

    cv2.rectangle(frame,
                 (int(x1), int(y1)),
                 (int(x2), int(y2)),
                 box_color, 2)

    label_pos = (int(x1), int(y1) - 50)
    letter = model.names[int(cls)]
    draw_text(frame, f"{letter} ({conf:.2f})", label_pos, scale=1)

# Create info panel with detected letters and controls
def create_info_panel(width, state, time_left):
    info_panel = np.zeros((200, width, 3), dtype=np.uint8)

    # Arabic word (right-aligned) - reverse for correct RTL display
    arabic_word = ''.join((state['detected_letters']))  # Reverse here for RTL
    arabic_text = f"الكلمة: {arabic_word}"

    # English letters (left-aligned)
    english_text = ' '.join(state['english_letters'])

    # Draw text
    draw_text(info_panel, arabic_text, (width - 10, 40), scale=1.2, align='right', max_width=width - 20)
    draw_text(info_panel, english_text, (10, 80), scale=1.2, align='left', max_width=width - 20)

    # Status and controls
    draw_text(info_panel, "Space: Add Space | Backspace: Delete | Enter: Read Sentence | Q/Esc: Exit",
             (width//2, 120), align='center', max_width=width - 20)
    ready_text = "Ready to Capture" if time_left == 0 else f"Wait {time_left:.1f}"
    draw_text(info_panel, f"Status: {ready_text}", (width - 10, 160), align='right', max_width=width - 20)

    return info_panel

# Handle keyboard events
def handle_key_events(key, state):
    if key == ord('q') or key == 27:  # q or Esc
        return True
    elif key == ord(' '): # Space
        state['detected_letters'].append(' ')
        state['english_letters'].append(' ')
    elif key == 8:  # Backspace
        if state['detected_letters']:
            state['detected_letters'].pop()
            state['english_letters'].pop()
    elif key == 13: # Enter
        word = ''.join(state['detected_letters'])
        play_detected_word(word)
    return False

# Play detected word using Google Text-to-Speech
def play_detected_word(word):
    # Create folder to store mp3 files if not already exists
    if not os.path.exists("Detected_Words/"):
        os.makedirs("Detected_Words/")

    print("Playing detected text...")
    current_time = time.time()
    tts = gTTS(text=word, lang='ar')
    tts.save(f"Detected_Words/{current_time}.mp3")
    os.system(f"start Detected_Words/{current_time}.mp3")

# Main function for real-time detection
def real_time_detection():
    print("Initializing state...")
    state = initialize_state()
    print("State initialized.")
    print("Opening video capture...")
    try:
        capture = cv2.VideoCapture(0)
        print("Video capture opened.")
    except:
        print("Error opening video capture. Make sure a camera is connected and working properly.")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    capture.set(cv2.CAP_PROP_FPS, 60)
    print("Video resolution set to 1280x720, FPS to 60.")

    setup_window()

    time_left = 0

    while True:
        if not capture.isOpened():
            break

        ret, frame = capture.read()
        if not ret:
            break

        # Draw ROI and process frame
        height, width = frame.shape[:2]
        results = state['model'](frame)[0]

        # Process detections
        highest_conf_detection =  process_detection(frame, results, state['model'])

        if highest_conf_detection is not None:
            frame, time_left = handle_letter_detection(frame, highest_conf_detection, state)
            draw_detection_box(frame, highest_conf_detection, state['model'])  # Pass model from state

        # Create and combine info panel
        info_panel = create_info_panel(width, state, time_left)
        combined_frame = np.vstack([frame, info_panel])
        cv2.imshow('ArSL Detection', combined_frame)

        # Exit the while loop if correct button is pressed
        if handle_key_events(cv2.waitKey(1) & 0xFF, state):
            break

    # Cleanup after the while loop ends
    capture.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == "__main__":
    try:
        real_time_detection()
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
        cv2.destroyAllWindows()