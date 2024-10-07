import csv
import pandas as pd
import streamlit as st
import nltk
from email.mime.image import MIMEImage
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    T5Tokenizer,
    T5ForConditionalGeneration
)
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import yake
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import time
import requests

# Preprocessing functions
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def verify_email(user_email):
    api_key = '1c9ecad90dbdd787a5022f68cbf7fdc4f7af9a77'
    url = f'https://api.hunter.io/v2/email-verifier?email={user_email}&api_key={api_key}'

    response = requests.get(url)
    print(response.text)  # Add this line to print the response

    data = response.json()

    if data['data']['result'] == 'deliverable':
        return True
    else:
        return False


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Text classification model
def train_classifier(texts, labels):
    preprocessed_texts = [preprocess_text(text) for text in texts]
    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(preprocessed_texts)
    classifier = MultinomialNB()
    classifier.fit(X, labels)
    return classifier, tfidf_vectorizer

# Sarcasm detection model (using Hugging Face Transformers)
def load_sarcasm_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

# Data summarization model (using Hugging Face Transformers)
def load_summarization_model():
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

def detect_sarcasm(text, model, tokenizer):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted label
    predicted_label = "Sarcastic" if torch.argmax(outputs.logits) == 1 else "Not Sarcastic"
    return predicted_label

# Keyword extraction function
def extract_keywords(text):
    kw_extractor = yake.KeywordExtractor()
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]

# Email transfer functionality
class EmailTransfer:
    def __init__(self):
        pass

    def transfer_email(self, department_emails, email_content, predicted_department, keywords):
        if predicted_department in department_emails:
            department = predicted_department
            email = department_emails[predicted_department]
            msg = MIMEMultipart()
            msg['From'] = 'atturpavankumar123@gmail.com'  # Your email address
            msg['To'] = email
            msg['Subject'] = 'Feedback for Department: ' + department

            msg.attach(MIMEText(email_content[department], 'html'))

            # Save Word Cloud plot as an image
            plt.figure(figsize=(8, 6))
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(keywords))
            wordcloud_image_path = 'wordcloud.png'
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Keywords Word Cloud')
            plt.savefig(wordcloud_image_path)
            plt.close()

            with open('wordcloud.png', 'rb') as img:
                msg_image = MIMEImage(img.read())
                msg_image.add_header('Content-ID', '<wordcloud.png>')
                msg_image.add_header("Content-Disposition", "inline", filename="wordcloud.png")
                msg.attach(msg_image)

            try:
                with smtplib.SMTP('smtp.gmail.com', 587) as server:
                    server.starttls()
                    server.login('atturpavankumar123@gmail.com', 'ldsi ocvs sses hxck')  # Your email credentials

                    server.send_message(msg)
                    print("Email sent successfully to", email)
            except smtplib.SMTPAuthenticationError as e:
                st.write("SMTP authentication error:", e)
            except smtplib.SMTPException as e:
                st.write("SMTP error:", e)
            except Exception as e:
                st.write("An error occurred:", e)
        else:
            print("Department email address not found for predicted department.")

# Load suspicious IPs from CSV
suspicious_ips_df = pd.read_csv('suspicious_ips.csv')

# Function to check if an IP address is suspicious
def is_suspicious(ip_address):
    return ip_address in set(suspicious_ips_df['IP'].str.strip())

# Function to get user's IP address
def get_user_ip():
    try:
        # Request the user's IP address from an external API
        response = requests.get('https://api.ipify.org?format=json')
        if response.status_code == 200:
            ip_data = response.json()
            return ip_data.get('ip')
    except Exception as e:
        st.error(f"Error fetching IP address: {e}")
        return None


# Streamlit app

def main():
    email_content = {}
    st.title("Customer Complaint Analysis")
    user_ip = get_user_ip()
    if user_ip:
        st.write("Your IP Address:", user_ip)

        if is_suspicious(user_ip):
            st.error("Your IP address is suspicious.")
            return
        else:
            input_text = st.text_area("Enter your complaint here:", "")

            user_email = st.text_input("Enter your email:")
            user_name = st.text_input("Enter your name:")

            # Validate user name
            if user_name and user_name[0].isupper() and user_name[1:].islower():
                st.success("Name format is valid!")
            else:
                st.error("Name must start with an uppercase letter followed by lowercase letters only.")
                return

            # Validate user email
            if user_email:
                if verify_email(user_email):
                    st.success("Email is valid!")
                else:
                    st.error("Email is invalid. Please provide a valid email address.")
                    return

            if input_text and user_email and user_name:
                st.success("Form submitted successfully!")

                # Check user submission history
                recent_submissions = []
                with open('user_activity.csv', 'r', newline='') as file:
                    reader = csv.reader(file)
                    for row in reader:
                        if row[0] == user_email:
                            recent_submissions.append(float(row[2]))

                # Check if user submitted within the last hour
                if len(recent_submissions) >= 2:
                    current_time = time.time()
                    last_submission_time = max(recent_submissions)
                    if current_time - last_submission_time <= 3600:  # 3600 seconds = 1 hour
                        st.error("You have already submitted complaints twice within the last hour. Please try again later.")
                        return

                # Log user submission
                with open('user_activity.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([user_email, user_name, time.time()])

                preprocessed_text = preprocess_text(input_text)

                honeypot_field = st.empty()

                if honeypot_field.text_input("Leave this field empty", key="honeypot") != "":
                    st.error("Spam detected. Please try again.")
                    st.write("Spam detected. Please try again.")
                    return

                with st.spinner("Predicting department label..."):
                    time.sleep(2)
                    department_label = classifier.predict(tfidf_vectorizer.transform([preprocessed_text]))[0]
                    st.write("Department label predicted:")
                    st.success(department_label)

                with st.spinner("Detecting sarcasm..."):
                    time.sleep(2)
                    sarcasm_label = detect_sarcasm(preprocessed_text, sarcasm_model, sarcasm_tokenizer)
                    st.write("Sarcasm detected label:")
                    st.success(sarcasm_label)

                with st.spinner("Summarizing input text..."):
                    time.sleep(2)
                    input_tokens = summarization_tokenizer.encode("summarize: " + input_text, return_tensors="pt",
                                                                  max_length=512, truncation=True)
                    summary_ids = summarization_model.generate(input_tokens, max_length=150, min_length=40,
                                                               length_penalty=2.0,
                                                               num_beams=4, early_stopping=True)
                    summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    st.write("Summary:")
                    st.success(summary)

                with st.spinner("Extracting keywords..."):
                    time.sleep(2)
                    keywords = extract_keywords(input_text)
                    st.write("Keywords:")
                    st.write(keywords)
                    plt.figure(figsize=(8, 6))
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(keywords))
                    wordcloud_image_path = 'wordcloud.png'
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.title('Keywords Word Cloud')
                    plt.savefig(wordcloud_image_path)
                    plt.close()
                    st.image(wordcloud_image_path)

                for department in department_emails:
                    email_content[department] = f"<h2>Feedback for Department: {department}</h2>"
                    email_content[department] += f"<p><strong>Input Text:</strong> {input_text}</p>"
                    email_content[
                        department] += f"<p><strong>Predicted Department Label:</strong> {department_label}</p>"
                    email_content[department] += f"<p><strong>Sarcasm Detection:</strong> {sarcasm_label}</p>"
                    email_content[department] += f"<p><strong>Summary:</strong> {summary}</p>"
                    email_content[department] += f"<p><strong>Keywords:</strong> {', '.join(keywords)}</p>"

                    plt.figure(figsize=(8, 6))
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(keywords))
                    wordcloud_image_path = 'wordcloud.png'
                    wordcloud.to_file(wordcloud_image_path)
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.title('Keywords Word Cloud')
                    plt.close()

                    with open('wordcloud.png', 'rb') as img:
                        msg_image = MIMEImage(img.read())
                        msg_image.add_header('Content-ID', '<wordcloud.png>')
                        msg_image.add_header("Content-Disposition", "inline", filename="wordcloud.png")
                        email_content[
                            department] += "<br><img src='cid:wordcloud.png'><br>"

        email_transfer = EmailTransfer()
        email_transfer.transfer_email(department_emails, email_content, department_label, keywords)






texts = [
    # Electronics Department Complaints
    "Defective charging port on purchased smartphone.",
    "TV purchased last week malfunctioning, screen flickers intermittently.",
    "Laptop keyboard keys sticking, affecting typing speed.",
    "Faulty HDMI port on newly bought gaming console.",
    "Bluetooth speaker not pairing with devices as advertised.",
    "Headphones purchased are emitting crackling sounds.",
    "Digital camera lens not focusing properly, resulting in blurry images.",
    "Tablet screen displaying abnormal flickering after recent software update.",
    "Smartwatch battery draining rapidly despite minimal usage.",
    "Router overheating frequently, causing intermittent internet connectivity.",
    "DVD player skipping and freezing during playback.",
    "Microwave purchased last month not heating food properly.",
    "Printer not recognizing newly installed ink cartridges.",
    "Gaming controller buttons sticking, affecting gameplay experience.",
    "Home theater system producing distorted audio.",
    "Air conditioner making loud noises when turned on.",
    "Refrigerator not cooling adequately, causing food to spoil.",
    "Vacuum cleaner losing suction power after a few uses.",
    "Alarm clock not functioning, failing to wake up on time.",
    "Blender blades not rotating smoothly, leaving chunks in smoothies.",
    # Electronics Department Complaints (continued)
    "Battery draining quickly on newly purchased tablet.",
    "Smartphone camera producing blurry images.",
    "Gaming console experiencing frequent crashes during gameplay.",
    "Bluetooth connection issues with wireless headphones.",
    "Television remote control not functioning properly.",
    "Faulty power button on laptop, unable to power on.",
    "Tablet screen cracked upon delivery.",
    "Smartwatch not receiving notifications from connected phone.",
    "Printer paper jamming frequently during printing.",
    "Digital camera lens cover stuck, preventing use.",
    "Router not broadcasting Wi-Fi signal intermittently.",
    "Home theater system remote control missing buttons.",
    "My smartphone battery drains too quickly, even though I barely use it.",
    "The Wi-Fi signal keeps dropping intermittently, disrupting my work.",
    "The screen of my laptop suddenly flickers or goes black.",
    "I can't hear the other person clearly during phone calls; the sound is distorted.",
    "The power button on my TV remote doesn't respond, even after changing the batteries.",
    "The printer constantly jams paper, making it impossible to print anything.",
    "The charging port on my tablet seems loose, and it's difficult to get a stable connection.",
    "My gaming console overheats frequently, causing it to shut down unexpectedly.",
    "There are dead pixels on my monitor, which is distracting when watching videos or working.",
    "The DVD player keeps skipping or freezing during playback, ruining the movie experience.",
    # Food Department Complaints
    "The food served was undercooked and completely unacceptable. It's concerning that basic food safety standards were not met.",
    "The portion sizes were incredibly small for the prices charged. We left the restaurant feeling hungry and unsatisfied.",
    "The menu was misleading with items listed that were not available. This led to disappointment and frustration.",
    "The waiter was rude and unprofessional, displaying a lack of basic courtesy and respect towards us as customers.",
    "The cleanliness of the restaurant was appalling. There were visible stains on the tablecloths and an unpleasant odor in the air.",
    "The vegetarian options were limited and uninspired. It's disappointing to see such a lack of creativity in catering to diverse dietary preferences.",
    "The food took an unreasonably long time to arrive, despite the restaurant not being particularly busy. This lack of efficiency was frustrating.",
    "The prices on the menu did not match what was charged on the bill, resulting in an unexpected and unwelcome expense.",
    "The restaurant was extremely noisy, making it difficult to enjoy our meal and have a conversation without shouting.",
    "The food was bland and tasteless, lacking any discernible flavor or seasoning. It was a disappointing dining experience overall.",
    "Expired products being sold on shelves.",
    "Moldy bread purchased from the bakery section.",
    "Raw meat packaged with expired sell-by date.",
    "Canned goods found to be dented and possibly compromised.",
    "Stale pastries served at the bakery counter.",
    "Unpleasant odor emanating from deli section, potentially spoiled products.",
    "Fresh produce (fruits and vegetables) found to be rotting prematurely.",
    "Incorrect pricing at the checkout, charged more than advertised.",
    "Food poisoning after consuming prepared meals from the hot deli.",
    "Inadequate labeling on allergens, causing allergic reactions.",
    "Contaminated salad bar items causing foodborne illness.",
    "Unhygienic food handling practices witnessed by customers.",
    "Misleading nutritional information on packaged foods.",
    "Overripe fruits being sold at regular price.",
    "Limited vegetarian and vegan options available.",
    "Insect infestation discovered in grains and cereals aisle.",
    "Dairy products found to be sour before expiration date.",
    "Unresponsive customer service regarding food quality concerns.",
    "Overcooked and dry rotisserie chicken.",
    "Disorganized shelves making it difficult to locate products.",
       # Food Department Complaints (continued)
    "Unsealed packaging on dairy products.",
    "Expired yogurt sold in refrigerated section.",
    "Stale chips discovered in snack aisle.",
    "Mold found on packaged cheese slices.",
    "Rancid smell from packaged deli meats.",
    "Misleading expiration dates on canned goods.",
    "Fruit flies observed around produce section.",
    "Unlabeled allergens in baked goods.",
    "Overpriced organic produce compared to non-organic.",
    "Sticky residue on canned soda labels.",
    "Mismatched product advertised compared to actual content.",
    "Fruit basket missing variety of fruits as advertised.",
    # Clothing Department Complaints
    "Torn seams on recently purchased jeans.",
    "Faded colors on shirts after first wash, not as advertised.",
    "Shoes purchased showing signs of premature wear and tear.",
    "Zipper on jacket malfunctioning, getting stuck frequently.",
    "Sweater shrinking significantly after following washing instructions.",
    "Dress material fraying after a single wear.",
    "Shirt size discrepancy, labeled size does not match actual fit.",
    "Stain on trousers discovered upon bringing them home.",
    "Uncomfortable fabric causing skin irritation.",
    "Missing buttons on blouses, affecting overall appearance.",
    "Unavailable sizes for popular clothing items.",
    "Poor stitching quality on seams of dresses.",
    "Misleading sale pricing, advertised discounts not applied at checkout.",
    "Poorly designed pockets on pants, items falling out easily.",
    "Unwanted fabric pilling on sweaters after minimal use.",
    "Unresponsive zippers on hoodies, difficult to open and close.",
    "Jacket purchased showing signs of color bleeding after washing.",
    "Unraveling threads on hems of skirts.",
    "Wrinkled clothing items on display, giving a poor impression.",
    "Scarves unravel.",
     # Clothing Department Complaints (continued)
    "Button missing on newly purchased blouse.",
    "Seam unraveling on dress after first wash.",
    "Shoe sole detaching from shoe body.",
    "Stain on shirt discovered after purchase.",
    "Zipper broken on jacket, unable to zip up.",
    "Incorrect size sent for online clothing order.",
    "Fabric tearing on pants after minimal wear.",
    "Threadbare material on sweater after first wash.",
    "Unsightly pilling on scarf after use.",
    "Socks unraveling at seam after first wear.",
    "Loose threads on jacket cuffs.",
    "Wrinkled clothing received from online order.",
]

labels = ['Electronics'] * 42 + ['Food'] * 42 + ['Clothing'] * 32

# Train text classification model
print("Training text classification model...")
classifier, tfidf_vectorizer = train_classifier(texts, labels)
print("Text classification model trained successfully.")

# Load sarcasm detection model
print("Loading sarcasm detection model...")
sarcasm_model, sarcasm_tokenizer = load_sarcasm_model()
print("Sarcasm detection model loaded successfully.")

# Load data summarization model
print("Loading data summarization model...")
summarization_model, summarization_tokenizer = load_summarization_model()
print("Data summarization model loaded successfully.")

# Department email addresses
department_emails = {
    'Electronics': 'atturpavankumar123@gmail.com',
    'Food': 'lokeshakishore@gmail.com',
    'Clothing': 'lokeshakishore@gmail.com',
}

if __name__ == "__main__":
    main()



