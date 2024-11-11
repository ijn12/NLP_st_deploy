import streamlit as st
import json
from io import BytesIO, StringIO
import base64
import csv
import genanki
import random
from pdf_generator import generate_pdf
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
from datetime import datetime

def generate_study_materials(vectordb, topic: str, client) -> dict:
    """Generate structured study materials using OpenAI's chat completions"""
    # Get relevant context from vector database
    context = ""
    if vectordb:
        search_results = vectordb.similarity_search(topic, k=3)
        for result in search_results:
            context += f"\nFrom {result.metadata.get('filename', 'document')}, Page {result.metadata.get('page', 'unknown')}:\n"
            context += f"{result.page_content}\n"

    messages = [
        {
            "role": "system",
            "content": "You are an expert educational content creator and professor. Create comprehensive, university-level study materials that thoroughly explore the topic in depth."
        },
        {
            "role": "user",
            "content": f"""Create detailed study materials about {topic}.
            
            Use this reference material if available:
            {context}
            
            For the study guide:
            - Provide a comprehensive, in-depth explanation suitable for a university course
            - Include theoretical foundations and practical applications
            - Cover historical context and modern developments
            - Explain core concepts, methodologies, and best practices
            - Include real-world examples and case studies
            - Discuss limitations, challenges, and future directions
            - Add relevant formulas, diagrams, or technical details if applicable
            
            Also include 12 flashcards and 4 exercises with detailed solutions."""
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "study_materials_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "study_guide": {
                                "type": "object",
                                "properties": {
                                    "detailed_explanation": {"type": "string"},
                                    "key_concepts": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["detailed_explanation", "key_concepts"],
                                "additionalProperties": False
                            },
                            "flashcards": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "front": {"type": "string"},
                                        "back": {"type": "string"}
                                    },
                                    "required": ["front", "back"],
                                    "additionalProperties": False
                                }
                            },
                            "exercises": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "question": {"type": "string"},
                                        "solution": {"type": "string"}
                                    },
                                    "required": ["question", "solution"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["study_guide", "flashcards", "exercises"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        )
        
        generated_content = json.loads(response.choices[0].message.content)
        
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        return {
            'title': f"{topic} Study Guide",
            'subtitle': f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}",
            'main_content': generated_content['study_guide']['detailed_explanation'],
            'sidebar_content': "\n".join(generated_content['study_guide']['key_concepts']),
            'flashcards': generated_content['flashcards'][:12],
            'exercises': generated_content['exercises'][:4]
        }
    except Exception as e:
        st.error(f"Error generating study materials: {e}")
        return None

def generate_downloads(content):
    """Generate and display download buttons for study materials"""
    pdf = generate_pdf(content)
    st.success("Materials generated successfully!")
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    base_filename = f"NLP_material_{current_time}"
    
    st.markdown("### 📥 Download Materials")
    st.markdown(get_download_link(pdf, f"{base_filename}.pdf", "📄 Download PDF Study Guide"), 
               unsafe_allow_html=True)
    
    if content['flashcards']:
        st.markdown("### 📱 Flashcard Exports")
        
        quizlet_text = generate_quizlet_format(content['flashcards'])
        st.markdown(get_download_link(quizlet_text, f"{base_filename}_quizlet.txt", "📚 Download for Quizlet Import"), 
                   unsafe_allow_html=True)
        
        try:
            anki_deck = generate_anki_deck(content['flashcards'], content['title'])
            st.markdown(get_download_link(anki_deck, f"{base_filename}_anki.apkg", "🎴 Download Anki Deck"), 
                       unsafe_allow_html=True)
        except Exception as e:
            st.warning("Anki deck generation failed. You can still use the other export options.")
        
        csv_content = generate_csv_format(content['flashcards'])
        st.markdown(get_download_link(csv_content, f"{base_filename}_flashcards.csv", "📊 Download CSV"), 
                   unsafe_allow_html=True)
    
    display_preview(content)

def generate_quizlet_format(flashcards):
    """Generate a tab-separated format suitable for Quizlet import"""
    output = StringIO()
    for card in flashcards:
        front = card['front'].replace('\n', ' ').replace('\t', ' ')
        back = card['back'].replace('\n', ' ').replace('\t', ' ')
        output.write(f"{front}\t{back}\n")
    return output.getvalue()

def generate_anki_deck(flashcards, deck_title):
    """Generate an Anki deck from flashcards"""
    model = genanki.Model(
        random.randrange(1 << 30, 1 << 31),
        'Simple Model',
        fields=[{'name': 'Question'}, {'name': 'Answer'}],
        templates=[{
            'name': 'Card 1',
            'qfmt': '{{Question}}',
            'afmt': '{{FrontSide}}<hr id="answer">{{Answer}}',
        }]
    )
    
    deck = genanki.Deck(random.randrange(1 << 30, 1 << 31), deck_title)
    
    for card in flashcards:
        note = genanki.Note(
            model=model,
            fields=[card['front'], card['back']]
        )
        deck.add_note(note)
    
    package = genanki.Package(deck)
    temp_file = BytesIO()
    package.write_to_file(temp_file)
    return temp_file.getvalue()

def generate_csv_format(flashcards):
    """Generate a CSV format of the flashcards"""
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Front', 'Back'])
    for card in flashcards:
        writer.writerow([card['front'], card['back']])
    return output.getvalue()

def get_download_link(file_content, filename, display_text):
    """Generate an HTML download link for file content"""
    b64 = base64.b64encode(file_content.encode() if isinstance(file_content, str) else file_content).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}" class="download-button">{display_text}</a>'

def display_preview(content):
    """Display preview of generated content"""
    with st.expander("👀 Preview Generated Content"):
        st.subheader("Study Guide")
        st.write(content['main_content'])
        
        st.subheader("Key Concepts")
        st.write(content['sidebar_content'])
        
        st.subheader("Flashcards")
        for i, card in enumerate(content['flashcards'], 1):
            st.markdown(f"**Card {i}**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Front:**")
                st.write(card['front'])
            with col2:
                st.markdown("**Back:**")
                st.write(card['back'])
            st.divider()
        
        st.subheader("Exercises")
        for i, exercise in enumerate(content['exercises'], 1):
            st.markdown(f"**Exercise {i}**")
            st.markdown("**Question:**")
            st.write(exercise['question'])
            st.markdown("**Solution:**")
            st.write(exercise['solution'])
            st.divider()