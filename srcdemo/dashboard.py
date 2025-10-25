import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')

parser = st.sidebar
report_path = parser.text_input('Report JSON path', value='./data/tmp/report.json')

if report_path:
    try:
        with open(report_path) as f:
            rep = json.load(f)
    except Exception as e:
        st.error('Could not load report: ' + str(e))
        rep = None

if rep:
    st.title('Class Engagement Report')
    st.metric('Engagement Score', f"{rep['score']:.1f}", delta=None)
    st.metric('Engagement Class', rep['class'])

    col1, col2 = st.columns(2)
    with col1:
        st.header('HOT vs LOT')
        total_q = rep['nlp']['questions_total']
        hot = rep['nlp']['hot']
        lot = rep['nlp']['lot']
        st.write(f"Total questions: {total_q}")
        st.write(f"HOT: {hot} ({(hot/total_q*100) if total_q else 0:.1f}%)")
        st.write(f"LOT: {lot} ({(lot/total_q*100) if total_q else 0:.1f}%)")

    with col2:
        st.header('Participation & Emotions')
        st.write('Participation diversity (0-1):', rep.get('participation_diversity', rep['nlp'].get('participation_diversity',0)))
        st.write('Positive emotion %:', rep['video'].get('positive_emotion_pct'))

    st.header('Details')
    st.subheader('Audio metrics')
    st.json(rep['audio'])
    st.subheader('Video metrics')
    st.json(rep['video'])
    st.subheader('NLP (first 10 questions)')
    qs = rep['nlp'].get('questions', [])[:10]
    for q in qs:
        st.write('-', q.get('text'), f"({q.get('type')})")

    st.markdown('---')
    st.write('Export summary (PDF) — not implemented in GUI. Use script to create PDF from report.json')
