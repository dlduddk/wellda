import os
import streamlit as st

csv_file = st.file_uploader('CSV 파일 업로드', type=['csv'])
print(csv_file)
if csv_file is not None:

    import re
    from pinecone import Pinecone, ServerlessSpec
    from sentence_transformers import SentenceTransformer
    from langchain.docstore.document import Document
    import csv
    from text_processor import TextProcessor
    from dotenv import load_dotenv
    load_dotenv()


    # 파일 업로드 함수
    def save_uploaded_file(directory, file):
        if not os.path.exists(directory): # 해당 이름의 폴더가 존재하는지 여부 확인
            os.makedirs(directory) # 폴더가 없다면 폴더를 생성한다.
        
        with open(os.path.join(directory, file.name), 'wb') as f: #해당 경로의 폴더에서 파일의 이름으로 생성하겠다.
            f.write(file.getbuffer()) # 해당 내용은 Buffer로 작성하겠다.
            # 기본적으로 이미즈는 buffer로 저장되고 출력할때도 buffer로 출력한다.
        return st.success('파일 업로드 성공')
  
    file_path = f'./data/{csv_file.name}'
    save_uploaded_file('data', csv_file)
    with open(file_path, 'r', encoding='utf-8') as f:
        file_content = f.read()

    index_name = "wellda"
    processor = TextProcessor(os.environ.get('pinecone_api'), index_name)
    try:
        with st.spinner("upserting..."):
            processor.process_and_upsert_csv(file_content)
            st.success('벡터DB upsert 성공')
    except Exception as e:    # 모든 예외의 에러 메시지를 출력할 때는 Exception을 사용
        print('예외가 발생했습니다.', e)
        st.error('벡터DB upsert 실패')
    
