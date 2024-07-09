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
    # Initialize embedding model
    #model = SentenceTransformer("upskyy/kf-deberta-multitask")
    # Initialize Pinecone
    #pc = Pinecone(api_key=os.environ['pinecone_api'])
    # Define the index name
    index_name = "wellda"
    processor = TextProcessor(os.environ.get('pinecone_api'), index_name)
    #print(processor.process_csv(file_content))
    try:
        with st.spinner("upserting..."):
            processor.process_and_upsert_csv(file_content)
            st.success('벡터DB upsert 성공')
    except Exception as e:    # 모든 예외의 에러 메시지를 출력할 때는 Exception을 사용
        print('예외가 발생했습니다.', e)
        st.success('벡터DB upsert 실패')
    
    # Create the index if it doesn't existS
#    if index_name not in pc.list_indexes():
#        pc.create_index(index_name, 
#                            dimension=768,
#                            metric='cosine',
#                            spec=ServerlessSpec(
#                                cloud="aws",
#                                region="us-east-1") 
#                                ) # 768
#    # Instantiate the index
#    index = pc.Index(index_name)
#

    # Define a function to preprocess text
#    def preprocess_text(text):
#        # Replace consecutive spaces, newlines and tabs
#        text = re.sub(r'\s+', '\n', text)
#        return text

#    def process_csv(file_path):
#      # load csv data
#      #print(dir(file_path))
#      #print(file_path.name)
#        f = open(file_path, 'r',encoding='utf-8')
#        data = csv.reader(f,delimiter = ',')
#        header = next(data)
#      
#      # print(header)
#      # 번호 / 분류 / 키워드 / 답변 / 이미지/ 변환 이미지 / 레퍼런스 / 제작 진행 / last edited by
#        documents = []
#        for row in data:
#            content = "키워드\n"+row[2].strip()
#            content += "\n\n답변 : "+row[3]
#            print(content)
#          #documents.append(Document(page_content=content))
#            metadata={'번호':row[0],
#                    '분류':row[1],
#                    '키워드':row[2],
#                    '이미지':row[4],
#                    '변환 이미지':row[5],
#                    '레퍼런스':row[6],
#                    '제작 진행':row[7],
#                    'last edited by':row[8]}
#            documents.append(Document(page_content=content, metadata=metadata))
#      #f.close()
#
#        return documents

#  # Define a function to create embeddings
#    def create_embeddings(ts):
#        embeddings_list = []
#        for text in ts:
#            input_em = model.encode(text.page_content).tolist()
#            embeddings_list.append(input_em)
#
#        return embeddings_list

  # Define a function to upsert embeddings to Pinecone
#    def upsert_embeddings_to_pinecone(index, embeddings, ids):
#        data = []
#        for idx, embedding, doc in zip(ids, embeddings, documents):
#            data.append((str(idx),embedding,doc.metadata))
#
#        index.upsert(vectors=data)

  # Process a csv and create embeddings
  



#    import re
#
#
#    def clean_str(text):
#        pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'  # 한글 자음, 모음 제거
#        text = re.sub(pattern=pattern, repl='', string=text)
#      
#        pattern = '<[^>]*>'         # HTML 태그 제거
#        text = re.sub(pattern=pattern, repl='', string=text)
#      
#        text = re.sub(r'\s+', ' ', text)
#        return text 
#
#    for doc in documents:
#        text = doc.page_content
#        text=clean_str(text)
#        doc.metadata['page_content'] = text
#
#    embeddings = create_embeddings(documents)
#    upsert_embeddings_to_pinecone(index, embeddings, [str(i) for i in range(len(embeddings))])
#