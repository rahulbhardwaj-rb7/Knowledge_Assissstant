import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv
from components.question_answering import QuestionAnsweringSystem
import plotly.express as px
import pandas as pd

st.title("🤖 Knowledge Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = []
if "topic_context" not in st.session_state:
    st.session_state.topic_context = ""

@st.cache_resource
def initialize_qa_system():
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    if not google_api_key:
        st.error("GOOGLE_API_KEY not found in Streamlit secrets")
        return None
    
    try:
        return QuestionAnsweringSystem.create_instance(google_api_key)
    except Exception as e:
        st.error(f"Failed to initialize QA system: {e}")
        return None

qa_system = initialize_qa_system()

with st.sidebar:
    # st.header("📊 System Status")
    
    # if qa_system:
    #     stats = qa_system.get_database_stats()
    #     st.write(f"**Database Type:** {stats.get('database_type', 'Unknown')}")
    #     # st.write(f"**Total Documents:** {stats.get('total_documents', 0)}")
    # else:
    #     st.error("System Offline")
    #     st.write("Please check your Google API key")

    if st.button("🗑️ Clear Conversation", key="clear_conversation"):
        st.session_state.conversation_context = []
        st.session_state.topic_context = ""
        st.session_state.messages = []
        st.success("Conversation context cleared!")
        st.rerun()
    
    st.divider()
    
    st.header("📁 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['pdf', 'txt', 'csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
    )
    
    if uploaded_files and qa_system:
        if st.button("📤 Process Files", key="process_files"):
            with st.spinner("Processing documents and extracting tables..."):
                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = f"temp_{uploaded_file.name}"
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)
                
                try:
                    result = qa_system.add_documents_to_db(file_paths)
                    st.success(result)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing files: {e}")
                
                finally:
                    for file_path in file_paths:
                        try:
                            os.remove(file_path)
                        except:
                            pass
    
    st.divider()
    
    st.header("🗑️ Database Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear DB", key="clear_db"):
            if qa_system:
                result = qa_system.clear_database()
                st.success(result)
                st.session_state.conversation_context = []
                st.session_state.topic_context = ""
                st.session_state.messages = []
                if 'current_chart' in st.session_state:
                    del st.session_state['current_chart']
                st.rerun()
    
    with col2:
        if st.button("🔄 Refresh", key="refresh"):
            st.rerun()
    
    st.divider()
    
    st.header("📊 Chart Generation")
    
    if qa_system:
        available_tables = qa_system.get_available_tables()
        
        if available_tables:
            st.success(f"✅ Found {len(available_tables)} tables")
            
            table_options = {f"{table['id']} ({table['description']})": table['id'] 
                           for table in available_tables}
            
            selected_table_display = st.selectbox(
                "Select a table for chart generation:",
                list(table_options.keys())
            )
            
            if selected_table_display:
                selected_table_id = table_options[selected_table_display]
                
                table_info = qa_system.get_table_info(selected_table_id)
                if table_info:
                    with st.expander("📋 Table Details"):
                        st.write(f"**Shape:** {table_info['shape'][0]} rows × {table_info['shape'][1]} columns")
                        st.write(f"**Columns:** {', '.join(table_info['columns'])}")
                        
                        if table_info['sample_data']:
                            st.write("**Sample Data:**")
                            sample_df = pd.DataFrame(table_info['sample_data'])
                            st.dataframe(sample_df, use_container_width=True)
                
                st.subheader("🎨 Generate Chart")
                
                suggestions = qa_system.suggest_charts_for_table(selected_table_id)
                
                if suggestions:
                    chart_options = {f"{s['type'].title()}: {s['description']}": s['type'] 
                                   for s in suggestions}
                    
                    selected_chart_display = st.selectbox(
                        "Select chart type:",
                        list(chart_options.keys())
                    )
                    
                    if selected_chart_display:
                        selected_chart_type = chart_options[selected_chart_display]
                        
                        chart_suggestion = next(s for s in suggestions if s['type'] == selected_chart_type)
                        
                        x_column = None
                        y_column = None
                        color_column = None
                        
                        if chart_suggestion['x_options']:
                            x_column = st.selectbox(
                                "Select X-axis column:",
                                chart_suggestion['x_options']
                            )
                        
                        if chart_suggestion['y_options']:
                            y_column = st.selectbox(
                                "Select Y-axis column:",
                                chart_suggestion['y_options']
                            )
                        
                        if selected_chart_type in ['bar', 'scatter', 'line'] and table_info:
                            color_options = ['None'] + table_info['categorical_columns']
                            color_selection = st.selectbox(
                                "Color by (optional):",
                                color_options
                            )
                            if color_selection != 'None':
                                color_column = color_selection
                        
                        if st.button("📈 Generate Chart"):
                            with st.spinner("Generating chart..."):
                                fig, message = qa_system.generate_chart(
                                    selected_table_id, 
                                    selected_chart_type, 
                                    x_column, 
                                    y_column, 
                                    color_column
                                )
                                
                                if fig:
                                    st.success(message)
                                    st.session_state['current_chart'] = {
                                        'figure': fig,
                                        'title': f"{selected_chart_type.title()} Chart",
                                        'table_id': selected_table_id
                                    }
                                    st.rerun()
                                else:
                                    st.error(message)
                else:
                    st.info("No chart suggestions available for this table")
        else:
            st.info("📊 No tables found in uploaded documents")
            st.write("Upload PDF, Excel, or CSV files with tables to enable chart generation")

st.header("💬 Chat with your Documents")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        if qa_system:
            with st.spinner("Thinking..."):
                answer, augmented_questions, new_topic = qa_system.answer_question_with_context(
                    question=prompt,
                    conversation_context=st.session_state.conversation_context,
                    current_topic=st.session_state.topic_context
                )

                st.markdown(answer)
                
                if augmented_questions:
                    st.markdown("\n**Related questions you might ask:**")
                    for i, question in enumerate(augmented_questions, 1):
                        st.markdown(f"{i}. {question}")
                        
                st.session_state.conversation_context.append({
                    "question": prompt,
                    "answer": answer,
                    "timestamp": datetime.now().isoformat()
                })
                
                if new_topic:
                    st.session_state.topic_context = new_topic
                
                if len(st.session_state.conversation_context) > 10:
                    st.session_state.conversation_context = st.session_state.conversation_context[-5:]
                
                full_response = answer
                if augmented_questions:
                    full_response += "\n\n**Related questions:**\n" + "\n".join([f"{i}. {q}" for i, q in enumerate(augmented_questions, 1)])
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            error_msg = "❌ System not available. Please check your configuration."
            st.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

if 'current_chart' in st.session_state:
    st.header("📊 Generated Chart")
    
    chart_data = st.session_state['current_chart']
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.plotly_chart(chart_data['figure'], use_container_width=True)
    
    with col2:
        st.write(f"**Chart Type:** {chart_data['title']}")
        st.write(f"**Source Table:** {chart_data['table_id']}")
        
        if st.button("💾 Download Chart"):
            html_string = chart_data['figure'].to_html()
            st.download_button(
                label="📥 Download as HTML",
                data=html_string,
                file_name=f"chart_{chart_data['table_id']}.html",
                mime="text/html"
            )
        
        if st.button("🗑️ Clear Chart"):
            del st.session_state['current_chart']
            st.rerun()