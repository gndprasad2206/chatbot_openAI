from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
import os
import json

# Load environment variables from .env file
load_dotenv()

# Set environment variables for LangChain
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Initialize ChatOpenAI client
chat_openai = ChatOpenAI(openai_api_key=os.getenv('OPEN_API_KEY'),model="gpt-4o")

# Define prompt templates
prompts = {
    "extract": """
        You are an experienced recruiter. Please extract the following information from the job description and return it as a JSON object. The input might be in key-value pairs or plain text. If the input is plain text, intelligently map the information to the appropriate keys based on context.

        The keys to extract are:
        1. Job Title
        2. Job Summary
        3. Key Responsibilities
        4. Required Skills
        5. Preferred Qualifications
        6. Experience Required
        7. Work Environment and Conditions
        8. Salary and Benefits
        9. Company Overview
        10. Application Instructions
        11. Equal Opportunity Statement
        12. Additional Information

        Job Description:
        {job_desc}

        Instructions:
        1. If the job description includes key-value pairs, directly use the provided keys and values.
        2. If the job description is in plain text, analyze the content to determine and map the relevant information to the appropriate keys.
        3. Extract any additional relevant details and include them in the JSON output.
        4. Use natural language processing to identify and categorize the information accurately.
        5. Ensure all key headings are captured, even if some details are not explicitly mentioned.

        Return the extracted information as a JSON object with the specified keys.
    """,
    "generalized": """
        You are an expert in job descriptions and hiring practices. Your task is to review and analyze the following job description (JD) to identify any areas needing improvement. Generate five specific questions that, when answered, will enhance the clarity, completeness, and overall quality of the JD. Start the numbering of questions from 0.

        Job Description:
        {job_description}

        Please follow these steps:
        1. Analyze the JD: Identify any sections that lack detail or clarity.
        2. Generate Questions: Create five specific and actionable questions aimed at gathering the missing details or clarifying any ambiguous information. These questions should focus on enhancing the technical details, role expectations, company information, and any other relevant aspects to make the JD more comprehensive and attractive to potential candidates.

        Please just only provide the questions in a numbered list format.
    """,
    "question": """
        Based on the following extracted job description entities, generate questions to fill in any gaps or missing information. Only ask questions for the data that is missing or incomplete. Each missing field should have a corresponding question.

        Entities:
        {entities}

        The following categories should be checked:
        1. Job Title
        2. Job Summary
        3. Key Responsibilities
        4. Required Skills
        5. Preferred Qualifications
        6. Experience Required
        7. Work Environment and Conditions
        8. Salary and Benefits
        9. Company Overview
        10. Application Instructions
        11. Equal Opportunity Statement
        12. Additional Information

        For each missing or empty field, generate a specific question to gather the necessary details. Ensure each question is clear and directly addresses the missing information.

        Please provide the questions in a numbered list format if jd have all the above required fields return only an empty list like this [].
    """,
    "refine": """
        You are an experienced recruiter. Based on the original job description, extracted entities, and additional information provided by the user, generate an enhanced job description in JSON format.

        Original Job Description:
        {job_desc}

        Extracted Entities:
        {entities}

        Additional Information:
        {answers}

        Please provide the refined job description as a JSON object.
    """,
    "follow_up": """
        You are an experienced recruiter. Based on the following responses to the initial questions, generate specific follow-up questions to gain deeper insights into the job requirements and provide more clarity on the answers provided.

        Responses:
        {answers}

        Instructions:
        1. Analyze each response to identify areas that need further clarification or additional details.
        2. Generate specific and actionable follow-up questions for each response to gather more detailed information.
        3. Ensure each question is clear, concise, and directly addresses the information provided in the response.
        4. Focus on aspects that would help a potential candidate better understand the job requirements, responsibilities, and expectations.
        5. response should only contain questions dont give any kind of description .

        Please provide the top 5 questions in a numbered list format.
    """
}

# Helper functions for interacting with LangChain
def generate_openai_response(prompt, prompt_type):
    try:
        template = PromptTemplate(template=prompt, input_variables=list(prompt_type.keys()))
        llm_chain = LLMChain(llm=chat_openai, prompt=template)
        response = llm_chain.run(prompt_type)
        return response.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def extract_entities(job_desc):
    response = generate_openai_response(prompts["extract"], {"job_desc": job_desc})
    try:
        return json.loads(response.strip("```json").strip("```JSON").strip())
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return {"error": "Response is not valid JSON"}

def generate_questions(extracted_entities, generalized_questions=False, follow_up=False):
    entities_str = json.dumps(extracted_entities, indent=2)
    prompt_type = {"entities": entities_str}

    if generalized_questions:
        prompt_type = {"job_description": entities_str}
        prompt = prompts["generalized"]
    elif follow_up:
        prompt_type = {"answers": entities_str}
        prompt = prompts["follow_up"]
    else:
        prompt = prompts["question"]

    response = generate_openai_response(prompt, prompt_type)
    return response.split('\n') if response else ["An error occurred while generating questions."]

def generate_refined_job_description(job_desc, extracted_entities, answers):
    prompt_type = {
        "job_desc": job_desc,
        "entities": json.dumps(extracted_entities, indent=2),
        "answers": json.dumps(answers, indent=2)
    }
    response = generate_openai_response(prompts["refine"], prompt_type)
    try:
        return json.loads(response.strip("```json").strip("```").strip())
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return {"error": "Response is not valid JSON"}

# Streamlit setup
st.title('Job Description Refinement')

# Initialize session state variables
for key in ['job_desc', 'extracted_entities', 'questions', 'generalized_questions', 'answers', 'follow_up_questions', 'current_question_index']:
    if key not in st.session_state:
        st.session_state[key] = "" if key == 'job_desc' else [] if 'questions' in key else {}

# User input for job description
st.session_state.job_desc = st.text_area("Enter the job description", value=st.session_state.job_desc)
if st.button("Extract Entities and Generate Questions"):
    st.session_state.extracted_entities = extract_entities(st.session_state.job_desc)
    st.write(st.session_state.extracted_entities)
    st.session_state.questions = generate_questions(st.session_state.extracted_entities)
    st.write(st.session_state.questions)
    st.session_state.current_question_index = 0
    st.session_state.answers = {}

# Interactive section for collecting responses to questions
if st.session_state.questions and not st.session_state.generalized_questions and not st.session_state.follow_up_questions:
    st.subheader("Please answer the following missing data to refine the job description:")
    current_index = st.session_state.current_question_index
    if current_index < len(st.session_state.questions):
        question = st.session_state.questions[current_index]
        answer_key = f"answer_{current_index}"
        answer = st.text_input(question, key=answer_key)
        if st.button("Submit Answer"):
            st.session_state.answers[answer_key] = answer
            st.session_state.current_question_index += 1
            st.rerun()
    else:
        st.write("Display further refined job description here (to be implemented)")

# Generalized refinement
if st.button("Generate Generalized Questions"):
    answers = {f"answer_{i}": st.session_state.answers.get(f"answer_{i}", "") for i in range(len(st.session_state.questions))}
    st.session_state.generalized_job_desc = generate_refined_job_description(st.session_state.job_desc, st.session_state.extracted_entities, answers)
    st.session_state.generalized_questions = generate_questions(st.session_state.generalized_job_desc, generalized_questions=True)
    st.write(st.session_state.generalized_job_desc)
    st.write(st.session_state.generalized_questions)
    st.session_state.current_question_index = 0
    st.session_state.answers = {}

# Collecting generalized question answers
if st.session_state.generalized_questions and not st.session_state.follow_up_questions:
    st.subheader("Generalized Questions:")
    current_index = st.session_state.current_question_index

    # for i in range(current_index,len( len(st.session_state.generalized_questions))):
    #     question = st.session_state.generalized_questions[current_index]

    
    if current_index < len(st.session_state.generalized_questions):
        question = st.session_state.generalized_questions[current_index]
        if question!="":
            answer_key = f"generalized_answer_{current_index}"
            answer = st.text_input(question, key=answer_key)

            if st.button("Submit Generalized Answer"):
                st.session_state.answers[answer_key] = answer
                st.session_state.current_question_index += 1
                st.rerun()
        if question=="":
            st.session_state.current_question_index += 1
            st.rerun()

    else:
        st.write("Generalized refinement done. Further process to be implemented.")

# Follow-up questions refinement
if st.button("Generate Follow-Up Questions"):
    all_answers = {**{f"answer_{i}": st.session_state.answers.get(f"answer_{i}", "") for i in range(len(st.session_state.questions))},
                   **{f"generalized_answer_{i}": st.session_state.answers.get(f"generalized_answer_{i}", "") for i in range(len(st.session_state.generalized_questions))}}
    st.session_state.follow_up_questions = generate_questions(all_answers, follow_up=True)
    st.write(st.session_state.follow_up_questions)
    st.session_state.current_question_index = 0
    st.session_state.answers = {}

# Collecting follow-up question answers
if st.session_state.follow_up_questions and st.session_state.current_question_index < len(st.session_state.follow_up_questions):
    st.subheader("Follow-Up Questions:")
    current_index = st.session_state.current_question_index

    if current_index < len(st.session_state.follow_up_questions):
        question = st.session_state.follow_up_questions[current_index]
        if question!="":
            answer_key = f"follow_up_answer_{current_index}"
            answer = st.text_input(question, key=answer_key)

            if st.button("Submit Follow-Up Answer"):
                st.session_state.answers[answer_key] = answer
                st.session_state.current_question_index += 1
                st.rerun()
        if question=="":
            st.session_state.current_question_index += 1
            st.rerun()

    else:
        st.write("Follow-up refinement done. Further process to be implemented.")

# Finalize job description
if st.button("Finalize Job Description"):
    all_answers = {**{f"answer_{i}": st.session_state.answers.get(f"answer_{i}", "") for i in range(len(st.session_state.questions))},
                   **{f"generalized_answer_{i}": st.session_state.answers.get(f"generalized_answer_{i}", "") for i in range(len(st.session_state.generalized_questions))},
                   **{f"follow_up_answer_{i}": st.session_state.answers.get(f"follow_up_answer_{i}", "") for i in range(len(st.session_state.follow_up_questions))}}
    final_refined_job_desc = generate_refined_job_description(st.session_state.job_desc, st.session_state.extracted_entities, all_answers)
    st.write(final_refined_job_desc)