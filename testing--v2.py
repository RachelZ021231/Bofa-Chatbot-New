import os
import io
import openai
import sqlite3
import pandas as pd
import streamlit as st
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import hashlib  # For generating unique keys

# ----------------------------
# User Input: OpenAI API Key
# ----------------------------
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    # Assign API Key
    openai.api_key = openai_api_key

    # ----------------------------
    # Function Definitions
    # ----------------------------

    def generate_sql(user_question, error_message=None):
        """
        Generates an SQL query from a natural language question using OpenAI's GPT.
        Includes logic to correct the query if the previous attempt caused an error.
        """
        error_prompt = (
            f"\nThe previous SQL query resulted in the following error:\n{error_message}\n"
            "Please correct the SQL query."
            if error_message else ""
        )

        prompt = f"""
Pretend you are an expert at converting natural language questions into accurate SQL queries. Please generate an accurate SQL query based on
the following natural language question and database schema provided below. Think sequentially and refer to the sample natural language
questions with correct and incorrect outputs as well.

Database Schema:
Table 1: t_zacks_fc (This table contains fundamental indicators for companies)
Columns: 'ticker', 'comp_name', 'exchange', 'per_end_date', 'per_type', 'filing_date', 'filing_type', 'zacks_sector_code', 'eps_diluted_net_basic', 'lterm_debt_net_tot'
Keys: ticker, per_end_date, per_type

Table 2: t_zacks_fr (This table contains fundamental ratios for companies)
Columns: 'ticker', 'per_end_date', 'per_type', 'ret_invst', 'tot_debt_tot_equity'
Keys: ticker, per_end_date, per_type.

Table 3: t_zacks_mktv (This table contains market value data for companies)
Columns: 'ticker', 'per_end_date', 'per_type', 'mkt_val'
Keys: ticker, per_end_date, per_type.

Table 4: t_zacks_shrs (This table contains shares outstanding data for companies)
Columns: 'ticker', 'per_end_date', 'per_type', 'shares_out'
Keys: ticker, per_end_date, per_type.

Table 5: t_zacks_sectors (This table contains the zacks sector codes and their corresponding sectors)
Columns: 'zacks_sector_code', 'sector'
Keys: zacks_sector_code

Sample natural language questions with correct and incorrect outputs:
Sample prompt 1: Output ticker with the largest market value recorded on any given period end date.
Correct output for prompt 1: SELECT ticker, per_end_date, MAX(mkt_val) AS max_market_value FROM t_zacks_mktv GROUP BY per_end_date ORDER BY
max_market_value DESC LIMIT 1;
Incorrect output for prompt 1: SELECT MAX(mkt_val) , ticker FROM t_zacks_mktv GROUP BY ticker

Sample prompt 2: What is the company name with the lowest market cap?
Correct output for prompt 2: SELECT fc.comp_name, mktv.ticker, mktv.mkt_val FROM t_zacks_mktv AS mktv JOIN t_zacks_fc AS fc ON mktv.ticker =
fc.ticker WHERE mktv.mkt_val = (SELECT MIN(mkt_val) FROM t_zacks_mktv);
Incorrect output for prompt 2:  SELECT T1.comp_name FROM t_zacks_fc AS T1 INNER JOIN t_zacks_mktv AS T2 ON T1.ticker = T2.ticker AND
T1.per_end_date = T2.per_end_date AND T1.per_type = T2.per_type ORDER BY T2.mkt_val LIMIT 1

Sample prompt 3: Filter t_zacks_fc to only show companies with a total debt-to-equity ratio greater than 1.
Correct output for prompt 3: SELECT * FROM t_zacks_fr WHERE tot_debt_tot_equity > 1;
Incorrect output for prompt 3: SELECT * FROM t_zacks_fr WHERE t_zacks_mktv > 1;

Sample prompt 4: Filter t_zacks_shrs to include companies with more than 500 million shares outstanding as of the most recent quarter.
Correct output for prompt 4: SELECT *
FROM t_zacks_shrs
WHERE shares_out > 5000
ORDER BY per_end_date DESC;
Incorrect output for prompt 4: SELECT * FROM t_zacks_shrs WHERE shares_out > 500000000

Sample prompt 5: Combine t_zacks_mktv and t_zacks_shrs to show tickers with market cap and shares outstanding in the latest period end date.
Correct output for prompt 5: SELECT mktv.ticker, mktv.per_end_date, mktv.mkt_val, shrs.shares_out
FROM t_zacks_mktv mktv
JOIN t_zacks_shrs shrs ON mktv.ticker = shrs.ticker AND mktv.per_end_date = shrs.per_end_date
ORDER BY mktv.per_end_date DESC;
Incorrect output for prompt 5: SELECT ticker, mkt_val, shares_out FROM t_zacks_mktv INNER JOIN t_zacks_shrs ON t_zacks_mktv.ticker =
t_zacks_shrs.ticker AND t_zacks_mktv.per_end_date = t_zacks_shrs.per_end_date ORDER BY per_end_date DESC LIMIT 1

Sample prompt 6: Join t_zacks_fc and t_zacks_fr to show tickers with total debt-to-equity ratios and EPS from NASDAQ as of Q2 2024.
Correct output for prompt 6: SELECT fc.ticker, fc.eps_diluted_net_basic, fr.tot_debt_tot_equity
FROM t_zacks_fc fc
JOIN t_zacks_fr fr ON fc.ticker = fr.ticker AND fc.per_end_date = fr.per_end_date AND fc.per_type = fr.per_type
WHERE fc.exchange = 'NASDAQ' AND fc.per_type = 'Q' AND fc.per_end_date BETWEEN '2024-04-01' AND '2024-06-30';
Incorrect output for prompt 6: SELECT T1.ticker, T1.eps_diluted_net_basic, T2.ret_invst, T2.tot_debt_tot_equity FROM t_zacks_fc AS T1 INNER
JOIN t_zacks_fr AS T2 ON T1.ticker = T2.ticker AND T1.per_end_date = T2.per_end_date WHERE T1.exchange = 'NASDAQ' AND T1.per_type = 'Q2';

Please make sure that when you are joining 2 or more tables, you are using all 3 keys (ticker, per_end_date & per_type). Also, ensure that the 
SQL query is syntactically correct and provides the expected output based on the natural language question provided.

User Question:
{user_question}
{error_prompt}

Please provide only the SQL query without any markdown, code block syntax, or explanations.
    """
        response = openai.ChatCompletion.create(
            model="ft:gpt-4o-2024-08-06:personal::AXYv83vn",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.0
        )
        return response.choices[0].message["content"].strip()

    def execute_sql(sql, tables):
        """
        Executes a given SQL query on an in-memory SQLite database populated with provided tables.
        Returns a DataFrame if successful, otherwise returns the error message.
        """
        conn = None
        try:
            conn = sqlite3.connect(":memory:")
            for table_name, df in tables.items():
                df.to_sql(table_name, conn, index=False, if_exists="replace")
            result = pd.read_sql_query(sql, conn)
            return result
        except Exception as e:
            return str(e)
        finally:
            if conn:
                conn.close()

    def modify_sql_query(current_sql, modification_request):
        """
        Modifies the given SQL query based on user-provided feedback using GPT.
        """
        prompt = f"""
The current SQL query is:
{current_sql}

The user has requested the following modifications or adjustments:
{modification_request}

Please provide a corrected or modified SQL query based on the user's request.
Ensure the query is syntactically correct and matches the provided requirements.
"""
        response = openai.ChatCompletion.create(
            model="ft:gpt-4o-2024-08-06:personal::AXYv83vn",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.0
        )
        return response.choices[0].message["content"].strip()

    def analyze_data(user_question, data):
        """
        Analyzes SQL query results using OpenAI GPT, providing equity analysis insights as plain text.
        """
        table_md = data.to_markdown(index=False)
        prompt = f"""
I have executed a SQL query based on the following user question and obtained the data below.

User's Question:
{user_question}

Data Table:
{table_md}

Pretend you are an experienced equity analyst working in the banking industry. Please analyze this data in the style of an expert equity analyst, highlighting trends, comparing companies, analyzing significance of metrics, and noting any other interesting insights regarding this data. 
Give at least 3-4 in-depth paragraphs with your analysis and feel free to incorporate external information related to the data being analyzed. Do NOT use any latex / equations in your output and only give plain text. Again, ensure your output is formatted as just text in a few paragraphs WITH NO markdown formatting involved.
"""
        response = openai.ChatCompletion.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message["content"].strip()

    def save_analysis_as_pdf(user_question, analysis_text):
        """
        Creates a formatted PDF report of the analysis.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("Equity Analysis Report", styles['Title']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("User's Question:", styles['Heading2']))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(user_question, styles['BodyText']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Analysis:", styles['Heading2']))
        elements.append(Spacer(1, 6))
        for para in analysis_text.split('\n\n'):
            elements.append(Paragraph(para.strip(), styles['BodyText']))
            elements.append(Spacer(1, 12))

        doc.build(elements)
        buffer.seek(0)
        return buffer

    def save_full_conversation_as_pdf(conversation):
        """
        Saves the entire conversation (user questions and bot analyses) into a formatted PDF.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("Full Equity Analysis Report", styles['Title']))
        elements.append(Spacer(1, 12))

        for idx, entry in enumerate(conversation):
            if entry['role'] == 'user':
                elements.append(Paragraph(f"Question {idx // 2 + 1}:", styles['Heading2']))
                elements.append(Spacer(1, 6))
                elements.append(Paragraph(entry['content'], styles['BodyText']))
                elements.append(Spacer(1, 12))
            else:
                elements.append(Paragraph("Analysis:", styles['Heading2']))
                elements.append(Spacer(1, 6))
                for para in entry['content'].split('\n\n'):
                    elements.append(Paragraph(para.strip(), styles['BodyText']))
                    elements.append(Spacer(1, 12))

        doc.build(elements)
        buffer.seek(0)
        return buffer

    def load_tables(files):
        """
        Loads data files into pandas DataFrames for SQLite processing.
        """
        tables = {}
        for table_name, file_path in files.items():
            if file_path.endswith(".parquet"):
                tables[table_name] = pd.read_parquet(file_path)
            elif file_path.endswith(".csv"):
                tables[table_name] = pd.read_csv(file_path)
        return tables

    def is_finance_related(question):
        """
        Determines if a user question is related to finance by searching for relevant keywords.
        """
        finance_keywords = {
            'company', 'market', 'shares', 'investment', 'debt', 'equity', 'finance', 'stock', 'ticker',
            'sector', 'EPS', 'return on investment', 'ROI', 'market cap', 'valuation', 'capital', 'filing',
            'ticker', 'comp_name', 'exchange', 'per_end_date', 'per_type', 'filing_date', 'filing_type',
            'zacks_sector_code', 'eps_diluted_net_basic', 'lterm_debt_net_tot', 'ret_invst',
            'tot_debt_tot_equity', 'mkt_val', 'shares_out', 'sector', 'earnings', 'profit', 'loss',
            'liabilities', 'assets', 'cash flow', 'balance sheet', 'income statement', 'revenue',
            'expenses', 'long-term debt', 'lt debt', 'P/E', 'price to earnings', 'D/E', 'debt to equity',
            'return on equity', 'ROE', 'ROA', 'return on assets', 'NASDAQ', 'NYSE', 'financial ratios'
        }
        return any(keyword.lower() in question.lower() for keyword in finance_keywords)

    def display_assistant_message(content):
        """
        Formats and displays assistant messages in the chat.
        Handles SQL queries with proper formatting.
        """
        st.session_state.messages.append({"role": "assistant", "content": content})
        with st.chat_message("assistant"):
            if "Generated SQL Query:" in content:
                # Extract SQL queries from content
                try:
                    sql_query = content.split("```sql\n")[1].split("\n```")[0]
                    st.markdown("**Generated SQL Query:**")
                    st.code(sql_query, language="sql")
                except IndexError:
                    st.markdown(content)
            else:
                st.markdown(content)

    # ----------------------------
    # Streamlit Interface & Logic
    # ----------------------------

    # Display title and description
    st.title("💬 Bank of America's Financial Data Analytics ChatBot")
    st.write("""
    Hi there. My name is BoFA-nanza and I am your personal AI chatbot. You can ask me questions regarding company fundamentals, financial ratios, market values, shares outstanding, and sector classifications for all US equities dating back until 2006. I have sourced all this data from NASDAQ's premier publisher of alternative data for institutional investors, Quandl. Specifically, I have access to The Zacks Fundamentals Collection A (ZFA) data feed. Please refer to the database schema below for additional information. 
    """)

    # Database architecture displayed for user reference
    with st.expander("Database Schema", expanded=True):
        st.write("""
        **Table 1: t_zacks_fc**
        Columns: ticker, comp_name, exchange, per_end_date, per_type, filing_date, filing_type, zacks_sector_code, eps_diluted_net_basic, lterm_debt_net_tot

        **Table 2: t_zacks_fr**
        Columns: ticker, per_end_date, per_type, ret_invst, tot_debt_tot_equity

        **Table 3: t_zacks_mktv**
        Columns: ticker, per_end_date, per_type, mkt_val

        **Table 4: t_zacks_shrs**
        Columns: ticker, per_end_date, per_type, shares_out

        **Table 5: t_zacks_sectors**
        Columns: zacks_sector_code, sector
        """)

    # Prompt the user to continue
    if "continue" not in st.session_state:
        if st.text_input("If you are willing to proceed, please type 'yes' to continue:").strip().lower() == "yes":
            st.session_state["continue"] = True

    if "conversation" not in st.session_state:
        st.session_state["conversation"] = []

    if "analyses" not in st.session_state:
        st.session_state["analyses"] = {}

    # Main Interaction
    if st.session_state.get("continue"):
        st.subheader("ChatBot Examples")
        st.write("""
        - Example 1: What company had the highest long-term debt in 2010?
        - Example 2: What is the average shares outstanding for all companies in the Computer & Technology Sector from 2017?
        - Example 3: Query the tickers and return on investment for companies with annual reports (per_type = 'A') and a return on investment greater than 10.
        """)

        # User question input
        with st.form(key='user_question_form'):
            user_question = st.text_input("Enter your question:")
            submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            if is_finance_related(user_question):
                # Add user issues to the session
                st.session_state["conversation"].append({"role": "user", "content": user_question})

                # Generate SQL queries
                sql_query = generate_sql(user_question)
                st.subheader("Generated SQL Query")
                st.code(sql_query, language="sql")

                # Load form
                tables_files = {
                    "t_zacks_fc": "t_zacks_fc.parquet",
                    "t_zacks_fr": "t_zacks_fr.parquet",
                    "t_zacks_mktv": "t_zacks_mktv.parquet",
                    "t_zacks_shrs": "t_zacks_shrs.parquet",
                    "t_zacks_sectors": "t_zacks_sectors.csv"
                }
                tables = load_tables(tables_files)

                # Execute SQL and handle errors
                results = execute_sql(sql_query, tables)
                attempt = 1
                max_attempts = 3
                while isinstance(results, str) and attempt <= max_attempts:
                    # Error encountered, regenerate query
                    error_message = results
                    st.error(f"Error executing query: {error_message}")
                    sql_query = generate_sql(user_question, error_message=error_message)
                    st.subheader(f"Re-generated SQL Query (Attempt {attempt})")
                    st.code(sql_query, language="sql")
                    results = execute_sql(sql_query, tables)
                    attempt += 1

                # Show results
                if isinstance(results, pd.DataFrame) and not results.empty:
                    st.subheader("Query Results")
                    st.dataframe(results)

                    next_step = st.radio(
                        "Based on the SQL query I have generated, would you like to proceed further or modify the query first?",
                        options=["Proceed Further", "Modify Query"]
                    )

                    if next_step == "Modify Query":
                        modification_request = st.text_area("Kindly describe the specific changes you'd like to make to the existing SQL query:")
                        if st.button("Submit Changes"):
                            modified_sql = modify_sql_query(sql_query, modification_request)
                            st.subheader("Modified SQL Query")
                            st.code(modified_sql, language="sql")

                            # Re-execute the modified query
                            results = execute_sql(modified_sql, tables)
                            if isinstance(results, pd.DataFrame) and not results.empty:
                                st.subheader("Modified Query Results")
                                st.dataframe(results)
                            else:
                                st.error(f"Error with the modified query: {results}")

                            # Update the current SQL query
                            sql_query = modified_sql

                    # Save results and issues to session state
                    st.session_state['results'] = results
                    st.session_state['user_question'] = user_question
                    st.session_state['analysis_generated'] = False

                    # Unique identifiers for preservation analyses
                    question_hash = hashlib.md5(user_question.encode()).hexdigest()

                    # Check if analyses are already available
                    if user_question in st.session_state['analyses']:
                        st.session_state['analysis_generated'] = True

                        st.subheader("Data Analysis")
                        st.write(st.session_state['analyses'][user_question])

                        # Generate PDF buffer
                        pdf_buffer = save_analysis_as_pdf(user_question, st.session_state['analyses'][user_question])
                        st.download_button(
                            label="Download Analysis as PDF",
                            data=pdf_buffer,
                            file_name="equity_analysis.pdf",
                            mime="application/pdf",
                            key=f"download_analysis_{question_hash}"
                        )

                        # Generate full dialogue PDF buffer
                        full_pdf_buffer = save_full_conversation_as_pdf(st.session_state["conversation"])
                        st.download_button(
                            label="Download Full Analysis as PDF",
                            data=full_pdf_buffer,
                            file_name="full_equity_analysis.pdf",
                            mime="application/pdf",
                            key=f"download_full_conversation_{question_hash}"
                        )
                    else:
                        # No analysis exists, then wait for user to request analysis
                        pass
                else:
                    st.error(f"Error executing query after {max_attempts} attempts: {results}")
            else:
                st.write("Hmm, I didn’t get that. Try asking me questions regarding company financials or other metrics as part of Quandl's Zacks Fundamentals Collection A (ZFA) data feed.")

        # Data analysis component
        if 'results' in st.session_state and 'user_question' in st.session_state:
            results = st.session_state['results']
            user_question = st.session_state['user_question']

            if user_question in st.session_state['analyses']:
                st.subheader("Data Analysis")
                st.write(st.session_state['analyses'][user_question])

                # Generate PDF buffer
                pdf_buffer = save_analysis_as_pdf(user_question, st.session_state['analyses'][user_question])
                question_hash = hashlib.md5(user_question.encode()).hexdigest()
                st.download_button(
                    label="Download Analysis as PDF",
                    data=pdf_buffer,
                    file_name="equity_analysis.pdf",
                    mime="application/pdf",
                    key=f"download_analysis_{question_hash}"
                )

                # Generate full dialogue PDF buffer
                full_pdf_buffer = save_full_conversation_as_pdf(st.session_state["conversation"])
                st.download_button(
                    label="Download Full Analysis as PDF",
                    data=full_pdf_buffer,
                    file_name="full_equity_analysis.pdf",
                    mime="application/pdf",
                    key=f"download_full_conversation_{question_hash}"
                )
            else:
                analyze = st.radio("Would you like an in-depth analysis of this queried data?", ("No", "Yes"), key='analyze')
                if analyze == "Yes" and not st.session_state.get('analysis_generated', False):
                    st.subheader("Data Analysis")
                    analysis = analyze_data(user_question, results)
                    st.write(analysis)

                    st.session_state['analyses'][user_question] = analysis
                    st.session_state['analysis_generated'] = True

                    st.session_state["conversation"].append({"role": "assistant", "content": analysis})

                    # Generate PDF buffer
                    pdf_buffer = save_analysis_as_pdf(user_question, analysis)
                    question_hash = hashlib.md5(user_question.encode()).hexdigest()
                    st.download_button(
                        label="Download Analysis as PDF",
                        data=pdf_buffer,
                        file_name="equity_analysis.pdf",
                        mime="application/pdf",
                        key=f"download_analysis_{question_hash}"
                    )

                    # Generate full dialogue PDF buffer
                    full_pdf_buffer = save_full_conversation_as_pdf(st.session_state["conversation"])
                    st.download_button(
                        label="Download Full Analysis as PDF",
                        data=full_pdf_buffer,
                        file_name="full_equity_analysis.pdf",
                        mime="application/pdf",
                        key=f"download_full_conversation_{question_hash}"
                    )

    # Dialogue with history
    if st.session_state["conversation"]:
        st.subheader("Conversation History")
        with st.expander("Show Conversation History", expanded=False):
            for idx, msg in enumerate(st.session_state["conversation"]):
                if msg["role"] == "user":
                    st.markdown(f"**User:** {msg['content']}")
                else:
                    st.markdown(f"**Assistant:** {msg['content']}")

            # Generate full dialogue PDF buffer
            # Use of dialogue length as part of the unique identifier
            conversation_length = len(st.session_state["conversation"])
            st.download_button(
                label="Download Full Analysis as PDF",
                data=save_full_conversation_as_pdf(st.session_state["conversation"]),
                file_name="full_equity_analysis.pdf",
                mime="application/pdf",
                key=f"download_full_conversation_{conversation_length}"
            )
