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
import re  # For regex operations

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

    def generate_sql(user_question, sector_code=None, error_message=None):
        """
        Generates an SQL query from a natural language question using OpenAI's GPT.
        Includes error correction logic if the previous query failed.

        Args:
            user_question (str): The natural language question from the user.
            sector_code (int, optional): The sector code to use in the SQL query.
            error_message (dict, optional): Dictionary containing 'sql' and 'error' from the previous attempt, if applicable.

        Returns:
            str: Generated SQL query.
        """
        error_prompt = (
            f"\nThe previous SQL query was:\n```sql\n{error_message['sql']}\n```\n"
            f"Executing it resulted in the following error:\n{error_message['error']}\n"
            "Please correct the SQL query based on the error above."
            if error_message else ""
        )
        sector_prompt = f" AND fc.zacks_sector_code = {sector_code}" if sector_code else ""

        prompt = f"""
Pretend you are an expert at converting natural language questions into accurate SQL queries. Please generate an accurate SQL query based on
the following natural language question and database schema provided below. Think sequentially and refer to the sample natural language
questions with correct and incorrect outputs as well.

Database Schema:
Table 1: t_zacks_fc (This table contains fundamental indicators for companies)
Columns: 'ticker' = Unique zacks Identifier for each company/stock, ticker or trading symbol, 'comp_name' = Company name, 'exchange' = Exchange
traded, 'per_end_date' = Period end date which represents quarterly data, 'per_type' = Period type (eg. Q for quarterly data), 'filing_date' =
Filing date, 'filing_type' = Filing type: 10-K, 10-Q, PRELIM, 'zacks_sector_code' = Zacks sector code (Numeric Value eg. 10 = Technology),
'eps_diluted_net_basic' = Earnings per share (EPS) net (Company's net earnings or losses attributable to common shareholders per basic share
basis), 'lterm_debt_net_tot' = Net long-term debt (The net amount of long term debt issued and repaid. This field is either calculated as the
sum of the long term debt fields or used if a company does not report debt issued and repaid separately).
Keys: ticker, per_end_date, per_type

Table 2: t_zacks_fr (This table contains fundamental ratios for companies)
Columns: 'ticker' = Unique zacks Identifier for each company/stock, ticker or trading symbol, 'per_end_date' = Period end date which represents
quarterly data, 'per_type' = Period type (eg. Q for quarterly data), 'ret_invst' = Return on investments (An indicator of how profitable a
company is relative to its assets invested by shareholders and long-term bond holders. Calculated by dividing a company's operating earnings by
its long-term debt and shareholders equity), 'tot_debt_tot_equity' = Total debt / total equity (A measure of a company's financial leverage
calculated by dividing its long-term debt by stockholders' equity).
Keys: ticker, per_end_date, per_type.

Table 3: t_zacks_mktv (This table contains market value data for companies)
Columns: 'ticker' = Unique zacks Identifier for each company/stock, ticker or trading symbol, 'per_end_date' = Period end date which represents
quarterly data, 'per_type' = Period type (eg. Q for quarterly data), 'mkt_val' = Market Cap of Company (shares out x last monthly price per
share - unit is in Millions).
Keys: ticker, per_end_date, per_type.

Table 4: t_zacks_shrs (This table contains shares outstanding data for companies)
Columns: 'ticker' = Unique zacks Identifier for each company/stock, ticker or trading symbol, 'per_end_date' = Period end date which represents
quarterly data, 'per_type' = Period type (eg. Q for quarterly data), 'shares_out' = Number of Common Shares Outstanding from the front page of
10K/Q.
Keys: ticker, per_end_date, per_type.

Table 5: t_zacks_sectors (This table contains the zacks sector codes and their corresponding sectors)
Columns: 'zacks_sector_code' = Unique identifier for each zacks sector, 'sector' = The sector descriptions that correspond to the sector code 
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
        Executes a given SQL query on in-memory SQLite database populated with provided tables.

        Args:
            sql (str): The SQL query to execute.
            tables (dict): Dictionary where keys are table names and values are pandas DataFrames.

        Returns:
            pd.DataFrame or dict: Query results as a DataFrame or a dictionary with 'sql' and 'error' keys if execution fails.
        """
        try:
            conn = sqlite3.connect(":memory:")  # Use in-memory database
            for table_name, df in tables.items():
                df.to_sql(table_name, conn, index=False, if_exists="replace")
            result = pd.read_sql_query(sql, conn)
            return result
        except Exception as e:
            return {"sql": sql, "error": str(e)}
        finally:
            conn.close()


    def modify_sql_query(current_sql, modification_request):
        """
        Modifies the given SQL query based on user-provided feedback using GPT.

        Args:
            current_sql (str): The current SQL query to be modified.
            modification_request (str): User's input describing the required changes to the query.

        Returns:
            str: Modified SQL query.
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
        Analyzes SQL query results using OpenAI GPT, providing equity analysis insights.

        Args:
            user_question (str): The original user query.
            data (pd.DataFrame): DataFrame containing SQL query results.

        Returns:
            str: Analytical insights in text format.
        """
        table_md = data.to_markdown(index=False)
        prompt = f"""
I have executed a SQL query based on the following user question and obtained the data below.

User's Question:
{user_question}

Data Table:
{table_md}

Pretend you are an experienced equity analyst working in the banking industry. Analyze this data in the style of an expert equity analyst, highlighting major trends, comparing competitor companies/sectors, analyzing the significance of metrics, and noting any other interesting insights regarding this data.

Provide 3-4 in-depth paragraphs with your analysis. Ensure the output contains only plain text. Do NOT use formatting such as italicizing, bold, underscores, or any special symbols. Write clean, clear paragraphs with no extra formatting.
        """
        response = openai.ChatCompletion.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )

        # Cleanup: Strip unwanted formatting
        cleaned_analysis = response.choices[0].message["content"].strip()
        # Remove any unintended single newlines between characters (common cause of broken text)
        cleaned_analysis = re.sub(r'(?<!\s)\n(?!\s)', ' ', cleaned_analysis)
        # Replace multiple newlines with double newlines for paragraph separation
        cleaned_analysis = re.sub(r'\n{2,}', '\n\n', cleaned_analysis)
        return cleaned_analysis


    def save_analysis_as_pdf(user_question, analysis_text):
        """
        Creates a formatted PDF report of the analysis.

        Args:
            user_question (str): The original user question.
            analysis_text (str): Analysis text to include in the PDF.

        Returns:
            BytesIO: Buffer containing the PDF file.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        # Add title, question, and analysis to the PDF
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

        Args:
            conversation (list): List of dictionaries with conversation entries.

        Returns:
            BytesIO: Buffer containing the PDF file.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("Full Equity Analysis Report", styles['Title']))
        elements.append(Spacer(1, 12))

        # Add conversation history to PDF
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

        Args:
            files (dict): Dictionary of table names and corresponding file paths.

        Returns:
            dict: Dictionary of table names and DataFrame objects.
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
        Determines if a user question is related to the chatbot's functionality by utilizing GPT.

        Args:
            question (str): The user's input question.

        Returns:
            bool: True if GPT determines the question is related to the chatbot's functionality, otherwise False.
        """
        prompt = f"""
Hi there. My name is BoFA-nanza, and I am your personal AI chatbot. You can ask me questions regarding:
- Company fundamentals
- Financial ratios
- Market values
- Shares outstanding
- Sector classifications for all US equities dating back until 2006

I have sourced this data from NASDAQ's premier publisher of alternative data for institutional investors, Quandl. Specifically, I have access to The Zacks Fundamentals Collection A (ZFA) data feed.

Database Schema:
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

Example Questions:
- What company had the highest long-term debt in 2010?
- What is the average shares outstanding for all companies in the Computer & Technology Sector from 2017?
- Query the tickers and return on investment for companies with annual reports (per_type = 'A') and a return on investment greater than 10.

Given this context, determine if the following question is related to my functionality:
{question}

Please respond with either "Yes" if it is related or "No" if it is not, without any additional explanation.
        """
        response = openai.ChatCompletion.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0
        )
        return response.choices[0].message["content"].strip().lower() == "yes"


    def extract_sector_name(user_question):
        """
        Extracts the sector name from the user's question using OpenAI GPT.

        Args:
            user_question (str): The user's input question.

        Returns:
            str: Extracted sector name or None if not found.
        """
        prompt = f"""
Analyze the following user question and extract the sector name mentioned, if any.

User Question:
{user_question}

Respond with only the sector name in plain text. If no sector is mentioned, respond with "None".
        """
        response = openai.ChatCompletion.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        sector = response.choices[0].message["content"].strip()
        return sector if sector.lower() != "none" else None


    def get_sector_code(sector_name, tables):
        """
        Retrieves the sector code for a given sector name from the t_zacks_sectors table.

        Args:
            sector_name (str): The name of the sector.
            tables (dict): Dictionary of table names and DataFrame objects.

        Returns:
            int or None: The sector code if found, else None.
        """
        sectors_df = tables.get("t_zacks_sectors")
        if sectors_df is not None:
            sector_row = sectors_df[sectors_df['sector'].str.lower() == sector_name.lower()]
            if not sector_row.empty:
                return sector_row.iloc[0]['zacks_sector_code']
        return None


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
        st.markdown("""
**Table 1: t_zacks_fc**
- **Columns**: ticker, comp_name, exchange, per_end_date, per_type, filing_date, filing_type, zacks_sector_code, eps_diluted_net_basic, lterm_debt_net_tot

**Table 2: t_zacks_fr**
- **Columns**: ticker, per_end_date, per_type, ret_invst, tot_debt_tot_equity

**Table 3: t_zacks_mktv**
- **Columns**: ticker, per_end_date, per_type, mkt_val

**Table 4: t_zacks_shrs**
- **Columns**: ticker, per_end_date, per_type, shares_out

**Table 5: t_zacks_sectors**
- **Columns**: zacks_sector_code, sector
    """)

    # Prompt the user to continue
    if "continue" not in st.session_state:
        continue_input = st.text_input("If you are willing to proceed, please type 'yes' to continue:").strip().lower()
        if continue_input == "yes":
            st.session_state["continue"] = True

    if "conversation" not in st.session_state:
        st.session_state["conversation"] = []

    if "analyses" not in st.session_state:
        st.session_state["analyses"] = {}

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Main Interaction
    if st.session_state.get("continue"):
        st.subheader("ChatBot Examples")
        st.markdown("""
- **Example 1**: What company had the highest long-term debt in 2010?
- **Example 2**: What is the average shares outstanding for all companies in the Computer & Technology Sector from 2017?
- **Example 3**: Query the tickers and return on investment for companies with annual reports (per_type = 'A') and a return on investment greater than 10.
""")

        # User question input
        with st.form(key='user_question_form'):
            user_question = st.text_input("Enter your question:")
            submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            if is_finance_related(user_question):
                # Add user question to the conversation
                st.session_state["conversation"].append({"role": "user", "content": user_question})

                # Initialize list to store all SQL attempts
                sql_attempts = []

                # Load tables
                tables_files = {
                    "t_zacks_fc": "t_zacks_fc.parquet",
                    "t_zacks_fr": "t_zacks_fr.parquet",
                    "t_zacks_mktv": "t_zacks_mktv.parquet",
                    "t_zacks_shrs": "t_zacks_shrs.parquet",
                    "t_zacks_sectors": "t_zacks_sectors.csv"
                }
                tables = load_tables(tables_files)

                # Extract sector name
                sector_name = extract_sector_name(user_question)
                sector_code = get_sector_code(sector_name, tables) if sector_name else None

                # if sector_name and sector_code:
                #     st.info(f"Identified Sector: {sector_name} (Code: {sector_code})")
                # elif sector_name and not sector_code:
                #     st.warning(f"Sector '{sector_name}' not found in the sectors table.")
                # else:
                #     st.info("No specific sector identified in the question.")

                # Attempt counter
                attempt = 1
                max_attempts = 3
                success = False

                while attempt <= max_attempts and not success:
                    if attempt == 1:
                        # Generate initial SQL query with sector code if available
                        sql_query = generate_sql(user_question, sector_code=sector_code)
                    else:
                        # Generate SQL based on previous error
                        previous_error = sql_attempts[-1]['error']
                        sql_query = generate_sql(user_question, sector_code=sector_code, error_message={"sql": sql_attempts[-1]['sql'], "error": previous_error})

                    # Execute SQL
                    results = execute_sql(sql_query, tables)

                    if isinstance(results, pd.DataFrame):
                        if not results.empty:
                            # Successful execution
                            sql_attempts.append({"attempt": attempt, "sql": sql_query, "result": results, "error": None})
                            success = True
                        else:
                            # Empty result set, treat as failure
                            sql_attempts.append({"attempt": attempt, "sql": sql_query, "result": results, "error": "Empty result set."})
                    else:
                        # Execution failed, capture error
                        sql_attempts.append({"attempt": attempt, "sql": sql_query, "result": None, "error": results['error']})

                    attempt += 1

                # Display all SQL attempts
                st.subheader("SQL Query Execution Results")
                for attempt_entry in sql_attempts:
                    st.markdown(f"**Attempt {attempt_entry['attempt']} SQL Query:**")
                    st.code(attempt_entry['sql'], language="sql")
                    if isinstance(attempt_entry['result'], pd.DataFrame):
                        if not attempt_entry['result'].empty:
                            st.markdown(f"**Result for Attempt {attempt_entry['attempt']}:**")
                            st.dataframe(attempt_entry['result'])
                        else:
                            st.warning(f"Attempt {attempt_entry['attempt']} returned an empty result.")
                    elif attempt_entry['error']:
                        st.markdown(f"**Error for Attempt {attempt_entry['attempt']}:** {attempt_entry['error']}")
                        st.write("---")

                # Determine if any attempt was successful
                successful_attempt = next((att for att in sql_attempts if isinstance(att['result'], pd.DataFrame) and not att['result'].empty), None)

                if successful_attempt:
                    st.success(f"SQL query executed successfully on Attempt {successful_attempt['attempt']}.")
                    st.session_state['results'] = successful_attempt['result']
                    st.session_state['user_question'] = user_question
                    st.session_state['analysis_generated'] = False

                    # Add only the successful SQL and result to the conversation
                    # Removed the following line to exclude SQL from conversation history:
                    # st.session_state["conversation"].append({"role": "assistant", "content": f"Executed SQL on Attempt {successful_attempt['attempt']}:\n{successful_attempt['sql']}"})

                    # Add option to modify the SQL query
                    modify_choice = st.radio(
                        "Would you like to modify the generated SQL query?",
                        ("No", "Yes"),
                        key="modify_choice"
                    )

                    if modify_choice == "Yes":
                        with st.form(key='modify_sql_form'):
                            modification_request = st.text_input("Please describe how you'd like to modify the SQL query:")
                            modify_submit = st.form_submit_button(label='Modify SQL Query')

                        if modify_submit:
                            if modification_request.strip() == "":
                                st.warning("Please provide a description for the modification.")
                            else:
                                # Generate modified SQL query
                                with st.spinner('Modifying SQL query...'):
                                    modified_sql = modify_sql_query(successful_attempt['sql'], modification_request)
                                
                                # Execute the modified SQL query
                                modified_results = execute_sql(modified_sql, tables)

                                # Add the modification request to the conversation
                                st.session_state["conversation"].append({"role": "user", "content": f"I'd like to modify the SQL query: {modification_request}"})
                                # Removed the following line to exclude modified SQL from conversation history:
                                # st.session_state["conversation"].append({"role": "assistant", "content": f"Modified SQL Query:\n{modified_sql}"})

                                if isinstance(modified_results, pd.DataFrame):
                                    if not modified_results.empty:
                                        st.success("Modified SQL query executed successfully.")
                                        st.session_state['results'] = modified_results
                                        st.session_state['user_question'] = user_question
                                        st.session_state['analysis_generated'] = False

                                        # Add the data analysis only to the conversation
                                        analysis = analyze_data(user_question, modified_results)
                                        st.session_state["conversation"].append({"role": "assistant", "content": analysis})

                                        # Display the modified SQL and its results
                                        st.markdown("**Modified SQL Query:**")
                                        st.code(modified_sql, language="sql")

                                        st.markdown("**Results for Modified SQL Query:**")
                                        st.dataframe(modified_results)
                                    else:
                                        st.warning("The modified SQL query returned an empty result set.")
                                else:
                                    st.error(f"Error executing modified SQL query: {modified_results['error']}")
                else:
                    st.error("Failed to execute SQL query after maximum attempts.")
            else:
                st.warning("Hmm, I didn’t get that. Try asking me questions regarding company financials or other metrics as part of Quandl's Zacks Fundamentals Collection A (ZFA) data feed.")

        # Data analysis component
        if 'results' in st.session_state and 'user_question' in st.session_state:
            results = st.session_state['results']
            user_question = st.session_state['user_question']

            if user_question in st.session_state['analyses']:
                st.subheader("Data Analysis")
                st.text(st.session_state['analyses'][user_question])

                # Generate PDF buffer
                pdf_buffer = save_analysis_as_pdf(user_question, st.session_state['analyses'][user_question])
                st.download_button(
                    label="Download Analysis as PDF",
                    data=pdf_buffer,
                    file_name="equity_analysis.pdf",
                    mime="application/pdf",
                    key=f"download_analysis_{hashlib.md5(user_question.encode()).hexdigest()}"
                )

                # Generate full dialogue PDF buffer
                full_pdf_buffer = save_full_conversation_as_pdf(st.session_state["conversation"])
                st.download_button(
                    label="Download Full Analysis as PDF",
                    data=full_pdf_buffer,
                    file_name="full_equity_analysis.pdf",
                    mime="application/pdf",
                    key=f"download_full_conversation_{hashlib.md5(user_question.encode()).hexdigest()}"
                )
            else:
                analyze = st.radio("Would you like an in-depth analysis of this queried data?", ("No", "Yes"), key='analyze')
                if analyze == "Yes" and not st.session_state.get('analysis_generated', False):
                    st.subheader("Data Analysis")
                    analysis = analyze_data(user_question, results)
                    st.text(analysis)

                    st.session_state['analyses'][user_question] = analysis
                    st.session_state['analysis_generated'] = True

                    # Add only the data analysis to the conversation
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
            for msg in st.session_state["conversation"]:
                if msg["role"] == "user":
                    st.markdown("**User:**")
                    st.text(msg["content"])
                else:
                    st.markdown("**Assistant:**")
                    st.text(msg["content"])
                # st.write("---")  # Adds a horizontal rule between messages

            # Generate full dialogue PDF buffer
            # Use of conversation length as part of the unique identifier
            conversation_length = len(st.session_state["conversation"])
            st.download_button(
                label="Download Full Analysis as PDF",
                data=save_full_conversation_as_pdf(st.session_state["conversation"]),
                file_name="full_equity_analysis.pdf",
                mime="application/pdf",
                key=f"download_full_conversation_{conversation_length}"
            )
