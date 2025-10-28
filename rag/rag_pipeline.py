
import openai
import os
from typing import List, Dict, Any, Tuple
"""

"""
# Set your OpenAI API key
with open("src/openai_key.txt", "r") as f:
    openai.api_key = f.read().strip()  # strip /n

class BuffettRAG:
    def __init__(self, letters_data: Dict[str, Any], term_index: Dict[str, Dict],
                 company_sector_index: Dict[str, List]):
        """
        Args:
            letters_data (Dict): Dict mapping letter IDs to letter content
            term_index (Dict): Index of financial terms and their locations
            company_sector_index (Dict): Index of sectors, companies, and their locations
        """
        self.letters = letters_data
        self.term_index = term_index
        self.company_sector_index = company_sector_index

    def extract_paragraphs_from_letter(self, letter_id: str, paragraph_idx: List[int]) -> List[str]:
        """
        Extract specific paragraphs from a letter
        
        Returns:
            List[str]: The extracted paragraphs
        """
        if letter_id not in self.letters:
            return []

        paragraphs = self.letters[letter_id]["paragraphs"]

        if not paragraphs:
            return []

        extracted = []
        for idx in paragraph_idx:
            extracted.append(paragraphs[idx])

        return extracted

    def find_relevant_indicator_context(self, question: str) -> Tuple[List[str], str]:
        """
        Find financial indicators relevant to the question and extract context

        Args:
            question (str): The question to analyze

        Returns:
            Tuple[List[str], str]: (List of indicator terms found, Combined paragraph context)
        """
        question_lower = question.lower()
        relevant_indicators = []
        all_paragraphs = []

        # Check each indicator term against the question
        for term, locations in self.term_index.items():
            if term.lower() in question_lower:
                relevant_indicators.append(term)
                # Extract paragraphs for each occurrence
                for location_dict in locations:  # locations is a list of dicts
                    for letter_id, paragraph_idx in location_dict.items():  # each dict has one key-value pair
                        # paragraph_idx is a single integer (not a list)
                        paragraphs = self.extract_paragraphs_from_letter(letter_id, [paragraph_idx])  # wrap in list
                        for p in paragraphs:
                            if p not in all_paragraphs:  # Avoid duplicates
                                all_paragraphs.append(p)

        # Combine paragraphs into context
        context = "\n\n".join(
            all_paragraphs[-2:] if len(' '.join(all_paragraphs[-2:]).split(" ")) >= 200 else all_paragraphs[-3:])

        return relevant_indicators, context

    def find_relevant_industry_context(self, question: str) -> Tuple[List[Dict], str]:
        """
        Find industries and companies relevant to the question and extract context

        Args:
            question (str): The question to analyze

        Returns:
            Tuple[List[Dict], str]: (List of sector/company info, Combined paragraph context)
        """
        question_lower = question.lower()
        relevant_sectors = []
        all_paragraphs = []

        # Check each sector against the question
        for sector, companies in self.company_sector_index.items():
            sector_relevant = sector.lower() in question_lower

            for company, locations in companies.items():
                company_relevant = company.lower() in question_lower
                # If either sector or company is relevant
                if sector_relevant or company_relevant:
                    relevant_sectors.append({
                        "sector": sector,
                        "company": company})
                    # Extract paragraphs
                for letter_id, paragraph_indices in locations.items():
                    paragraphs = self.extract_paragraphs_from_letter(letter_id, paragraph_indices)
                    for p in paragraphs:
                        if p not in all_paragraphs:  # Avoid duplicates
                            all_paragraphs.append(p)

        # Combine paragraphs into context
        context = "\n\n".join(
            all_paragraphs[-2:] if len(' '.join(all_paragraphs[-2:]).split(" ")) >= 200 else all_paragraphs[-3:])

        return relevant_sectors, context

    def refine_answer(self, question: str, initial_answer: str) -> str:
        """
        Refine the initial answer to improve quality and authenticity.

        Args:
            question (str): The original question
            initial_answer (str): The initial answer to refine

        Returns:
            str: Refined answer in Warren Buffett's voice
        """
        # Find relevant indicator context

        # Find relevant indicator context
        indicators, indicator_context = self.find_relevant_indicator_context(question)

        # Find relevant industry context
        industries, industry_context = self.find_relevant_industry_context(question)

        prompt = f"""
         
         Question: {question}

        Current Answer: {initial_answer}
        

        Following this structure:

        1. Refine Current Answer with a clear and direct financial response to the question, grounded in th {indicators} and {industries} mentioned if {indicators} or {industries} not null. 

        2. If {indicators} and {industries} are null, ignore this rule, else focus only on the key {indicators} and {industries} in the question. Ignore any unrelated financial terms or companies in the context. can mention meaningful financial drivers (e.g., CapEx intensity, interest rates, free cash flow, etc.) when relevant

        3. Conclude with a brief, insightful takeaway that reflects a timeless financial or business principle. If appropriate, connect it to broader Buffett principles — such as preserving optionality, maintaining “dry powder,” or the value of conservatism in capital structure. Do not give any investment advice or suggest buy/sell actions.

        Keep the tone direct, thoughtful, and conversational — like Buffett’s shareholder letters. 
        Use simple, memorable language with occasional analogies or metaphors, but avoid excessive storytelling. 
        Prioritize clarity, insight, and density of meaning over elaborate metaphor or storytelling.
        Limit the response to around 150–160 words."""


        # Call OpenAI API
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )

        return response.choices[0].message.content.strip()

    def add_real_world_examples(self, question: str, answer: str) -> str:
        """
        Enhance the answer with relevant real-world examples

        Args:
            question (str): The original question
            answer (str): The answer to enhance

        Returns:
            str: Enhanced answer with real-world examples
        """
        # Find relevant indicator context
        indicators, indicator_context = self.find_relevant_indicator_context(question)

        # Find relevant industry context
        industries, industry_context = self.find_relevant_industry_context(question)

        # If no relevant context found, return the original answer
        if not indicator_context and not industry_context:
            return answer

        # Prepare prompt for adding examples
        prompt = f"""
                    The question is: {question}
                    Relevant indicator information from Buffett's letters:
                    {indicator_context}

                    Relevant historical examples from Buffett's investments:
                    {industry_context}

                    Instructions:

                    Revise and enhance the current answer in Warren Buffett’s authentic voice, following this structure:

                    1. Begin with a clear and direct financial response to the question, grounded in the {indicators} and {industries} mentioned. Focus only on the key {indicators} and {industries} in the question. Ignore any unrelated financial terms or companies in the context. can mention meaningful financial drivers (e.g., CapEx intensity, interest rates, free cash flow, etc.) when relevant

                    2. Smoothly incorporate one or more historical examples from the provided companies or sectors, showing how Buffett evaluated similar situations. Do not say “for example” or “such as” — instead, tell it like a story or reflection.

                    3. Conclude with a brief, insightful takeaway that reflects a timeless financial or business principle. If appropriate, connect it to broader Buffett principles — such as preserving optionality, maintaining “dry powder,” or the value of conservatism in capital structure. Do not give any investment advice or suggest buy/sell actions.

                    Keep the tone direct, thoughtful, and conversational — like Buffett’s shareholder letters. 
                    Use simple, memorable language with occasional analogies or metaphors, but avoid excessive storytelling. 
                    Prioritize clarity, insight, and density of meaning over elaborate metaphor or storytelling.
                    Limit the response to around 150–160 words."""

        # Call OpenAI API
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )

        return response.choices[0].message.content.strip()

    def get_refined_answer(self, question: str, initial_answer: str = "") -> str:
        """
        Complete pipeline to process a question - refine initial answer

        Args:
            question (str): The question to answer
            initial_answer (str): Optional initial answer to refine

        Returns:
            str: Final answer with refinements
        """
        # If no initial answer provided, generate one
        if initial_answer=="":
            initial_prompt = f"""Please provide a brief, factual answer to this question: {question} as follwing requirment:
                              1. Begin with a direct, clear response to the question that shows deep financial understanding. .
                             Instructions:

                    Revise and enhance the current answer in Warren Buffett’s authentic voice, following this structure:

                    1. Begin with a clear and direct financial response to the question.
                    2. Conclude with a brief, insightful takeaway that reflects a timeless financial or business principle. If appropriate, connect it to broader Buffett principles — such as preserving optionality, maintaining “dry powder,” or the value of conservatism in capital structure. Do not give any investment advice or suggest buy/sell actions.

                    Keep the tone direct, thoughtful, and conversational — like Buffett’s shareholder letters. 
                    Use simple, memorable language with occasional analogies or metaphors, but avoid excessive storytelling. 
                    Prioritize clarity, insight, and density of meaning over elaborate metaphor or storytelling.
                    Limit the response to around 150–160 words."""
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": initial_prompt}],
                temperature=0.3
            )
            initial_answer = response.choices[0].message.content.strip()

        # # Step 1: Refine the answer
        return self.refine_answer(question, initial_answer)

    def get_real_world_examples(self, question: str, refine_answer: str = "") -> str:
        """
        Complete pipeline to process a question - add examples to initial answer and

        Args:
            question (str): The question to answer

        Returns:
            str: Final answer with refinements and examples
        """
        # If no initial answer provided, generate one
        if not refine_answer:
            initial_prompt = f"Please provide a brief, factual answer to this question: {question}"
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": initial_prompt}],
                temperature=0.3
            )
            refine_answer = response.choices[0].message.content.strip()

        # Step 2: Add real-world examples
        final_answer = self.add_real_world_examples(question, refine_answer)

        return final_answer