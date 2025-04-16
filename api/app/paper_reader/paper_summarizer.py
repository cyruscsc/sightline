from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import Document
from pydantic import BaseModel, Field

class PaperSummary(BaseModel):
    """Structure for the paper summary output."""

    title: str = Field(description="The title of the paper")
    authors: list[str] = Field(description="List of paper authors")
    abstract: str = Field(description="A concise summary of the paper's abstract")
    key_points: list[str] = Field(description="Key points and findings from the paper")
    methodology: str = Field(description="Description of the methodology used")
    results: str = Field(description="Main results and conclusions")
    implications: str = Field(
        description="Implications and potential impact of the research"
    )


class PaperSummarizer:
    def __init__(self):
        """
        Initialize the PaperSummarizer.
        """
        self._llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
        self._section_output_parser = StrOutputParser()
        self._overall_output_parser = PydanticOutputParser(pydantic_object=PaperSummary)
        self._section_prompt_template = self._create_section_prompt_template()
        self._overall_prompt_template = self._create_overall_prompt_template()

    def _create_section_prompt_template(self) -> ChatPromptTemplate:
        """
        Create the prompt template for section summarization.

        Returns:
            ChatPromptTemplate: The formatted prompt template
        """
        system_template = """You are an expert at analyzing sections of academic papers. Your task is to analyze a specific section of a paper and create a focused summary that captures the key information from that section."""

        human_template = """Please analyze the following section of the paper and create a focused summary.

Paper Title: {title}
Section Content:
{content}

Please provide a brief summary of this section, focusing on:
1. Main ideas and arguments presented
2. Key findings or methodological details
3. How this section contributes to the overall paper
4. Any significant equations, results, or conclusions

Make the summary clear and concise while preserving all important technical details."""

        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        return ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    def _create_overall_prompt_template(self) -> ChatPromptTemplate:
        """
        Create the prompt template for paper summarization.

        Returns:
            ChatPromptTemplate: The formatted prompt template
        """
        system_template = """You are an expert at summarizing academic papers. Your task is to analyze academic papers and create comprehensive, well-structured summaries that capture the key aspects of the research."""

        human_template = """Please analyze the following paper and create a comprehensive summary.

Paper Title: {title}
Authors: {authors}
Abstract: {abstract}

Summaries of All Sections:
{section_summaries}

Please provide a detailed summary following this structure:
{format_instructions}

Focus on:
1. Capturing the main contributions and findings
2. Explaining the methodology clearly
3. Highlighting key results and their significance
4. Discussing the implications of the research

Make the summary clear, concise, and well-structured."""

        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template
        )
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        return ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

    def _prepare_section_prompt_inputs(self, paper_details: dict, document: Document) -> dict:
        """
        Prepare inputs for the section prompt template.

        Args:
            paper_details (Dict): Paper data from ArXivPaper.get_paper_data()
            document (Document): The part of the paper to be summarized

        Returns:
            Dict: Formatted inputs for the prompt template
        """
        return {
            "title": paper_details["title"],
            "content": document.page_content,
        }

    def _prepare_overall_prompt_inputs(self, paper_details: dict, section_summaries: list[str]) -> dict:
        """
        Prepare inputs for the overall prompt template.

        Args:
            paper_details (Dict): Paper data from ArXivPaper.get_paper_data()
            section_summaries (list[str]): List of section summaries

        Returns:
            Dict: Formatted inputs for the prompt template
        """
        return {
            "title": paper_details["title"],
            "authors": ", ".join(paper_details["authors"]),
            "abstract": paper_details["abstract"],
            "section_summaries": "\n\n".join(section_summaries),
            "format_instructions": self._overall_output_parser.get_format_instructions(),
        }

    def generate_summary(self, paper_data: dict) -> dict:
        """
        Generate a markdown summary of the paper.

        Args:
            paper_data (Dict): Paper data from ArXivPaper.get_paper_data()

        Returns:
            str: Markdown-formatted summary of the paper
        """
        if len(paper_data["documents"]) > 1:
            section_summaries = []
            section_chain = self._section_prompt_template | self._llm | self._section_output_parser

            for doc in paper_data["documents"]:
                section_prompt_inputs = self._prepare_section_prompt_inputs(paper_data["details"], doc)
                section_summary = section_chain.invoke(section_prompt_inputs)
                section_summaries.append(section_summary)

            overall_prompt_inputs = self._prepare_overall_prompt_inputs(paper_data["details"], section_summaries)
        
        else:
            overall_prompt_inputs = self._prepare_overall_prompt_inputs(paper_data["details"], [paper_data["documents"][0].page_content])

        overall_chain = self._overall_prompt_template | self._llm | self._overall_output_parser
        summary = overall_chain.invoke(overall_prompt_inputs)

        return summary.model_dump()
