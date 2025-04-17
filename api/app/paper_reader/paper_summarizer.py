from operator import itemgetter
from pydantic import BaseModel, Field
from textwrap import dedent
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI


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
        self._section_prompt_template = self._create_section_prompt_template()
        self._overall_prompt_template = self._create_overall_prompt_template()
        self._section_output_parser = StrOutputParser()
        self._overall_output_parser = PydanticOutputParser(pydantic_object=PaperSummary)

    def _create_section_prompt_template(self) -> ChatPromptTemplate:
        """
        Create the prompt template for section summarization.

        Returns:
            ChatPromptTemplate: The formatted prompt template
        """
        system_template = """You are an expert at analyzing sections of academic papers. Your task is to analyze a specific section of a paper and create a focused summary that captures the key information from that section."""

        human_template = dedent(
            """\
            Please analyze the following section of the paper and create a focused summary.

            Section Content:
            {content}

            Please provide a brief summary of this section, focusing on:
            1. Main ideas and arguments presented
            2. Key findings or methodological details
            3. How this section contributes to the overall paper
            4. Any significant equations, results, or conclusions

            Make the summary clear and concise while preserving all important technical details."""
        )

        system_message = SystemMessagePromptTemplate.from_template(system_template)
        human_message = HumanMessagePromptTemplate.from_template(human_template)

        return ChatPromptTemplate.from_messages([system_message, human_message])

    def _create_overall_prompt_template(self) -> ChatPromptTemplate:
        """
        Create the prompt template for paper summarization.

        Returns:
            ChatPromptTemplate: The formatted prompt template
        """
        system_template = """You are an expert at summarizing academic papers. Your task is to analyze academic papers and create comprehensive, well-structured summaries that capture the key aspects of the research."""

        human_template = dedent(
            """\
            Please analyze the following paper and create a comprehensive summary.

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
        )

        system_message = SystemMessagePromptTemplate.from_template(system_template)
        human_message = HumanMessagePromptTemplate.from_template(human_template)

        return ChatPromptTemplate.from_messages([system_message, human_message])

    async def generate_summary(self, paper_data: dict) -> dict:
        """
        Generate a markdown summary of the paper.

        Args:
            paper_data (Dict): Paper data from ArXivPaper.get_paper_data()

        Returns:
            str: Markdown-formatted summary of the paper
        """

        section_chain = {
            f"doc_{i}": (
                itemgetter(f"doc_{i}")
                | RunnableLambda(lambda doc: {"content": doc.page_content})
                | self._section_prompt_template
                | self._llm
                | self._section_output_parser
            )
            for i in range(len(paper_data["documents"]))
        } | RunnableLambda(lambda outputs: "\n\n".join(list(outputs.values())))

        overall_chain = (
            self._overall_prompt_template | self._llm | self._overall_output_parser
        )

        summary = await overall_chain.ainvoke(
            {
                "title": paper_data["details"]["title"],
                "authors": ", ".join(paper_data["details"]["authors"]),
                "abstract": paper_data["details"]["abstract"],
                "section_summaries": await section_chain.ainvoke(
                    {f"doc_{i}": doc for i, doc in enumerate(paper_data["documents"])}
                ),
                "format_instructions": self._overall_output_parser.get_format_instructions(),
            }
        )

        return summary.model_dump()
