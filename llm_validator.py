import json
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from advanced_rag_agent import get_agent_executor
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)
sys.path.insert(0, project_root)

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "DuoLife-RAG-Validation"
# Initialize the validator LLM (using gpt-4o-mini as requested)
validator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Pydantic models for structured output
class KeywordValidation(BaseModel):
    keyword: str = Field(description="The specific keyword being validated")
    present: bool = Field(description="Whether the keyword is present in the answer")
    confidence: float = Field(description="Confidence score 0-100 for this keyword's presence")

class AnswerValidation(BaseModel):
    question_id: int = Field(description="Index of the question in the QA dataset")
    question: str = Field(description="The original question")
    overall_score: float = Field(description="Overall validation score 0-100")
    keyword_validations: List[KeywordValidation] = Field(description="Validation results for each expected keyword")
    missing_keywords: List[str] = Field(description="List of keywords that were expected but not found")
    extra_information: bool = Field(description="Whether the answer contains relevant information beyond expected keywords")
    compliance_check: bool = Field(description="Whether the answer follows DuoLife guidelines (medical disclaimers, affiliate links, etc.)")

# Validation prompt template
validation_prompt_template = """
You are an expert validator for DuoLife chatbot responses. Your task is to validate if an AI agent's answer contains the expected keywords and follows proper guidelines.

VALIDATION CRITERIA:
1. **Keyword Presence**: Check if each expected keyword is present in the answer (exact match or semantically equivalent)
2. **Semantic Equivalence**: Accept variations like "DuoLife Collagen" vs "Collagen from DuoLife"
3. **Medical Compliance**: Verify medical disclaimers are present for health-related answers
4. **Affiliate Links**: Check if product recommendations include affiliate links as expected
5. **Professional Tone**: Ensure the answer maintains DuoLife's professional, helpful tone

SCORING GUIDELINES:
- Each keyword found = points based on importance
- Product names and links = high importance (20-30 points each)
- Health benefits and ingredients = medium importance (10-15 points each)
- General terms = lower importance (5-10 points each)
- Medical disclaimers and compliance = bonus points
- Missing critical information = significant point deduction

QUESTION: {question}

EXPECTED KEYWORDS: {expected_keywords}

AGENT'S ANSWER: {agent_answer}

VALIDATION INSTRUCTIONS:
1. For each expected keyword, determine if it's present (exactly or semantically)
2. Calculate confidence score (0-100) for each keyword's presence
3. Identify missing keywords that should have been included
4. Check if answer includes helpful extra information beyond keywords
5. Verify compliance with DuoLife guidelines (disclaimers, links, tone)
6. Calculate overall score based on keyword coverage and quality

{format_instructions}
"""

# Create the validation prompt
validation_parser = PydanticOutputParser(pydantic_object=AnswerValidation)
validation_prompt = ChatPromptTemplate.from_template(validation_prompt_template).partial(
    format_instructions=validation_parser.get_format_instructions()
)

class LLMValidator:
    def __init__(self, qa_file_path: str = "QA.json"):
        """Initialize the validator with QA dataset and agent executor."""
        self.qa_file_path = qa_file_path
        self.qa_data = self.load_qa_data()
        self.agent_executor = get_agent_executor()
        self.validation_chain = validation_prompt | validator_llm | validation_parser
        
    def load_qa_data(self) -> List[Dict[str, Any]]:
        """Load the QA dataset from JSON file."""
        try:
            with open(self.qa_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"QA file not found: {self.qa_file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in: {self.qa_file_path}")
    
    def get_agent_answer(self, question: str, session_id: str = "validation_session") -> str:
        """Get answer from the main RAG agent."""
        try:
            response = self.agent_executor.invoke(
                {"question": question},
                config={"configurable": {"session_id": session_id}}
            )
            
            # Extract the conversational answer from the response
            final_answer = response.get("final_answer", {})
            if isinstance(final_answer, dict):
                return final_answer.get("conversational_answer", "No answer generated")
            return str(final_answer)
        
        except Exception as e:
            print(f"Error getting agent answer: {e}")
            return f"Error: {str(e)}"
    
    def validate_single_answer(self, question_id: int, question: str, expected_keywords: List[str], agent_answer: str) -> AnswerValidation:
        """Validate a single answer against expected keywords."""
        try:
            result = self.validation_chain.invoke({
                "question": question,
                "expected_keywords": expected_keywords,
                "agent_answer": agent_answer
            })
            
            # Ensure question_id is set correctly
            result.question_id = question_id
            result.question = question
            
            return result
        
        except Exception as e:
            print(f"Error validating answer for question {question_id}: {e}")
            # Return a fallback validation result
            return AnswerValidation(
                question_id=question_id,
                question=question,
                overall_score=0.0,
                keyword_validations=[],
                missing_keywords=expected_keywords,
                extra_information=False,
                compliance_check=False
            )
    
    def validate_all_questions(self, save_results: bool = True, results_file: str = "validation_results.json") -> List[AnswerValidation]:
        """Validate all questions in the QA dataset."""
        results = []
        total_questions = len(self.qa_data)
        
        print(f"Starting validation of {total_questions} questions...")
        
        for i, qa_item in enumerate(self.qa_data):
            question = qa_item["question"]
            expected_keywords = qa_item["answer_keywords"]
            
            print(f"\nValidating question {i+1}/{total_questions}: {question[:100]}...")
            
            # Get answer from the agent
            agent_answer = self.get_agent_answer(question, f"validation_session_{i}")
            
            # Validate the answer
            validation_result = self.validate_single_answer(i, question, expected_keywords, agent_answer)
            results.append(validation_result)
            
            print(f"Score: {validation_result.overall_score:.1f}/100")
        
        if save_results:
            self.save_results(results, results_file)
        
        return results
    
    def save_results(self, results: List[AnswerValidation], filename: str):
        """Save validation results to JSON file."""
        results_dict = {
            "validation_summary": {
                "total_questions": len(results),
                "average_score": sum(r.overall_score for r in results) / len(results) if results else 0,
                "questions_passed": sum(1 for r in results if r.overall_score >= 70),
                "questions_failed": sum(1 for r in results if r.overall_score < 70)
            },
            "detailed_results": [result.model_dump() for result in results]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\nValidation results saved to: {filename}")
    
    def validate_specific_questions(self, question_indices: List[int]) -> List[AnswerValidation]:
        """Validate specific questions by their indices."""
        results = []
        
        for i in question_indices:
            if i >= len(self.qa_data):
                print(f"Question index {i} is out of range. Skipping.")
                continue
            
            qa_item = self.qa_data[i]
            question = qa_item["question"]
            expected_keywords = qa_item["answer_keywords"]
            
            print(f"\nValidating question {i}: {question[:100]}...")
            
            agent_answer = self.get_agent_answer(question, f"validation_session_{i}")
            validation_result = self.validate_single_answer(i, question, expected_keywords, agent_answer)
            results.append(validation_result)
            
            print(f"Score: {validation_result.overall_score:.1f}/100")
        
        return results
    
    def generate_report(self, results: List[AnswerValidation]) -> Dict[str, Any]:
        """Generate a comprehensive validation report."""
        if not results:
            return {"error": "No validation results provided"}
        
        # Calculate statistics
        scores = [r.overall_score for r in results]
        avg_score = sum(scores) / len(scores)
        
        # Categorize results
        excellent = [r for r in results if r.overall_score >= 90]
        good = [r for r in results if 70 <= r.overall_score < 90]
        needs_improvement = [r for r in results if 50 <= r.overall_score < 70]
        poor = [r for r in results if r.overall_score < 50]
        
        # Find common missing keywords
        all_missing = []
        for r in results:
            all_missing.extend(r.missing_keywords)
        
        missing_keyword_counts = {}
        for keyword in all_missing:
            missing_keyword_counts[keyword] = missing_keyword_counts.get(keyword, 0) + 1
        
        # Sort by frequency
        common_missing = sorted(missing_keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "summary": {
                "total_questions": len(results),
                "average_score": round(avg_score, 2),
                "excellent_count": len(excellent),
                "good_count": len(good),
                "needs_improvement_count": len(needs_improvement),
                "poor_count": len(poor)
            },
            "score_distribution": {
                "excellent (90-100)": len(excellent),
                "good (70-89)": len(good),
                "needs_improvement (50-69)": len(needs_improvement),
                "poor (0-49)": len(poor)
            },
            "common_missing_keywords": common_missing[:10],
            "recommendations": self.generate_recommendations(results)
        }
    
    def generate_recommendations(self, results: List[AnswerValidation]) -> List[str]:
        """Generate improvement recommendations based on validation results."""
        recommendations = []
        
        # Check compliance issues
        compliance_issues = sum(1 for r in results if not r.compliance_check)
        if compliance_issues > len(results) * 0.2:  # More than 20% have compliance issues
            recommendations.append("Improve medical disclaimers and affiliate link inclusion")
        
        # Check for common missing keywords
        all_missing = []
        for r in results:
            all_missing.extend(r.missing_keywords)
        
        if "affiliate link" in str(all_missing).lower():
            recommendations.append("Ensure product recommendations include affiliate links")
        
        if any("medical" in keyword.lower() or "disclaimer" in keyword.lower() for keyword in all_missing):
            recommendations.append("Add medical disclaimers to health-related responses")
        
        # Check average scores
        avg_score = sum(r.overall_score for r in results) / len(results)
        if avg_score < 70:
            recommendations.append("Overall knowledge base needs improvement - consider updating vector store")
        
        return recommendations

# Main execution function
def main():
    """Main function to run the validation."""
    print("Initializing LLM Validator...")
    validator = LLMValidator()
    
    # Option 1: Validate all questions
    print("\nStarting full validation...")
    results = validator.validate_all_questions()
    
    # Generate and display report
    report = validator.generate_report(results)
    print("\n" + "="*50)
    print("VALIDATION REPORT")
    print("="*50)
    print(f"Total Questions: {report['summary']['total_questions']}")
    print(f"Average Score: {report['summary']['average_score']}/100")
    print(f"Excellent (90-100): {report['score_distribution']['excellent (90-100)']}")
    print(f"Good (70-89): {report['score_distribution']['good (70-89)']}")
    print(f"Needs Improvement (50-69): {report['score_distribution']['needs_improvement (50-69)']}")
    print(f"Poor (0-49): {report['score_distribution']['poor (0-49)']}")
    
    if report['common_missing_keywords']:
        print("\nMost Common Missing Keywords:")
        for keyword, count in report['common_missing_keywords'][:5]:
            print(f"  - {keyword}: {count} times")
    
    if report['recommendations']:
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")

if __name__ == "__main__":
    # Option to validate specific questions for testing
    # validator = LLMValidator()
    # results = validator.validate_specific_questions([0, 1, 2])  # Test first 3 questions
    
    # Or run full validation
    main()