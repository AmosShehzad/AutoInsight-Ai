from pydantic import BaseModel, Field
from typing import List


class ColumnIntelligence(BaseModel):
    column: str = Field(description="Exact column name from the dataset")
    purpose: str = Field(description="One sentence: what this column represents in the business")
    business_interpretation: str = Field(description="One sentence: what the data pattern means for the business")
    risk_note: str = Field(description="One sentence: any risk or caution about this column, or 'None identified.'")


class Opportunity(BaseModel):
    title: str = Field(description="Short opportunity name, e.g. 'Increase TV Budget'")
    recommendation: str = Field(description="What action to take")
    expected_impact: str = Field(description="Plain-language expected result, grounded in the actual numbers")
    confidence: str = Field(description="One of: High, Medium, Low — based on how strong the underlying data signal is")


class Risk(BaseModel):
    title: str = Field(description="Short risk name")
    description: str = Field(description="What the risk is, grounded in actual data")
    priority: str = Field(description="One of: Critical, High, Medium, Low")


class ExecutiveNarrative(BaseModel):
    business_story: str = Field(
        description="A 3-5 sentence narrative paragraph explaining what the data shows, written like a "
                     "management consultant would — connect the numbers into a coherent story, not a list of stats."
    )
    column_intelligence: List[ColumnIntelligence] = Field(
        description="One entry per KEY numeric column (max 5) explaining what it means for the business."
    )
    opportunities: List[Opportunity] = Field(
        description="2-4 genuine opportunities, each grounded in an actual number from the statistics provided."
    )
    risks: List[Risk] = Field(
        description="1-3 real risks, each grounded in an actual number (outliers, high variance, weak correlation, low data quality)."
    )