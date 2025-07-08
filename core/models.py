# core/models.py
"""
该文件定义了项目所需的所有Pydantic数据模型。
"""
from pydantic import BaseModel, Field
from typing import List

class TestCase(BaseModel):
    """
    定义单个测试用例的结构。
    """
    title: str = Field(description="测试用例的简洁标题")
    preconditions: str = Field(description="执行测试前需要满足的前置条件")
    steps: List[str] = Field(description="执行测试的具体步骤列表")
    expected_result: str = Field(description="预期的测试结果")
    priority: str = Field(description="用例的优先级 (高, 中, 低)")

class TestCases(BaseModel):
    """
    用于接收包含多个测试用例的列表，方便LLM生成多个用例。
    """
    test_cases: List[TestCase]
