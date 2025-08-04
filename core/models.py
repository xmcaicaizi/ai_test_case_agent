# core/models.py
"""
该文件定义了项目所需的所有Pydantic数据模型。
"""
from pydantic import BaseModel, Field
from typing import List

class TestCase(BaseModel):
    """
    定义单个测试用例的结构，严格按照指定的表头字段。
    """
    ID: str = Field(description="测试用例的唯一标识符", default="")
    一级模块: str = Field(description="产品的一级模块名称", default="")
    二级模块: str = Field(description="产品的二级模块名称", default="")
    三级模块: str = Field(description="产品的三级模块名称", default="")
    四级模块: str = Field(description="产品的四级模块名称", default="")
    五级模块: str = Field(description="产品的五级模块名称", default="")
    六级模块: str = Field(description="产品的六级模块名称", default="")
    七级模块: str = Field(description="产品的七级模块名称", default="")
    八级模块: str = Field(description="产品的八级模块名称", default="")
    九级模块: str = Field(description="产品的九级模块名称", default="")
    用例名称: str = Field(description="测试用例的简洁标题", default="")
    优先级: str = Field(description="用例的优先级 (例如：高, 中, 低)", default="中")
    用例类型: str = Field(description="用例的类型 (例如：功能测试, 性能测试)", default="功能测试")
    前置条件: str = Field(description="执行测试前需要满足的前置条件", default="")
    步骤描述: str = Field(description="执行测试的具体步骤描述", default="")
    预期结果: str = Field(description="预期的测试结果", default="")
    备注: str = Field(description="其他相关备注信息", default="")
    维护人: str = Field(description="该用例的维护人", default="")
    测试方式: str = Field(description="测试方式 (例如：手动, 自动)", default="手动")
    创建版本: str = Field(description="用例创建时的软件版本", default="")
    更新版本: str = Field(description="用例最后更新时的软件版本", default="")
    评估是否可实现自动化: str = Field(description="评估该用例是否可以自动化 (是, 否, 部分)", default="否")
    是否重新执行: str = Field(description="是否需要重新执行 (是/否)", default="是")
    Summary: str = Field(description="用例的简要总结", default="")
    所属产品: str = Field(description="用例所属的产品线", default="")
    预估执行时间_h: str = Field(description="预估的测试执行时间（小时）", alias="预估执行时间（h）", default="")

class TestCases(BaseModel):
    """
    用于接收包含多个测试用例的列表，方便LLM生成多个用例。
    """
    test_cases: List[TestCase]
